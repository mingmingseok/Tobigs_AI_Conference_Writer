import os
import io
import json
import shutil
import traceback
from pathlib import Path
from contextlib import redirect_stdout
import gradio as gr
import preprocess
import ocr
import event
import audio


WORKSPACE = "gradio_runs"

def sec_to_ts(sec: float) -> str:
    try:
        sec = float(sec)
    except:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    return f"{m:02d}:{s:05.2f}"

# Merge 관련 함수
def str_time_to_sec(time_str):
    if isinstance(time_str, (int, float)):
        return float(time_str)
    try:
        parts = time_str.split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except:
        pass
    return 0.0

def _bar(pct: float, width: int = 18) -> str:
    
    pct = max(0.0, min(1.0, float(pct)))
    filled = int(round(pct * width))
    return "█" * filled + "░" * (width - filled)

def build_summary_markdown(data):
    video_name = data.get("video_name", "-")
    total_frames = data.get("total_frames", "-")

    transcripts = data.get("transcripts", [])
    events = data.get("events", [])
    votes = data.get("votes_ranking", [])
    segments = data.get("segments", [])

    # 회의 길이
    duration_sec = 0.0
    for t in transcripts:
        duration_sec = max(duration_sec, float(t.get("end", 0) or 0))

    # 참석자 목록
    uniq_names = []
    for seg in segments:
        n = seg.get("name")
        if n and n not in uniq_names:
            uniq_names.append(n)
    attendees = ", ".join(uniq_names) if uniq_names else "-"

    # 참여도
    vote_lines = []
    total_vote = sum(v[1] for v in votes) if votes else 0
    for name, cnt in votes[:10]:
        pct = (cnt / total_vote) if total_vote else 0.0
        bar = _bar(pct, width=18)
        vote_lines.append(f"- {name}: `{bar}` {pct*100:5.1f}%  ({cnt})")
    if not vote_lines:
        vote_lines = ["- (참여도 정보 없음)"]

    # 발표/화면전환 이벤트
    pres = [e for e in events if e.get("type") == "PRESENTATION"]
    pres_lines = []
    for e in pres[:10]:
        ts = e.get("timestamp_fmt") or sec_to_ts(e.get("timestamp_sec", 0))
        pres_lines.append(f" 📺 [{ts}] {e.get('description', 'PRESENTATION')}")
    if not pres_lines:
        pres_lines = ["- (발표/화면전환 이벤트 없음)"]


    md = f"""

### 회의 요약

**영상명:** {video_name}  
**총 프레임:** {total_frames}  
**영상 길이:** {sec_to_ts(duration_sec)}  
**참석자:** {attendees}

### 참석자 참여 비중
{chr(10).join(vote_lines)}

"""
    return md.strip()

def build_timeline_with_events_markdown(data):
    transcripts = sorted(
        data.get("transcripts", []),
        key=lambda x: float(x.get("start", 0) or 0)
    )
    events = sorted(
        data.get("events", []),
        key=lambda x: float(x.get("timestamp_sec", 0) or 0)
    )

    ev_i = 0
    lines = []
    prev_spk = None

    def render_event(ev):
        ts = ev.get("timestamp_fmt") or sec_to_ts(ev.get("timestamp_sec", 0))
        ev_type = ev.get("type", "EVENT")
        desc = ev.get("description", "")
        return (
            f'<div class="event-box">'
            f'<strong>EVENT · {ev_type}</strong> <span class="event-time">[{ts}]</span><br/>'
            f'{desc}'
            f'</div>'
        )

    for t in transcripts:
        t_s = float(t.get("start", 0) or 0)
        t_e = float(t.get("end", 0) or 0)

        # transcript 이전 이벤트
        while ev_i < len(events) and float(events[ev_i].get("timestamp_sec", 0) or 0) < t_s:
            lines.append(render_event(events[ev_i]))
            ev_i += 1

        text = (t.get("text", "") or "").strip()
        if not text:
            continue

        spk = t.get("speaker", "Unknown") or "Unknown"
        start = sec_to_ts(t_s)
        end = sec_to_ts(t_e)

        if spk != prev_spk:
            if prev_spk is not None:
                lines.append("\n---\n")
            lines.append(f"### {spk}")
            prev_spk = spk

        # 발화
        lines.append(f"- [{start}–{end}] {text}")

        # transcript 내부 이벤트
        while ev_i < len(events) and t_s <= float(events[ev_i].get("timestamp_sec", 0) or 0) <= t_e:
            lines.append(render_event(events[ev_i]))
            ev_i += 1

    # 마지막 transcripts 이후 이벤트
    while ev_i < len(events):
        lines.append(render_event(events[ev_i]))
        ev_i += 1

    return "\n".join(lines) if lines else "(표시할 내용이 없습니다)"




def build_event_choices(data):
    events = sorted(data.get("events", []), key=lambda x: float(x.get("timestamp_sec", 0) or 0))
    choices = []
    for idx, e in enumerate(events):
        ts = e.get("timestamp_fmt") or sec_to_ts(e.get("timestamp_sec", 0))
        choices.append(f"{idx:02d} | [{ts}] {e.get('type')} | {e.get('description','')}")
    return choices, events

def resolve_event_image_path(final_output_dir, image_path):
    if not image_path:
        return None
    cand = os.path.join(final_output_dir, "event_frames", image_path)
    return cand if os.path.exists(cand) else None


# Merge 관련 함수
def merge_vision_and_audio(vision_json_path, audio_json_path, output_json_path):
    with open(vision_json_path, "r", encoding="utf-8") as f:
        vision_data = json.load(f)
    with open(audio_json_path, "r", encoding="utf-8") as f:
        audio_data = json.load(f)

    v_segs = vision_data.get("segments", [])
    for v in v_segs:
        v["_s"] = str_time_to_sec(v.get("first_seen", 0))
        v["_e"] = str_time_to_sec(v.get("last_seen", 0))

    for stt in audio_data.get("transcripts", []):
        s_start, s_end = stt.get("start", 0), stt.get("end", 0)
        best_spk = "Unknown"
        max_overlap = 0.0

        for v in v_segs:
            ov_s = max(s_start, v["_s"])
            ov_e = min(s_end, v["_e"])
            dur_ = max(0.0, ov_e - ov_s)
            if dur_ > max_overlap:
                max_overlap = dur_
                best_spk = v.get("name", "Unknown")

        stt["speaker"] = best_spk

    for v in v_segs:
        del v["_s"], v["_e"]

    vision_data["transcripts"] = audio_data.get("transcripts", [])

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(vision_data, f, ensure_ascii=False, indent=2)

    return output_json_path



def process_pipeline(video_path: str):
    if not video_path or not os.path.exists(video_path):
        return None, None

    video_name = Path(video_path).stem

    base_dir = os.path.dirname(video_path)
    output_root = os.path.join(os.path.dirname(base_dir), "output")
    final_output_dir = os.path.join(output_root, video_name)
    os.makedirs(final_output_dir, exist_ok=True)

    wav_path = preprocess.extract_audio(video_path)
    frames_dir = preprocess.split_video_to_frames(video_path)
    if not frames_dir or not wav_path:
        return None, final_output_dir

    ocr.run_ocr_on_folder(frames_dir, final_output_dir)
    event.run_event_detection(video_path, final_output_dir)

    vision_json = os.path.join(final_output_dir, "result.json")
    stt_json = audio.run_stt(wav_path, final_output_dir)

    if os.path.exists(vision_json) and stt_json and os.path.exists(stt_json):
        final_json = merge_vision_and_audio(vision_json, stt_json, vision_json)
        return final_json, final_output_dir

    return None, final_output_dir



def ui_start():
    return (
        gr.update(value="회의록 생성중...", visible=True),
        gr.update(interactive=False),
    )

def ui_run(video_file):
    if video_file is None:
        msg = "영상을 업로드해주세요."
        return (
            msg,
            msg,
            [],
            gr.update(choices=[], value=None),
            None,
            gr.update(value=msg, visible=True),
            gr.update(interactive=True),
            None,
            [],
        )

    os.makedirs(WORKSPACE, exist_ok=True)

    src = video_file
    dst = os.path.join(WORKSPACE, Path(src).name)
    shutil.copy(src, dst)

    try:
        with redirect_stdout(io.StringIO()):
            final_json, out_dir = process_pipeline(dst)

        if not final_json or not os.path.exists(final_json):
            msg = "결과 생성에 실패했습니다."
            return (
                msg, msg, [], gr.update(choices=[], value=None), None,
                gr.update(value=msg, visible=True),
                gr.update(interactive=True),
                None,
                [],
            )

        with open(final_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        summary_md = build_summary_markdown(data)
        timeline_md = build_timeline_with_events_markdown(data)

        choices, events_list = build_event_choices(data)

        rows = []
        for e in events_list:
            rows.append([
                e.get("timestamp_fmt") or sec_to_ts(e.get("timestamp_sec", 0)),
                e.get("type", ""),
                e.get("description", ""),
                e.get("diff_score", ""),
            ])

        dd_value = choices[0] if choices else None
        img = None
        if choices:
            img = resolve_event_image_path(out_dir, events_list[0].get("image_path"))

        return (
            summary_md,
            timeline_md,
            rows,
            gr.update(choices=choices, value=dd_value),
            img,
            gr.update(value="회의록 생성 완료", visible=True),
            gr.update(interactive=True),
            out_dir,
            events_list,
        )

    except Exception:
        err = f"실행 중 오류가 발생했습니다.\n\n```{traceback.format_exc()}```"
        return (
            err, err, [], gr.update(choices=[], value=None), None,
            gr.update(value="오류 발생", visible=True),
            gr.update(interactive=True),
            None,
            [],
        )

def update_event_image(choice, out_dir, events_list):
    if not choice or not out_dir or not events_list:
        return None
    try:
        idx = int(choice.split("|")[0].strip())
    except:
        return None
    if idx < 0 or idx >= len(events_list):
        return None
    return resolve_event_image_path(out_dir, events_list[idx].get("image_path"))



CSS = """

.gradio-tabs,
.prose {
  background: #ffffff !important;  
}
.gradio-tabs .tab-nav {
  background: #ffffff !important;   
  border-bottom: 1px solid #e5e7eb;
}
.gradio-tabs .tab-nav button {
  background: transparent;
  font-weight: 600;
}

.gradio-container {
  max-width: 1100px !important;
  margin: 0 auto !important;
}

.gradio-group > .label,
.gradio-group > .header {
  background: transparent !important;
  padding-bottom: 0 !important;
}

.video-big {
  min-height: 340px !important;  
}

.video-big video {
  max-height: 340px !important;   
}

.video-big .wrap {
  min-height: 340px !important;
  max-height: 340px !important;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}

.panel {
  border: 1px solid rgba(0,0,0,0.08);
  border-radius: 14px;
  padding: 14px;
  background: #f0f7ff;
}

.event-box {
  background: #fff6e5;
  border-left: 4px solid #ff9800;
  padding: 10px 12px;
  margin: 8px 0 12px 0;
  border-radius: 8px;
  font-size: 0.9em;
}

.event-time {
  color: #666;
  font-size: 0.85em;
}

"""

with gr.Blocks(title="Meeting Summary Writer") as demo:
    gr.Markdown("### Meeting Summary Writer")

    state_out_dir = gr.State(value=None)
    state_events = gr.State(value=[])

    with gr.Row():
        with gr.Column(scale=4, min_width=340):
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("### 회의 영상 업로드")
                video_in = gr.Video(label="Video", sources=["upload"], elem_classes=["video-big"])
                run_btn = gr.Button("회의 요약", variant="primary")
                status_md = gr.Markdown(value="", visible=False)

        with gr.Column(scale=8, min_width=520):
            with gr.Group(elem_classes=["panel"]):
                gr.Markdown("### 회의 요약 결과 보기")
            with gr.Tabs(elem_id="result_tabs"):

                with gr.Tab("요약"):
                    summary_out = gr.Markdown(value=" ")
                with gr.Tab("타임라인"):
                    timeline_out = gr.Markdown(value=" ")
                with gr.Tab("이벤트"):
                    event_dd = gr.Dropdown(label="이벤트 선택", choices=[], value=None)
                    event_img = gr.Image(label="이벤트 스냅샷", type="filepath", height=340)
                    event_table = gr.Dataframe(
                        headers=["time", "type", "description", "diff_score"],
                        datatype=["str", "str", "str", "number"],
                        row_count=8,
                        column_count=(4, "fixed"),
                        wrap=True,
                        label="이벤트 목록",
                    )

    run_btn.click(
        fn=ui_start,
        inputs=[],
        outputs=[status_md, run_btn],
        queue=False,
    ).then(
        fn=ui_run,
        inputs=[video_in],
        outputs=[summary_out, timeline_out, event_table, event_dd, event_img, status_md, run_btn, state_out_dir, state_events],
        queue=True,
    )

    event_dd.change(
        fn=update_event_image,
        inputs=[event_dd, state_out_dir, state_events],
        outputs=[event_img],
        queue=False,
    )

if __name__ == "__main__":
    demo.launch(css=CSS)
