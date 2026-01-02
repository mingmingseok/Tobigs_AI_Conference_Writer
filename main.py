import os
import argparse
import json
import preprocess  # (수정된 버전)
import ocr
import event
import audio       # (수정된 버전)
from pathlib import Path

# --- (Merge 관련 함수는 그대로 유지: str_time_to_sec, merge_vision_and_audio) ---
def str_time_to_sec(time_str):
    if isinstance(time_str, (int, float)): return float(time_str)
    try:
        parts = time_str.split(':')
        if len(parts) == 2: return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except: pass
    return 0.0

def merge_vision_and_audio(vision_json_path, audio_json_path, output_json_path):
    # ... (아까 작성해드린 병합 로직과 동일) ...
    # 코드가 길어지니 생략합니다. 위 답변의 merge 함수를 그대로 쓰시면 됩니다.
    # 핵심: STT 시간대와 가장 많이 겹치는 Vision 화자 이름을 매칭
    
    print(f"🔄 데이터 병합 중...")
    with open(vision_json_path, 'r', encoding='utf-8') as f: vision_data = json.load(f)
    with open(audio_json_path, 'r', encoding='utf-8') as f: audio_data = json.load(f)
    
    v_segs = vision_data.get("segments", [])
    for v in v_segs:
        v["_s"] = str_time_to_sec(v.get("first_seen", 0))
        v["_e"] = str_time_to_sec(v.get("last_seen", 0))
        
    for stt in audio_data.get("transcripts", []):
        s_start, s_end = stt["start"], stt["end"]
        best_spk = "Unknown"
        max_overlap = 0.0
        
        for v in v_segs:
            # Overlap 계산
            ov_s = max(s_start, v["_s"])
            ov_e = min(s_end, v["_e"])
            dur = max(0, ov_e - ov_s)
            if dur > max_overlap:
                max_overlap = dur
                best_spk = v["name"]
        stt["speaker"] = best_spk
        
    for v in v_segs: del v["_s"], v["_e"]
    
    vision_data["transcripts"] = audio_data.get("transcripts", [])
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(vision_data, f, ensure_ascii=False, indent=2)
    return output_json_path

# ==========================================
# [Main] 파이프라인
# ==========================================
def process_pipeline(video_path):
    if not os.path.exists(video_path): return None
    video_name = Path(video_path).stem
    print(f"🎬 [Project] 분석 시작: {video_name}")

    # 1. 공통 출력 경로 계산
    # data/video/test1.mp4 -> data/output/test1
    base_dir = os.path.dirname(video_path)
    output_root = os.path.join(os.path.dirname(base_dir), "output")
    final_output_dir = os.path.join(output_root, video_name)
    os.makedirs(final_output_dir, exist_ok=True)

    # =================================================
    # [Step 1] 전처리 (Preprocess) - 오디오 & 프레임 추출
    # =================================================
    # preprocess 모듈이 두 가지 일을 다 처리합니다.
    print("--- [1단계] 전처리 ---")
    
    # 1-1. 오디오 추출 (.wav)
    wav_path = preprocess.extract_audio(video_path)
    
    # 1-2. 프레임 추출 (jpg 폴더)
    frames_dir = preprocess.split_video_to_frames(video_path)
    
    if not frames_dir or not wav_path:
        print("❌ 전처리 실패로 중단합니다.")
        return None

    # =================================================
    # [Step 2] Vision 분석 (OCR + Event)
    # =================================================
    print("--- [2단계] 비전 분석 ---")
    ocr.run_ocr_on_folder(frames_dir, final_output_dir)
    event.run_event_detection(video_path, final_output_dir)
    
    vision_json = os.path.join(final_output_dir, "result.json")

    # =================================================
    # [Step 3] Audio 분석 (STT)
    # =================================================
    print("--- [3단계] 오디오 분석 ---")
    # 전처리된 wav_path를 넘겨줍니다.
    stt_json = audio.run_stt(wav_path, final_output_dir)

    # =================================================
    # [Step 4] 병합 (Merge)
    # =================================================
    print("--- [4단계] 데이터 통합 ---")
    if os.path.exists(vision_json) and stt_json:
        final_result = merge_vision_and_audio(vision_json, stt_json, vision_json)
        
        print("="*40)
        print(f"🎉 모든 과정 완료!")
        print(f"📄 최종 결과: {final_result}")
        print("="*40)
        return final_result
    else:
        print("❌ 병합 실패: 파일 누락")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()
    process_pipeline(args.video)