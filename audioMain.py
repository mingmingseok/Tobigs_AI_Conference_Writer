# =====================================================
# [통합 파이프라인]
# 1. Video(.mp4) -> Audio(.wav) 변환
# 2. 화자 분리 (VAD + Embedding + Clustering)
# 3. STT (Faster-Whisper)
# =====================================================

import os
import sys
import numpy as np
import torch
import torchaudio
import dotenv

from typing import List, Tuple
from pyannote.audio import Pipeline
from sklearn.cluster import AgglomerativeClustering
from speechbrain.inference.speaker import EncoderClassifier
from faster_whisper import WhisperModel

# === MoviePy 호환성 처리 ===
try:
    from moviepy.editor import VideoFileClip
except ImportError:
    from moviepy.video.io.VideoFileClip import VideoFileClip


# =====================================================
# 0) 통합 설정 (Configuration)
# =====================================================
dotenv.load_dotenv()

# [경로 설정] Docker 환경 기준 (/root/)
# 윈도우라면: r"C:\Users\win10\Desktop\source\tobigs_last\zoom_video.mp4"  ,, "/root/zoom_video.mp4"
VIDEO_PATH = r"C:\Users\win10\Desktop\source\tobigs_last\test1.mp4"
WAV_PATH   = VIDEO_PATH.replace(".mp4", ".wav")  # 자동 생성될 오디오 경로

# [토큰 및 모델 설정]
HF_TOKEN = os.environ.get("HF_TOKEN")  # .env 파일 혹은 직접 입력
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# [분석 파라미터]
TARGET_SR = 16000
N_SPEAKERS = 3          # 화자 수 (고정)
VAD_MIN_DUR = 0.4
MIN_SEG_DUR = 0.8
MERGE_GAP = 0.3
CHUNK_SEC = 1.5
STRIDE_SEC = 0.75

# [Whisper 설정]
WHISPER_MODEL_SIZE = "medium" # large-v3
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"


# =====================================================
# 1단계: 동영상을 오디오로 추출 (audio.py 로직)
# =====================================================
def extract_audio_from_video(vid_path, aud_path):
    print("\n" + "="*50)
    print(" [STEP 1] 비디오에서 오디오 추출")
    print("="*50)
    
    if not os.path.exists(vid_path):
        # 만약 비디오가 없고, 이미 변환된 wav파일만 있다면 통과
        if os.path.exists(aud_path):
            print(f"비디오 파일이 없지만, 오디오 파일({aud_path})이 존재하여 이를 사용합니다.")
            return True
        else:
            print(f"[오류] 파일을 찾을 수 없습니다: {vid_path}")
            return False

    try:
        print(f"▶ 변환 시작: {vid_path} -> {aud_path}")
        video = VideoFileClip(vid_path)
        video.audio.write_audiofile(aud_path, codec='pcm_s16le', logger=None)
        video.close()
        print(f"[성공] 오디오 추출 완료!")
        return True
    except Exception as e:
        print(f"[실패] 오디오 추출 중 오류 발생:\n{e}")
        return False


# =====================================================
# 2단계: 화자 분리 및 STT 수행 (1223.py 로직)
# =====================================================
def run_diarization_and_stt(audio_path):
    print("\n" + "="*50)
    print(" [STEP 2] 화자 분리 및 STT 시작")
    print("="*50)

    # --- 1) 오디오 로드 ---
    print("1) 오디오 로드 및 리샘플링(16k)...")
    waveform, sr = torchaudio.load(audio_path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        waveform = torchaudio.functional.resample(waveform, sr, TARGET_SR)
        sr = TARGET_SR
    
    audio_np = waveform.squeeze(0).numpy().astype(np.float32)

    # --- 2) VAD ---
    print("2) VAD(음성 감지) 수행 중...")
    try:
        vad = Pipeline.from_pretrained("pyannote/voice-activity-detection", use_auth_token=HF_TOKEN)
    except TypeError:
        # 최신 버전 호환성 (token vs use_auth_token)
        vad = Pipeline.from_pretrained("pyannote/voice-activity-detection", token=HF_TOKEN)

    vad_out = vad({"waveform": waveform, "sample_rate": sr})
    speech_segments = []
    for seg in vad_out.get_timeline().support():
        s, e = float(seg.start), float(seg.end)
        if (e - s) >= VAD_MIN_DUR:
            speech_segments.append((s, e))
    speech_segments.sort(key=lambda x: x[0])

    if not speech_segments:
        print("[종료] VAD가 말 구간을 감지하지 못했습니다.")
        return

    # --- 3) Embedding 준비 ---
    print("3) Speaker Embedding 추출 중...")
    spk_model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE}
    )

    @torch.no_grad()
    def ecapa_embedding(seg_wav):
        wav_t = torch.from_numpy(seg_wav).float().unsqueeze(0).to(DEVICE)
        emb = spk_model.encode_batch(wav_t).squeeze()
        emb = torch.nn.functional.normalize(emb, dim=0)
        return emb.detach().cpu().numpy()

    def segment_to_embedding(s, e):
        # 긴 구간은 잘라서 평균, 짧은 구간은 그대로
        dur = e - s
        if dur <= CHUNK_SEC:
            chunks = [(s, e)]
        else:
            chunks = []
            t = s
            while t + CHUNK_SEC <= e:
                chunks.append((t, t + CHUNK_SEC))
                t += STRIDE_SEC
            if not chunks: chunks = [(s, e)]
        
        embs = []
        for cs, ce in chunks:
            s_i, e_i = int(cs * sr), int(ce * sr)
            seg = audio_np[s_i:e_i]
            if len(seg) < int(MIN_SEG_DUR * sr): continue
            embs.append(ecapa_embedding(seg))
        
        if not embs:
            # fallback
            s_i, e_i = int(s * sr), int(e * sr)
            return ecapa_embedding(audio_np[s_i:e_i])
        
        final_emb = np.mean(np.vstack(embs), axis=0)
        return final_emb / (np.linalg.norm(final_emb) + 1e-12)

    # --- 4) Embedding 수행 ---
    valid_segments = []
    embeddings = []
    for s, e in speech_segments:
        if (e - s) < MIN_SEG_DUR: continue
        embeddings.append(segment_to_embedding(s, e))
        valid_segments.append((s, e))

    if len(valid_segments) < N_SPEAKERS:
        print(f"[오류] 감지된 세그먼트 수({len(valid_segments)})가 설정한 화자 수({N_SPEAKERS})보다 적습니다.")
        return

    # --- 5) Clustering ---
    print("4) 화자 클러스터링 중...")
    clustering = AgglomerativeClustering(
        n_clusters=N_SPEAKERS, metric="cosine", linkage="average"
    )
    labels = clustering.fit_predict(np.vstack(embeddings))

    # --- 6) Mapping & Merging ---
    pred = sorted(zip(valid_segments, labels), key=lambda x: x[0][0])
    
    seen = {}
    next_id = 0
    merged = []
    
    # 1차 매핑
    temp_mapped = []
    for (s, e), lb in pred:
        if lb not in seen:
            seen[lb] = next_id
            next_id += 1
        temp_mapped.append([s, e, f"SPEAKER_{seen[lb]:02d}"])
        
    # 세그먼트 병합 (Merge)
    if temp_mapped:
        curr_s, curr_e, curr_spk = temp_mapped[0]
        for s, e, spk in temp_mapped[1:]:
            if spk == curr_spk and (s - curr_e) <= MERGE_GAP:
                curr_e = max(curr_e, e)
            else:
                merged.append((float(curr_s), float(curr_e), curr_spk))
                curr_s, curr_e, curr_spk = s, e, spk
        merged.append((float(curr_s), float(curr_e), curr_spk))

    # --- 7) Whisper STT ---
    print(f"5) Whisper STT 수행 ({WHISPER_MODEL_SIZE})...")
    try:
        whisper = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"[오류] Whisper 로드 실패: {e}")
        return

    final_results = []
    total = len(merged)
    print("   >>> 텍스트 변환 시작...")

    for idx, (s, e, spk) in enumerate(merged, 1):
        s_idx = int(s * sr)
        e_idx = int(e * sr)
        
        if e_idx - s_idx < int(0.1 * sr): continue
        
        seg_audio = audio_np[s_idx:e_idx]
        segments_gen, _ = whisper.transcribe(
            seg_audio, beam_size=5, language="ko", temperature=0.0
        )
        
        text = " ".join([sg.text for sg in segments_gen]).strip()
        final_results.append((s, e, spk, text))
        print(f"[{idx}/{total}] {s:.1f}s ~ {e:.1f}s ({spk}): {text}")

    # --- 8) 저장 ---
    output_txt = "result_script.txt"
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("Start(s)\tEnd(s)\tSpeaker\tText\n")
        for s, e, spk, txt in final_results:
            f.write(f"[{s:6.2f} - {e:6.2f}] {spk} : {txt}\n")
    
    print("\n모든 작업이 완료되었습니다.")
    print(f"결과 파일: {output_txt}")


# =====================================================
# Main Execution
# =====================================================
if __name__ == "__main__":
    # 1. 오디오 추출
    success = extract_audio_from_video(VIDEO_PATH, WAV_PATH)
    
    # 2. 분석 실행 (추출 성공 시)
    if success:
        run_diarization_and_stt(WAV_PATH)