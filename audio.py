import os
import json
import torch
import dotenv
from faster_whisper import WhisperModel

dotenv.load_dotenv()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
WHISPER_MODEL_SIZE = "medium"

def run_stt(wav_path, output_dir):
    """
    Wav 파일 경로를 받아서 STT를 수행하고 JSON 경로를 반환합니다.
    (오디오 추출 기능은 preprocess.py로 이관됨)
    """
    if not wav_path or not os.path.exists(wav_path):
        print(f"❌ STT 에러: 오디오 파일이 없습니다 -> {wav_path}")
        return None

    print(f"🚀 [Audio] STT 분석 시작 (Device: {DEVICE})")
    os.makedirs(output_dir, exist_ok=True)
    
    # Whisper 모델 로드
    print(f"📝 Whisper 모델 로딩 중... ({WHISPER_MODEL_SIZE})")
    try:
        model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    except Exception as e:
        print(f"❌ Whisper 모델 로드 실패: {e}")
        return None

    # STT 수행
    print("📝 텍스트 변환 중...")
    segments, info = model.transcribe(wav_path, beam_size=5, language="ko", temperature=0.0)

    stt_results = []
    for i, segment in enumerate(segments):
        stt_results.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip(),
            "speaker": None # Merge 단계에서 채워짐
        })
        if i % 20 == 0:
             print(f"  ... {segment.start:.1f}s: {segment.text[:20]}")

    # JSON 저장
    json_path = os.path.join(output_dir, "stt_result.json")
    final_data = {"transcripts": stt_results}
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"✅ STT 완료! JSON 저장됨: {json_path}")
    return json_path