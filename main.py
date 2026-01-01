import os
import argparse
import preprocess
import ocr
import event  # [추가] 방금 만든 event.py 불러오기
from pathlib import Path

def process_pipeline(video_path):
    # 1. 파일 경로 확인
    if not os.path.exists(video_path):
        print(f"❌ 에러: 파일이 없습니다 -> {video_path}")
        return

    video_name = Path(video_path).stem
    print(f"🎬 [Start] 파이프라인 시작: {video_name}")

    # ====================================================
    # 단계 1: 전처리 (Frame Extraction)
    # ====================================================
    # preprocess.py는 프레임이 저장된 폴더 경로를 반환합니다.
    frames_dir = preprocess.split_video_to_frames(video_path)
    if not frames_dir: return

    # ----------------------------------------------------
    # [경로 설정] output 폴더 자동 계산
    # frame/test1 -> output/test1 경로를 만듭니다.
    # ----------------------------------------------------
    frame_parent = os.path.dirname(frames_dir) # data/frame
    
    if "frame" in frame_parent:
        output_root = frame_parent.replace("frame", "output")
    else:
        output_root = os.path.join(os.path.dirname(frame_parent), "output")
        
    final_output_dir = os.path.join(output_root, video_name)
    os.makedirs(final_output_dir, exist_ok=True)


    # ====================================================
    # 단계 2: OCR (Speaker Detection)
    # ====================================================
    # OCR을 수행하고 result.json을 생성합니다.
    ocr.run_ocr_on_folder(frames_dir)


    # ====================================================
    # 단계 3: Event (Visual Change Detection) - [추가됨]
    # ====================================================
    # 기존 result.json에 '발화 변화', '발표 시작/종료' 정보를 추가합니다.
    event.run_event_detection(video_path, final_output_dir)


    print("=" * 40)
    print(f"🎉 모든 분석이 완료되었습니다!")
    print(f"📂 결과 폴더: {final_output_dir}")
    
    final_json_path = os.path.join(final_output_dir, 'result.json')
    print(f"📄 최종 JSON: {final_json_path}")
    print("=" * 40)
    return final_json_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="비디오 파일 경로")
    args = parser.parse_args()
    
    process_pipeline(args.video)