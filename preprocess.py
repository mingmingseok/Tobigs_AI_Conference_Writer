# preprocess.py
import cv2
import os
from pathlib import Path

def split_video_to_frames(video_path):
    # 1. 경로 및 설정 자동화
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem
    
    parent_dir = str(video_path_obj.parent)
    if "video" in parent_dir:
        output_root = parent_dir.replace("video", "frame")
    else:
        output_root = os.path.join(parent_dir, "frame")
        
    output_dir = os.path.join(output_root, video_name)

    # 2. 비디오 열기
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 에러: 동영상을 열 수 없습니다: {video_path}")
        return None  # 에러 시 None 반환

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    # (참고: 기존 로직 유지 - 0.5초 간격)
    interval = round(fps / 2) 
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"🎥 전처리 시작: {video_name}")
    print(f"📂 저장 경로: {output_dir}")

    frame_index = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if frame_index % interval == 0:
            filename = f"frame_{str(saved_count).zfill(4)}.jpg"
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, frame)
            saved_count += 1
            
        frame_index += 1

    cap.release()
    print(f"🎉 전처리 완료! ({saved_count}장 저장됨)")
    
    # [중요] 저장된 폴더 경로를 반환하여 main.py에서 쓸 수 있게 함
    return output_dir 

# 이 파일만 단독으로 실행할 때만 동작
if __name__ == "__main__":
    TEST_VIDEO = "data/video/test1.mp4"
    split_video_to_frames(TEST_VIDEO)