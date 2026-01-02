import cv2
import os
from pathlib import Path

# === MoviePy 호환성 처리 ===
try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        from moviepy.video.io.VideoFileClip import VideoFileClip

def extract_audio(video_path):
    """
    비디오에서 오디오(.wav)를 추출합니다.
    저장 경로: data/audio/{video_name}.wav
    """
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem
    
    # data/video/test1.mp4 -> data/audio/
    parent_dir = str(video_path_obj.parent)
    if "video" in parent_dir:
        output_root = parent_dir.replace("video", "audio")
    else:
        output_root = os.path.join(parent_dir, "../audio")
        
    os.makedirs(output_root, exist_ok=True)
    audio_output_path = os.path.join(output_root, f"{video_name}.wav")
    
    # 이미 있으면 건너뛰기
    if os.path.exists(audio_output_path):
        print(f"🔊 [Preprocess] 기존 오디오 파일 사용: {audio_output_path}")
        return audio_output_path

    print(f"🎵 [Preprocess] 오디오 추출 중... -> {audio_output_path}")
    try:
        # 16000Hz는 Whisper 모델이 가장 좋아하는 주파수입니다.
        clip = VideoFileClip(video_path)
        if clip.audio is not None:
            clip.audio.write_audiofile(audio_output_path, codec='pcm_s16le', fps=16000, logger=None)
            clip.close()
            return audio_output_path
        else:
            print("⚠️ 경고: 오디오 트랙이 없는 비디오입니다.")
            return None
    except Exception as e:
        print(f"❌ 오디오 추출 실패: {e}")
        return None

def split_video_to_frames(video_path):
    """
    비디오에서 프레임 이미지를 추출합니다.
    저장 경로: data/frame/{video_name}/
    """
    video_path_obj = Path(video_path)
    video_name = video_path_obj.stem
    
    # data/video/test1.mp4 -> data/frame/test1/
    parent_dir = str(video_path_obj.parent)
    if "video" in parent_dir:
        output_root = parent_dir.replace("video", "frame")
    else:
        output_root = os.path.join(parent_dir, "frame")
        
    output_dir = os.path.join(output_root, video_name)

    # 이미지를 너무 많이 뽑으면 느리므로 이미 폴더가 꽉 차있으면 스킵할 수도 있음 (선택사항)
    # 여기서는 덮어쓰기 로직 유지
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 에러: 동영상을 열 수 없습니다: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30
    
    # 2 FPS (0.5초 간격)
    interval = round(fps / 2) 
    
    print(f"🎥 [Preprocess] 프레임 추출 중... -> {output_dir}")
    
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
    print(f"✅ 프레임 추출 완료 ({saved_count}장)")
    
    return output_dir

if __name__ == "__main__":
    # 테스트
    TEST_VIDEO = "data/video/test1.mp4"
    extract_audio(TEST_VIDEO)
    split_video_to_frames(TEST_VIDEO)