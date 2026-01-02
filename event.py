import cv2
import os
import json
from pathlib import Path

# ==========================================
# [설정] 변화 감지 임계값
# ==========================================
THRESH_SPEECH = 2.0      # 예: 제스처, 고개 끄덕임 등 (발화 변화)
THRESH_EVENT = 40.0      # 예: PPT 전환, 화면 공유 등 (발표 시작/종료)

def sec_to_mmss(sec):
    """초 단위를 mm:ss 형식 문자열로 변환"""
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

def run_event_detection(video_path, output_dir):
    """
    비디오 변화를 분석하여 result.json에 이벤트 정보를 추가합니다.
    """
    if not os.path.exists(video_path):
        print(f"❌ [Event] 비디오 경로 오류: {video_path}")
        return

    # 1. 저장할 폴더 및 JSON 경로 설정
    # 이미지가 저장될 폴더 (증거 자료)
    event_img_dir = os.path.join(output_dir, "event_frames")
    os.makedirs(event_img_dir, exist_ok=True)
    
    # 업데이트할 JSON 파일
    json_path = os.path.join(output_dir, "result.json")

    # 2. 비디오 분석 시작
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0
    
    # 0.5초 간격으로 분석 (속도 최적화)
    interval = max(1, round(fps / 2)) 
    
    prev_frame = None
    frame_index = 0
    
    detected_events = [] # JSON에 들어갈 리스트

    print(f"🔍 [Event] 시각적 변화 분석 시작... (기준: {THRESH_SPEECH} / {THRESH_EVENT})")
    
    while True:
        ret, frame = cap.read()
        if not ret: break

        if frame_index % interval == 0:
            # 연산량 감소를 위해 흑백 + 리사이즈
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.resize(curr_gray, (640, 360))

            if prev_frame is not None:
                # 변화량 계산 (절대 차이의 평균)
                diff = cv2.absdiff(curr_gray, prev_frame)
                diff_score = diff.mean()

                timestamp = frame_index / fps
                event_info = None
                
                # ------------------------------------------------
                # [핵심 로직] 변화량에 따른 분류
                # ------------------------------------------------
                if diff_score >= THRESH_EVENT:
                    event_info = {
                        "type": "PRESENTATION",
                        "description": "발표 시작/종료/화면전환",
                        "priority": "HIGH"
                    }
                elif diff_score >= THRESH_SPEECH:
                    event_info = {
                        "type": "SPEECH_MOTION",
                        "description": "발화 제스처/움직임",
                        "priority": "LOW"
                    }

                # 이벤트가 감지되었다면 저장
                if event_info:
                    # 1) 증거 이미지 저장
                    filename = f"ev_{timestamp:.1f}s_{event_info['type']}.jpg"
                    save_path = os.path.join(event_img_dir, filename)
                    cv2.imwrite(save_path, frame)
                    
                    # 2) 리스트에 데이터 추가 (요청하신 시간, 타입 포함)
                    detected_events.append({
                        "timestamp_sec": round(timestamp, 2),    # 초 단위 (DB 저장용)
                        "timestamp_fmt": sec_to_mmss(timestamp), # 보기 편한 mm:ss
                        "type": event_info["type"],              # 분류 코드
                        "description": event_info["description"],# 한글 설명
                        "diff_score": round(diff_score, 2),      # 변화량 수치
                        "image_path": filename                   # 저장된 이미지 파일명
                    })
                    
                    # (선택) 중요 이벤트만 로그 출력
                    if event_info["priority"] == "HIGH":
                        print(f"  ⚠️ {sec_to_mmss(timestamp)} : {event_info['description']} 감지! (Score: {diff_score:.1f})")

            prev_frame = curr_gray
        
        frame_index += 1

    cap.release()

    # 3. JSON 파일 업데이트 (Merge)
    final_data = {}
    
    # 기존 JSON 로드
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                final_data = json.load(f)
            except:
                print("⚠️ 기존 JSON 파싱 실패, 새로 작성합니다.")

    # 이벤트 데이터 추가
    final_data["events"] = detected_events
    
    # 파일 다시 쓰기
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    print(f"✅ [Event] 분석 완료. {len(detected_events)}개의 이벤트가 추가되었습니다.")
    print(f"📂 업데이트된 파일: {json_path}")

if __name__ == "__main__":
    # 테스트용 코드
    TEST_VIDEO = "data/video/test1.mp4"
    TEST_OUTPUT = "data/output/test1"
    run_event_detection(TEST_VIDEO, TEST_OUTPUT)