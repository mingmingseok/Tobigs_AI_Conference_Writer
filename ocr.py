import os
import re
import json
import cv2
import easyocr
from difflib import SequenceMatcher
from collections import Counter

# =========================
# ⚙️ 설정 (상수)
# =========================
EXTRACT_FPS = 2.0    
FRAME_STEP = 1       
USE_GPU = True 
LANG_LIST = ['ko', 'en']

# ROI 좌표 설정
ROI_MAIN_X0, ROI_MAIN_Y0 = 0.00, 0.92
ROI_MAIN_X1, ROI_MAIN_Y1 = 0.35, 0.99
ROI_SIDE_X0, ROI_SIDE_Y0 = 0.75, 0.58
ROI_SIDE_X1, ROI_SIDE_Y1 = 0.85, 0.63

# 전처리/필터 설정
UPSCALE_MAIN = 3    
UPSCALE_SIDE = 4    
MIN_CONF = 0.4      
MERGE_SIM_TH = 0.88
RECORD_UNKNOWN = True
UNKNOWN_LABEL = "UNKNOWN"

# =========================
# 유틸리티 함수
# =========================
def sec_to_mmss(sec: float):
    m = int(sec // 60)
    s = int(sec % 60)
    return f"{m:02d}:{s:02d}"

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def _collapse_korean_spaces(s: str) -> str:
    if not s: return s
    s = re.sub(r"\s+", " ", s).strip()
    hangul_cnt = len(re.findall(r"[가-힣]", s))
    space_cnt = s.count(" ")
    if hangul_cnt >= 3 and space_cnt >= max(2, hangul_cnt - 1):
        s = s.replace(" ", "")
    s = re.sub(r"([가-힣A-Za-z0-9])\s+([가-힣A-Za-z0-9])", r"\1\2", s)
    return s.strip()

def clean_name(t: str) -> str:
    t = (t or "").strip()
    t = re.sub(r"[^0-9A-Za-z가-힣\s]", "", t).strip()
    t = _collapse_korean_spaces(t)
    if len(t) < 2 or len(t) > 40: return ""
    if re.search(r"(.)\1\1\1", t): return ""
    bad = {"발표", "발표중", "화면", "공유", "자막", "미트", "meet", "google", "구글", "프레젠테이션"}
    if t.lower() in bad: return ""
    return t

def crop_roi(img, x0_r, y0_r, x1_r, y1_r):
    h, w = img.shape[:2]
    x0 = int(w * x0_r); x1 = int(w * x1_r)
    y0 = int(h * y0_r); y1 = int(h * y1_r)
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(w, x1); y1 = min(h, y1)
    if x1 <= x0 or y1 <= y0: return None
    return img[y0:y1, x0:x1]

def preprocess_image(img_bgr, scale):
    if img_bgr is None or img_bgr.size == 0: return None
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if scale > 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return gray

def run_ocr_detection(reader, img):
    if img is None: return "", 0.0
    try:
        results = reader.readtext(img, detail=1)
    except:
        return "", 0.0
    best_text = ""
    best_conf = 0.0
    if not results: return "", 0.0
    for (bbox, text, conf) in results:
        if conf < MIN_CONF: continue
        cleaned = clean_name(text)
        if not cleaned: continue
        if conf > best_conf:
            best_conf = conf
            best_text = cleaned
    return best_text, best_conf

# =========================
# 🚀 [수정됨] 인자를 2개 받도록 변경
# =========================
def run_ocr_on_folder(frame_dir, output_dir):
    """
    frame_dir: 이미지가 있는 폴더
    output_dir: 결과 json을 저장할 폴더 (main.py에서 받아옴)
    """
    # 1. 경로 검증
    if not frame_dir or not os.path.exists(frame_dir):
        print(f"❌ OCR 에러: 경로가 없습니다 -> {frame_dir}")
        return

    # [중요] output_dir을 받아서 폴더 생성
    os.makedirs(output_dir, exist_ok=True)

    print(f"▶ EasyOCR 분석 시작... (GPU={USE_GPU})")
    reader = easyocr.Reader(LANG_LIST, gpu=USE_GPU)
    
    # 이미지 파일 리스트업
    image_files = sorted([
        f for f in os.listdir(frame_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ], key=natural_sort_key)

    if not image_files:
        print("❌ 분석할 이미지가 없습니다.")
        return

    print(f"✅ 총 {len(image_files)}개의 프레임 분석 시작")

    name_votes = Counter()
    segments = []
    cur_name = None
    cur_start = None
    cur_last = None

    for i, fname in enumerate(image_files):
        if i % FRAME_STEP != 0: continue
            
        path = os.path.join(frame_dir, fname)
        frame = cv2.imread(path)
        if frame is None: continue

        t_sec = i / EXTRACT_FPS

        # --- OCR Logic ---
        crop_main = crop_roi(frame, ROI_MAIN_X0, ROI_MAIN_Y0, ROI_MAIN_X1, ROI_MAIN_Y1)
        img_main = preprocess_image(crop_main, UPSCALE_MAIN)
        name_main, conf_main = run_ocr_detection(reader, img_main)
        
        crop_side = crop_roi(frame, ROI_SIDE_X0, ROI_SIDE_Y0, ROI_SIDE_X1, ROI_SIDE_Y1)
        img_side = preprocess_image(crop_side, UPSCALE_SIDE)
        name_side, conf_side = run_ocr_detection(reader, img_side)
        
        final_name = ""
        final_conf = 0.0
        source = "" 

        if not name_main and not name_side:
            final_name = UNKNOWN_LABEL
        elif name_main and not name_side:
            final_name = name_main; final_conf = conf_main; source = "MAIN"
        elif not name_main and name_side:
            final_name = name_side; final_conf = conf_side; source = "SIDE"
        else:
            if conf_main >= conf_side:
                final_name = name_main; final_conf = conf_main; source = "MAIN"
            else:
                final_name = name_side; final_conf = conf_side; source = "SIDE"

        if not RECORD_UNKNOWN and final_name == UNKNOWN_LABEL:
            final_name = ""

        # 로그 (50장마다)
        if i % 50 == 0:
            print(f"[{i}/{len(image_files)}] {sec_to_mmss(t_sec)} | {final_name} ({source})")

        if not final_name: continue

        # 투표 및 세그먼트
        seg_name = UNKNOWN_LABEL
        if final_name != UNKNOWN_LABEL:
            merged = None
            for k in list(name_votes.keys()):
                if similar(k, final_name) >= MERGE_SIM_TH:
                    merged = k
                    break
            key = merged if merged else final_name
            name_votes[key] += 1
            seg_name = key

        if cur_name is None:
            cur_name = seg_name; cur_start = t_sec; cur_last = t_sec
        else:
            if seg_name == cur_name:
                cur_last = t_sec
            else:
                segments.append({
                    "name": cur_name,
                    "first_seen": sec_to_mmss(cur_start),
                    "last_seen": sec_to_mmss(cur_last)
                })
                cur_name = seg_name; cur_start = t_sec; cur_last = t_sec

    if cur_name is not None:
        segments.append({
            "name": cur_name,
            "first_seen": sec_to_mmss(cur_start),
            "last_seen": sec_to_mmss(cur_last)
        })

    # [수정됨] 저장 로직: 인자로 받은 output_dir 사용
    out_json = os.path.join(output_dir, "result.json")
    
    result_data = {
        "video_name": os.path.basename(frame_dir),
        "total_frames": len(image_files),
        "segments": segments,
        "votes_ranking": name_votes.most_common()
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"✅ OCR 분석 완료! JSON 저장됨: {out_json}")

# 테스트 실행 시
if __name__ == "__main__":
    TEST_FRAME_DIR = "data/frame/test1"
    TEST_OUTPUT_DIR = "data/output/test1"
    run_ocr_on_folder(TEST_FRAME_DIR, TEST_OUTPUT_DIR)