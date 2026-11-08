"""
vision.py - 카메라로 공부 감시, 이벤트가 생기면 JSON
키: q 종료 / space 질문 (손바닥 펴고 1초 유지해도 질문)
"""
import json
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO


# ===================== 설정 =====================
JETSON = True # 젯슨에서 실행할 땐 True

YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine" if JETSON else "src/models/YOLO/yolo11n.pt"
HAND_MODEL_PATH = "src/models/MediaPipe/hand_landmarker.task"
EVENT_JSON_PATH = "src/output/event.json"
ROI_IMAGE_PATH = "src/output/roi.jpg"

PERSON, CELL_PHONE = 0, 67   # COCO 클래스 번호
LIMIT_SECONDS = {"phone": 3, "away": 3, "dark": 3}   # 이 시간 이상 계속되면 이벤트
DARK_THRESHOLD = 70          # HSV 밝기(V) 지적 평균 기준
PALM_FRAMES = 15             # 손바닥 펴서 질문하는 기준(1초)


# ===================== 손가락 질문 확인 (MediaPipe) =====================
finger_angle_points = ((1, 2, 3), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19))


def calculate_angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y, p1.z - p2.z])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y, p3.z - p2.z])
    cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) # 범위 클립 -> 각도 계산 -> 도로 변환


def count_fingers(hand):
    return sum(calculate_angle(hand[a], hand[b], hand[c]) >= 160 for a, b, c in finger_angle_points)
# 160 이상 -> 거의 펴짐 -> 보자기


# ===================== 이벤트 =====================
start_time = time.time()
condition_start = {"phone": None, "away": None, "dark": None}   # 조건 시작 시간
event_count = {"phone": 0, "away": 0, "dark": 0, "question": 0}


def check_event(name, condition):
    """condition 이 LIMIT_SECONDS 이상 계속되면 True"""
    now = time.time()
    if not condition:
        condition_start[name] = None
        return False
    if condition_start[name] is None:
        condition_start[name] = now
    if now - condition_start[name] >= LIMIT_SECONDS[name]:
        condition_start[name] = now   # 다시 0초부터 세기
        return True
    return False


def save_event(event_type, roi_image):
    event_count[event_type] += 1
    cv2.imwrite(ROI_IMAGE_PATH, roi_image)
    event = {
        "type": event_type,                      # phone / away / dark / question
        "seconds": LIMIT_SECONDS.get(event_type, 0),
        "count": event_count,
        "study_minutes": round((time.time() - start_time) / 60, 1),
        "roi_path": ROI_IMAGE_PATH,
    }
    with open(EVENT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(event, f, ensure_ascii=False, indent=4)
    print("[이벤트]", event)


# ===================== 모델, 카메라 =====================
yolo = YOLO(YOLO_MODEL_PATH, task="detect")

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=HAND_MODEL_PATH), num_hands=1)
hand_detector = vision.HandLandmarker.create_from_options(options)

if JETSON:   # 02: 젯슨 GStreamer 파이프라인
    pipeline = ("v4l2src device=/dev/video0 ! "
                "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                "queue leaky=downstream max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! "
                "appsink drop=true max-buffers=1 sync=false")
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
else:
    cap = cv2.VideoCapture(0)

palm_frames = 0
prev_time = time.time()


# ===================== 메인 루프 =====================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 밝기 (02 HSV)
    brightness = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2].mean()

    # 2. 사람, 폰 탐지 (04 YOLO 클래스 지정 탐지)
    results = yolo.predict(frame, conf=0.4, classes=[PERSON, CELL_PHONE], verbose=False)
    classes = results[0].boxes.cls.tolist()
    person_found = PERSON in classes
    phone_box = None
    if CELL_PHONE in classes:
        x1, y1, x2, y2 = results[0].boxes.xyxy[classes.index(CELL_PHONE)].int().tolist()
        phone_box = frame[y1:y2, x1:x2].copy()   # 06: ROI

    # 3. 손바닥 (02 MediaPipe)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hands = hand_detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)).hand_landmarks
    fingers = count_fingers(hands[0]) if hands else 0
    palm_frames = palm_frames + 1 if fingers == 5 else 0

    # 4. 이벤트 판정
    if check_event("phone", phone_box is not None):
        save_event("phone", phone_box)
    if check_event("away", not person_found):
        save_event("away", frame)
    if check_event("dark", brightness < DARK_THRESHOLD):
        save_event("dark", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    if key == ord(" ") or palm_frames == PALM_FRAMES:
        save_event("question", frame)

    # 5. 화면 표시 (04 FPS)
    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    show = results[0].plot()   # YOLO 박스 그린 이미지
    cv2.putText(show, f"FPS {fps:.1f}  Bright {brightness:.0f}  Fingers {fingers}",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Study Buddy", show)

cap.release()
cv2.destroyAllWindows()