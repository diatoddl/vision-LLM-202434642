"""
Color Hunt (색깔 물건 찾기 스피드 게임)

[규칙]
 - 화면에 [FIND: RED or GREEN or BLUE] 처럼 찾을 색이 나온다
   R, G, B로 만들 수 있는 7가지 조합(R, G, B, RG, RB, GB, RGB) 중 랜덤
 - 제한 시간 안에 그 색만 카메라에 보여주면 통과 (0.5초 유지)
   다른 색이 하나라도 같이 보이면 통과 X
 - 제한 시간: 1라운드는 색 1개 8초 / 2개 10초 / 3개 12초. 라운드마다 8%씩감소 
   최소 30%까지만 줄어듦
 - 시간 안에 못 찾으면 게임 종료 + 최종 점수 출력 (점수 = 색 개수 x 10 + 남은 초)

[설정]  SPACE 시작/재시작   D 마스크 보기   E CLAHE 켜기/끄기   Q 종료

[파이프라인]  프레임 -> 좌우반전 -> 가우시안 블러 -> (CLAHE) -> HSV -> inRange(색 분할)
             -> Opening/Closing -> findContours -> 큰 덩어리의 bounding box = 물건 위치
"""

import itertools # 색 조합 생성
import random
import sys
import time
import cv2
import numpy as np

# ===================== 설정 =====================
CAM_INDEX = 0
FRAME_W, FRAME_H = 640, 480
HOLD_TIME = 0.5            # 색상 노출이 이 시간 유지해야 통과
MIN_AREA = 1500            # 이 면적 이상인 것만 색 인정 (노이즈 X)
TIME_BY_COUNT = {1: 8, 2: 10, 3: 12}   # 1라운드 기준 {찾아야 할 색 개수: 제한 시간}
SHRINK_PER_ROUND = 0.08    # 라운드마다 제한 시간 감소할 비율
MIN_TIME_RATIO = 0.3       # 제한 시간 바닥

# 색별 HSV 범위 (H 0~179, S/V 0~255)
# 카메라/조명에 안 맞으면 D 키로 마스크를 보면서 조정
COLORS = {
    "red":   [((0, 180, 80), (6, 255, 255)), ((170, 180, 80), (179, 255, 255))],
    "green": [((40, 80, 60), (85, 255, 255))],
    "blue":  [((95, 120, 60), (130, 255, 255))],
}

# 화면에 그릴 색
BGR = {"red": (60, 60, 255), "green": (80, 220, 80), "blue": (255, 140, 40)}   

KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


# ===================== 색 물건 검출 =====================
def detect(frame, use_clahe):
    """프레임에서 각 색의 가장 큰 물건을 찾음.
    반환: found = {색: (x, y, w, h)}  (못 찾은 색은 없음),  masks = {색: 이진 마스크}"""
    img = cv2.GaussianBlur(frame, (7, 7), 0)                          # 필터링: 노이즈 제거
    if use_clahe:                                                      # Enhancement: 밝기(L)에만 CLAHE 적용
        l, a, b = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2LAB))
        img = cv2.cvtColor(cv2.merge((CLAHE.apply(l), a, b)), cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)                         # 색공간 변환

    found, masks = {}, {}
    for name, ranges in COLORS.items():
        mask = np.zeros(hsv.shape[:2], np.uint8)                       # 검정 이미지
        for lo, hi in ranges:                                          # Color Segmentation
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
            
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNEL, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, KERNEL, iterations=2)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)   # 바깥 컨투어만 꼭짓점으로 저장
        big = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]                      # 최소 윤곽 이상만
        if big:
            found[name] = cv2.boundingRect(max(big, key=cv2.contourArea))                  # 또 있으면 가장 큰 것만 found
        masks[name] = mask
    return found, masks


# ===================== 화면 그리기 =====================
def put_text(img, text, org, scale=1.0, color=(255, 255, 255), thick=2, center=False):
    """검은 외곽선 글자 + 색상 글자"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    if center: # 좌표 기준: 왼쪽 아래
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        org = (org[0] - tw // 2, org[1] + th // 2)
    cv2.putText(img, text, org, font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thick, cv2.LINE_AA)


def draw_boxes(out, found, need):
    """검출된 색에 박스 그림. 찾을 색이 아닌 물건은 회색 + X 표시"""
    for name, (x, y, w, h) in found.items():
        color = BGR[name] if name in need else (150, 150, 150)
        label = name.upper() if name in need else "X " + name.upper()
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 3)
        put_text(out, label, (x, max(20, y - 8)), 0.7, color, 2)


def draw_hud(out, need, found, remaining, limit, hold_ratio, score, round_no):
    """상단: 찾을 색 칩 / 시간 바 / 하단: 점수와 라운드."""
    w = out.shape[1]
    cv2.rectangle(out, (0, 0), (w, 70), (0, 0, 0), -1)
    put_text(out, "FIND:", (15, 45), 1.0)
    x = 130
    for name in need:
        done = name in found
        cv2.rectangle(out, (x, 18), (x + 120, 56), BGR[name], -1 if done else 3)   # 찾았으면 채움, 아니면 테두리
        put_text(out, name.upper(), (x + 60, 37), 0.7, (255, 255, 255) if done else BGR[name], 2, center=True)
        x += 135
    
    ratio = max(0.0, remaining / limit)
    bar = (80, 220, 80) if ratio > 0.5 else (0, 200, 255) if ratio > 0.25 else (60, 60, 255)
    cv2.rectangle(out, (0, 70), (int(w * ratio), 82), bar, -1)                     # 줄어드는 시간 바
    put_text(out, f"{max(0.0, remaining):.1f}s", (w // 2, 108), 0.9, center=True)
    if hold_ratio > 0:                                                              # 다 찾은 뒤 유지 진행도
        cv2.rectangle(out, (0, 84), (int(w * hold_ratio), 90), (0, 255, 255), -1)
    put_text(out, f"ROUND {round_no}   SCORE {score}", (15, out.shape[0] - 15), 0.8)


# ===================== 라운드 만들기 =====================
# 출제 가능한 7가지 조합: 색 1개짜리 3개 + 2개짜리 3개 + 3개짜리 1개
COMBOS = [list(c) for n in (1, 2, 3) for c in itertools.combinations(COLORS, n)]


def new_round(prev_need, round_no):
    """7가지 조합 중 하나 랜덤(직전 문제와 같은 조합은 제외)
    리턴: (찾을 색 목록, 제한 시간)"""
    need = random.choice([c for c in COMBOS if c != prev_need])
    ratio = max(MIN_TIME_RATIO, 1 - SHRINK_PER_ROUND * (round_no - 1))
    return need, round(TIME_BY_COUNT[len(need)] * ratio, 1)


# ===================== 카메라 =====================
def open_camera():
    """윈도우: DirectShow / 리눅스(Jetson USB캠): V4L2"""
    if sys.platform.startswith("win"):
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    return cap


# ===================== 메인 루프 =====================
def main():
    cap = open_camera()
    if not cap.isOpened():
        print("카메라를 열 수 없습니다")
        return

    state = "title"                 # title -> play -> over -> play ...
    use_clahe, show_mask = False, False
    score, round_no = 0, 0
    need, limit, t0, ok_since = [], 0, 0.0, None   # 이번 라운드 찾을 색, 제한시간, 시작시각, 다 찾기 시작한 시각

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                          # 거울 모드
        found, masks = detect(frame, use_clahe)
        out = frame.copy()                                  # 원본 X 복사본에 그림 그림
        now = time.time()

        if state == "title":
            out[:] = (out * 0.4).astype(np.uint8)
            put_text(out, "COLOR HUNT", (FRAME_W // 2, 150), 2.0, (0, 255, 255), 5, center=True)
            put_text(out, "Show objects of the called colors!", (FRAME_W // 2, 220), 0.8, center=True)
            put_text(out, "Press SPACE to start", (FRAME_W // 2, 300), 1.0, (80, 255, 80), 3, center=True)

        elif state == "play":
            remaining = limit - (now - t0)
            wrong = [c for c in found if c not in need]    # 문제가 아닌데 보이는 색
            exact = set(found) == set(need)                # 보이는 색 = 문제 색
            if exact:
                ok_since = ok_since or now                 # 처음 다 보인 시각
            else:
                ok_since = None                            # 하나라도 사라지면 유지 시간을 다시 카운트
            hold_ratio = (now - ok_since) / HOLD_TIME if ok_since else 0.0

            if hold_ratio >= 1.0:                          # 통과: 점수 = 색 개수 x 10 + 남은 초
                score += len(need) * 10 + int(remaining)
                round_no += 1
                need, limit = new_round(need, round_no)
                t0, ok_since = now, None
            elif remaining <= 0:                           # 시간 초과: 게임 종료
                state = "over"

            draw_boxes(out, found, need)
            draw_hud(out, need, found, remaining, limit, hold_ratio, score, round_no)
            if wrong:                                      # 통과 안 된 이유 경고
                put_text(out, "REMOVE: " + " ".join(c.upper() for c in wrong),
                         (FRAME_W // 2, 160), 1.2, (60, 60, 255), 3, center=True)

        else:  # over
            out[:] = (out * 0.3).astype(np.uint8)
            put_text(out, "GAME OVER", (FRAME_W // 2, 130), 2.0, (60, 60, 255), 5, center=True)
            put_text(out, f"FINAL SCORE  {score}", (FRAME_W // 2, 210), 1.5, (0, 255, 255), 4, center=True)
            put_text(out, f"Rounds cleared: {round_no - 1}", (FRAME_W // 2, 270), 0.9, center=True)
            put_text(out, "SPACE: retry   Q: quit", (FRAME_W // 2, 400), 0.9, (80, 255, 80), 2, center=True)

        if show_mask:                                      # D 키: 세 색상 마스크 (R | G | B)
            strip = np.hstack([cv2.resize(masks[c], (160, 120)) for c in COLORS])
            out[-128:-8, -488:-8] = cv2.cvtColor(strip, cv2.COLOR_GRAY2BGR)

        cv2.imshow("Color Hunt", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("e"):
            use_clahe = not use_clahe
        elif key == ord("d"):
            show_mask = not show_mask
        elif key == ord(" ") and state in ("title", "over"):
            state, score, round_no = "play", 0, 1
            need, limit = new_round(need, round_no)
            t0, ok_since = time.time(), None

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
