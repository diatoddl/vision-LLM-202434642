# Assignment02

import cv2
import numpy as np
import matplotlib.pyplot as plt
import skimage.data

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))



# LAB 색공간 임계값 범위 지정
lower_basketball_lab = np.array([70, 130, 70])
upper_basketball_lab = np.array([180, 230, 185])

# 이미지 불러오기 (BGR)
basketball = cv2.imread('src/images/basketball_crop.jpg')

# LAB 색공간으로 변환
basketball_lab = cv2.cvtColor(basketball, cv2.COLOR_BGR2LAB)

# 각 채널 분리
l, a, b = cv2.split(basketball_lab)

# CLAHE 생성 (clipLimit은 3.0, tileGridSize는 8x8로 설정)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

# L 채널에 CLAHE 적용 후 강화된 이미지 생성
l_clahe_enhanced = clahe.apply(l)
basketball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))

# 강화된 이미지로 binary mask 생성
mask_enhanced = cv2.inRange(basketball_enhanced_lab, lower_basketball_lab, upper_basketball_lab)



'''
# --- Opening / Closing 비교 ---
ball_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)
ball_mask_closed = cv2.morphologyEx(mask_enhanced, cv2.MORPH_CLOSE, kernel)

plt.figure(figsize=(16, 8))
plt.subplot(1, 3, 1), plt.imshow(mask_enhanced, cmap="gray"), plt.title("Binary Mask")
plt.subplot(1, 3, 2), plt.imshow(ball_mask_opened, cmap="gray"), plt.title("Mask after Opening")
plt.subplot(1, 3, 3), plt.imshow(ball_mask_closed, cmap="gray"), plt.title("Mask after Closing")
for ax in plt.gcf().axes:
    ax.axis("off")

# --- Opening Iteration 비교 ---
ball_mask_opened1 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)
ball_mask_opened2 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=2)
ball_mask_opened3 = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=3)

plt.figure(figsize=(16, 8))
plt.subplot(2, 3, 2), plt.imshow(cv2.cvtColor(basketball, cv2.COLOR_BGR2RGB)), plt.title("Original Image")
plt.subplot(2, 3, 4), plt.imshow(ball_mask_opened1, cmap="gray"), plt.title("iteration = 1")
plt.subplot(2, 3, 5), plt.imshow(ball_mask_opened2, cmap="gray"), plt.title("iteration = 2")
plt.subplot(2, 3, 6), plt.imshow(ball_mask_opened3, cmap="gray"), plt.title("iteration = 3")
for ax in plt.gcf().axes:
    ax.axis("off")

# --- Opening x2 + Closing Iteration 비교 ---
ball_mask_closed1 = cv2.morphologyEx(ball_mask_opened2, cv2.MORPH_CLOSE, kernel)
ball_mask_closed2 = cv2.morphologyEx(ball_mask_opened2, cv2.MORPH_CLOSE, kernel, iterations=2)
ball_mask_closed3 = cv2.morphologyEx(ball_mask_opened2, cv2.MORPH_CLOSE, kernel, iterations=3)

plt.figure(figsize=(16, 8))
plt.subplot(2, 3, 2), plt.imshow(ball_mask_opened2, cmap="gray"), plt.title("Opening x2")
plt.subplot(2, 3, 4), plt.imshow(ball_mask_closed1, cmap="gray"), plt.title("Opening x2 & Closing x1")
plt.subplot(2, 3, 5), plt.imshow(ball_mask_closed2, cmap="gray"), plt.title("Opening x2 & Closing x2")
plt.subplot(2, 3, 6), plt.imshow(ball_mask_closed3, cmap="gray"), plt.title("Opening x2 & Closing x3")
for ax in plt.gcf().axes:
    ax.axis("off")
'''


# 최종 Open x2 -> Close x2 선택
ball_mask_opened = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel, iterations=2)
basketball_mask_closed = cv2.morphologyEx(ball_mask_opened, cv2.MORPH_CLOSE, kernel, iterations=2)

# 최종 binary mask
final_ball_mask = basketball_mask_closed



# Matplotlib 시각화를 위해 BGR 이미지를 RGB로 변환하여 복사
final_img = cv2.cvtColor(basketball, cv2.COLOR_BGR2RGB)

# 컨투어 추출
contours, _ = cv2.findContours(
    final_ball_mask,        # binary mask
    cv2.RETR_EXTERNAL,      # 바깥쪽 테두리만 검출
    cv2.CHAIN_APPROX_SIMPLE  # 꼭짓점 좌표만 저장
)

if len(contours) > 0:
    # 가장 큰 contour 선택
    largest_contour = max(contours, key=cv2.contourArea)

    # 가장 큰 contour만 채운 binary mask 생성
    final_mask = np.zeros_like(final_ball_mask)
    cv2.drawContours(
        final_mask,
        [largest_contour],
        -1,
        255,
        thickness=cv2.FILLED
    )

    # RGB 이미지 기준 초록색 오버레이 생성 (RGB 중 G 채널 = index 1)
    mask_color = np.zeros_like(final_img)
    mask_color[:, :, 1] = final_mask  # Green 채널에 마스크 적용
    alpha = 0.35
    final_img = cv2.addWeighted(final_img, 1.0, mask_color, alpha, 0)

    # Contour 그리기 (RGB 기준 초록색 = (0, 255, 0))
    cv2.drawContours(final_img, [largest_contour], -1, (0, 255, 0), 2)

# 결과 출력
plt.figure(figsize=(8, 8))
plt.imshow(final_img)
plt.axis("off")
plt.show()
