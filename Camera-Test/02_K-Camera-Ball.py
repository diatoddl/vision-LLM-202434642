import numpy as np
import cv2
import matplotlib.pyplot as plt


# TODO: 배운 기술(이미지 전처리, 특징 추출)을 활용하여 실시간으로 객체를 검출

# TODO: 각 프레임 이미지를 전처리
# TODO: Color Segmentation으로 Binary Mask 생성
# TODO: Morphological Operation을 적용하여 객체 영역 추출
# TODO: 객체의 중심과 반지름을 구해 프레임 위에 overlay

pipeline = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

while True:
    ret, frame = cap.read()

    if not ret:
        break
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


    green_lower = np.array([30, 60, 90], dtype=np.uint8)
    green_upper = np.array([230, 115, 180], dtype=np.uint8)


    ball_lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(ball_lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe_enhanced = clahe.apply(l)
    ball_enhanced_lab = cv2.merge((l_clahe_enhanced, a, b))
    
    mask_enhanced = cv2.inRange(ball_enhanced_lab, green_lower, green_upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    final_ball_mask = cv2.morphologyEx(mask_enhanced, cv2.MORPH_OPEN, kernel)

    final_img_enhanced = frame.copy()
    a = cv2.cvtColor(ball_enhanced_lab, cv2.COLOR_LAB2RGB)



    contours, _ = cv2.findContours(
        final_ball_mask,        # binary mask
        cv2.RETR_EXTERNAL,      # 바깥쪽 테두리만 검출
        cv2.CHAIN_APPROX_SIMPLE  # 꼭짓점 좌표만 저장
    )

    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)

        # 가장 큰 contour만 채운 binary mask 생성
        largest_mask = np.zeros_like(final_ball_mask)
        cv2.drawContours(
            largest_mask,
            [largest_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        # 가장 큰 contour만 채운 binary mask를 반투명 빨간색으로 overlay
        mask_color = np.zeros_like(final_img_enhanced)
        mask_color[:, :, 0] = largest_mask
        alpha = 0.15
        final_img_enhanced = cv2.addWeighted(final_img_enhanced, 1.0, mask_color, alpha, 0)

        # Contour 그리기 (빨간색 외곽선)
        cv2.drawContours(final_img_enhanced, [largest_contour], -1, (255,0,0), 2)

        # 최소 외접원 중심 및 반지름
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center_circle = (int(x), int(y))
        radius = int(radius)

        # 파란색 원
        cv2.circle(final_img_enhanced, center_circle, radius, (0,0,255), 3)
        cv2.circle(final_img_enhanced, center_circle, 5, (0,0,255), -1)


        # Moments를 통해 무게중심 (Centroid) 산출
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        centroid = (cX, cY)

        # 노란색 무게중심
        cv2.circle(final_img_enhanced, centroid, 5, (255,255,0), -1)
        
        cv2.imshow("VideoCapture with GStreamer", final_img_enhanced)


cap.release()
cv2.destroyAllWindows()


