import time

import cv2
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt

IMAGE_PATH = "src/images/seagull.jpg"

TARGET_WIDTH = 1280
TARGET_HEIGHT = 720

TRAIL_LENGTH = 32

img_gray = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

if img_gray is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {IMAGE_PATH}")

img_gray = cv2.resize(
    img_gray,
    (TARGET_WIDTH, TARGET_HEIGHT),
    interpolation=cv2.INTER_AREA
)

input_np = img_gray.astype(np.float32) / 255.0


def relu(x):
    return np.maximum(0, x)


def convolution2d_np(image, kernel, bias=0.0, stride=1, padding=0):
    kernel_height, kernel_width = kernel.shape

    # 이미지 가장자리에 zero padding 적용
    padded_image = np.pad(
        image,
        ((padding, padding), (padding, padding)),
        mode="constant",
        constant_values=0
    )

    # (H, W: 입력 이미지 높이 및 너비, P: Padding 픽셀 값, S: Stride 픽셀 값)
    # Convolution 출력 이미지 높이 = floor((H + 2P - K) / S) + 1
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1

    # Convolution 출력 이미지 너비 = floor((W + 2P - K) / S) + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1

    # 계산한 크기로 출력 이미지 준비
    output = np.zeros((output_height, output_width), dtype=np.float32)

    # 커널을 움직이며 필터 적용
    for output_y in range(output_height):
        for output_x in range(output_width):
            # 커널을 적용할 이미지 영역의 좌측상단
            start_y = output_y * stride
            start_x = output_x * stride

            # 커널을 적용할 이미지 영역을 crop
            image_region = padded_image[
                start_y:start_y + kernel_height,
                start_x:start_x + kernel_width
            ]

            # 필터를 적용하여 가중합 계산
            weighted_sum = np.sum(image_region * kernel)

            # 편향 적용
            output[output_y, output_x] = weighted_sum + bias

    return output


dx_edge_filter = np.array([[-1, 0, 1],
                           [-1, 0, 1],
                           [-1, 0, 1]], dtype=np.float32)
 
dy_edge_filter = np.array([[-1, -1, -1],
                           [ 0,  0,  0],
                           [ 1,  1,  1]], dtype=np.float32)
 
dx_bias = -0.2
dy_bias = -0.2


# TODO: CuPy를 사용하여 2차원 convolution 함수 정의

def convolution2d_cp(image, kernel, bias=0.0, stride=1, padding=0):
    """
    CuPy를 이용한 2차원 convolution.

    커널의 위치는 Python 반복문으로 순차 처리하고,
    각 위치에 해당하는 출력 배열 전체 연산은 GPU에서 처리한다.
    """

    kernel_height, kernel_width = kernel.shape
 
    padded_image = cp.pad(image, ((padding, padding), (padding, padding)))
 
    output_height = (padded_image.shape[0] - kernel_height) // stride + 1
    output_width = (padded_image.shape[1] - kernel_width) // stride + 1
 
    output = cp.zeros((output_height, output_width), dtype=cp.float32)
 
    for kernel_y in range(kernel_height):
        for kernel_x in range(kernel_width):
            end_y = kernel_y + output_height * stride
            end_x = kernel_x + output_width * stride
 
            image_region = padded_image[kernel_y:end_y:stride, kernel_x:end_x:stride]
 
            output += kernel[kernel_y, kernel_x] * image_region
 
    return output + bias


# TODO: 기존에 정의한 convolution2d 함수를 사용하여 convolution을 CPU로 처리
# TODO: 변수 재사용 가능

start = time.perf_counter()

output_dx_np = convolution2d_np(input_np, dx_edge_filter, dx_bias, 1, 1)
output_dy_np = convolution2d_np(input_np, dy_edge_filter, dy_bias, 1, 1)

cpu_time = time.perf_counter() - start

relu_dx_np = relu(output_dx_np)
relu_dy_np = relu(output_dy_np)


# TODO: convolution을 GPU로 처리 (변환-워밍업-처리 파이프라인 구현)
# TODO: 변수 재사용 가능

# 1. 변환 (CPU -> GPU)
input_cp = cp.asarray(input_np)
dx_edge_filter_cp = cp.asarray(dx_edge_filter)
dy_edge_filter_cp = cp.asarray(dy_edge_filter)


# 2. 워밍업
_ = convolution2d_cp(input_cp, dx_edge_filter_cp, dx_bias, 1, 1)
cp.cuda.Stream.null.synchronize()


# 3. 처리
start = time.perf_counter()

output_dx_cp = convolution2d_cp(input_cp, dx_edge_filter_cp, dx_bias, 1, 1)
output_dy_cp = convolution2d_cp(input_cp, dy_edge_filter_cp, dy_bias, 1, 1)

cp.cuda.Stream.null.synchronize()
gpu_time = time.perf_counter() - start

relu_dx_cp_np = relu(cp.asnumpy(output_dx_cp))
relu_dy_cp_np = relu(cp.asnumpy(output_dy_cp))


# TODO: CPU/GPU 결과 비교 및 출력

print(f"CPU time: {cpu_time:.4f} seconds")
print(f"GPU time: {gpu_time:.4f} seconds")
print(f"Speedup: {cpu_time / gpu_time:.2f}x")



plt.figure(figsize=(18,8))

plt.subplot(2,3,1), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,2), plt.imshow(relu_dx_np, cmap="gray"), plt.title("Vertical Convolution (CPU)")
plt.subplot(2,3,3), plt.imshow(relu_dx_cp_np, cmap="gray"), plt.title("Vertical Convolution (GPU)")
plt.subplot(2,3,4), plt.imshow(input_np, cmap="gray"), plt.title("Input Image")
plt.subplot(2,3,5), plt.imshow(relu_dy_np, cmap="gray"), plt.title("Horizontal Convolution (CPU)")
plt.subplot(2,3,6), plt.imshow(relu_dy_cp_np, cmap="gray"), plt.title("Horizontal Convolution (GPU)")

for ax in plt.gcf().axes:
    ax.axis("off")

plt.show()
