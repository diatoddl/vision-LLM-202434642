# Assignment03

## 파일 구조
- **03_Afterimages.py**: 261025 3주차 CuPy를 이용한 배열 연산 GPU 가속 1-이미지 잔상 효과 구현 과제
- **03_Convolution2D.py**: 261025 3주차 CuPy를 이용한 배열 연산 GPU 가속 2-이미지 Convolution 연산 과제
- **03_MNIST-Training.py**: 261025 3주차 PyTorch를 이용한 GPU 기반 MNIST 데이터셋 학습 과제



## 실행 방식
- **실행 환경**: OpenCV, Matplotlib, CuPy, NumPy, PyTorch, torchvision

- **실행**:
```bash
python 03_Afterimages.py
python 03_Convolution2D.py
python 03_MNIST-Training.py
```

## 실행 결과

- **AfterImages**
```text
========== NumPy CPU ==========
Processing time: 16.7676 seconds

========== CuPy GPU ==========
CPU to GPU: 0.0022 seconds
GPU processing: 0.0047 seconds
GPU to CPU: 0.0017 seconds
Total GPU time: 0.0086 seconds

========== Comparison ==========
Compute-only speedup: 3556.12x
Including transfers: 1947.63x
```

- **Convolution2D**
```text
CPU time: 11.8904 seconds
GPU time: 0.0031 seconds
Speedup: 3883.94x
```

- **MNIST-training**
```text
=== CPU Training ===
Epoch 1/3 | Train loss: 0.3133 | Train accuracy: 90.52% | Test loss: 0.0953 | Test accuracy: 96.99% | Time: 20.41s
Epoch 2/3 | Train loss: 0.0925 | Train accuracy: 97.15% | Test loss: 0.0693 | Test accuracy: 97.73% | Time: 21.36s
Epoch 3/3 | Train loss: 0.0670 | Train accuracy: 97.89% | Test loss: 0.0508 | Test accuracy: 98.20% | Time: 20.26s
Total CPU training time: 62.04s

=== GPU Training ===
Epoch 1/3 | Train loss: 0.3133 | Train accuracy: 90.54% | Test loss: 0.0966 | Test accuracy: 96.92% | Time: 10.11s
Epoch 2/3 | Train loss: 0.0918 | Train accuracy: 97.15% | Test loss: 0.0682 | Test accuracy: 97.84% | Time: 10.44s
Epoch 3/3 | Train loss: 0.0662 | Train accuracy: 97.94% | Test loss: 0.0489 | Test accuracy: 98.25% | Time: 10.04s
Total GPU training time: 30.59s

=== CPU vs. GPU Comparison ===
CPU training time: 62.04s
GPU training time: 30.59s
Time difference: 31.44s
GPU speedup: 2.03x
```
