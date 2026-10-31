# Assignment03

## 파일 구조
- **03_Afterimages.py**: 261025 3주차 CuPy를 이용한 배열 연산 GPU 가속 1-이미지 잔상 효과 구현 과제
- **03_Convolution2D.py**: 261025 3주차 CuPy를 이용한 배열 연산 GPU 가속 2-이미지 Convolution 연산 과제


## 실행 방식
- **실행 환경**: OpenCV, Matplotlib, CuPy, NumPy

- **실행**:
```bash
python 03_Convolution2D.py
python 03_Convolution2D_2.py
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
