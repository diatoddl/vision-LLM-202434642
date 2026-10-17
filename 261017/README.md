# Assignment02

## 파일 구조
**02_Basketball-Contour-Assignment.py**: 261017 2주차 농구공 외곽선 검출 TODO 과제

## 실행 방식
python 파일 다운로드 후 실행 (이미지 다운 필수)

## 코드 설명
CLAHE 전처리와 LAB 색공간을 활용해 농구공 영역을 추출하고, \
모포놀로지 연산(Opening/Closing)의 Iteration 수에 따른 노이즈 제거 및 팽창 효과를 비교 실험하여 \
최종 이진 마스크와 초록색 오버레이 Contour를 검출하는 파이프라인
