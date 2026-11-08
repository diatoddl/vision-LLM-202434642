# 열품카 - 열정을 품은 타이

카메라로 공부하는 모습을 보다가 **딴짓을 하면 Gemma가 장면을 보고 음성으로 혼내는** Vision-LLM 멀티모달 시스템입니다.
손바닥을 펴면 음성으로 질문할 수 있고, Gemma가 공부 기록을 보고 음성으로 대답합니다.

<br>

---

## 파일 구조

* `vision.py`: 눈 역할. 카메라로 상황을 감지하고 이벤트를 저장
* `assistant.py`: 입과 귀 역할. 이벤트를 읽어 Gemma로 답을 만들고 음성으로 출력
* `README.md`: 이 문서

| 파일 | 주요 구성 | 역할 |
|---|---|---|
| `vision.py` | `count_fingers()` | 손가락 각도로 펴진 손가락 수 세기 |
| | `check_event()` | 같은 상황이 3초 이상 계속되면 이벤트 발생 |
| | `save_event()` | 이벤트 정보(JSON)와 사진(ROI) 저장 |
| `assistant.py` | `speech_to_text()` / `text_to_speech()` | Whisper로 듣기 / Piper로 말하기 |
| | `event_to_text()` | JSON을 Gemma가 읽을 문장으로 변환 |
| | `ask_gemma()` | 문장 + 사진 + 최근 대화를 Gemma에 전달 |

두 파일은 `src/output/event.json`과 `roi.jpg`를 통해 연결됩니다.

<br>

---

## 실행 방식

### 준비
```bash
pip install numpy opencv-python ultralytics mediapipe sounddevice llama-cpp-python piper-tts
```
- 모델: `yolo11n.pt`, `hand_landmarker.task`, Gemma 4 GGUF + mmproj, Whisper `ggml-base.bin`, Piper `ko_KR-kss-medium`
- 젯슨에서는 두 파일 맨 위의 `JETSON = True`로 변경 (TensorRT INT8 엔진, GStreamer 카메라, GPU 사용)

### 실행 (저장소 최상위 폴더에서 터미널 2개 사용)
```bash
python Final-Project/vision.py
python Final-Project/assistant.py
```

### 조작 키
| 키 | 동작 |
|---|---|
| 손바닥 1초 / `SPACE` | 음성 질문 |
| `Q` | 종료 (카메라 창) |

### 감지하는 상황
| 상황 | 조건 | 반응 |
|---|---|---|
| Phone | YOLO가 폰을 3초 이상 탐지 | 폰 사진을 보고 혼냄 |
| Away | 사람이 3초 이상 없음 | 돌아오라고 말함 |
| Dark | HSV 밝기 평균 70 미만 3초 | 불 켜라고 말함 |
| Question | 손가락 5개 펴고 15프레임 유지 | "네, 말하세요" → 듣고 대답 |

<br>

---

## 창의성 및 확장성

### 아이디어 설명
공부하다 무의식적으로 폰을 보는 습관을 고치기 위해, 옆에서 지켜보고 바로 혼내주는 AI 코치를 만들었습니다.
단순히 정해진 문구가 아닌, Gemma가 실제 장면 사진을 보고 말하도록 했습니다.

### 구현한 내용
- **파이프라인 분리**: Gemma가 답을 만드는 동안에도 카메라가 멈추지 않도록 두 프로그램으로 나누고, JSON 파일의 수정 시간으로 연결
- **지속 시간 판정**: 상황이 처음 시작된 시각을 기록하고, 3초 이상 계속될 때만 이벤트 발생. 끊기면 초기화해서 순간적인 오검출 방지
- **ROI 전달**: 폰 이벤트는 폰 부분만 잘라서 Gemma에 전달
- **상황별 프롬프트**: 이벤트일 때는 혼내라는 지시만, 질문일 때는 공부 기록을 함께 전달
- **대화 메모리**: 최근 대화 6개만 Gemma에 다시 넣어 이어서 대화
- **환경 전환**: JETSON = True로 모델, 카메라, 녹음·재생 방식, GPU 사용 전환

### 사용한 수업 기술
| 실습 | 사용한 곳 |
|---|---|
| 01 Python | Dictionary로 상태·횟수 관리, JSON으로 이벤트 저장 |
| 02 Computer Vision | HSV 밝기, MediaPipe 손가락 각도, GStreamer 카메라 |
| 03 DL & GPU | 젯슨 GPU로 YOLO·Gemma 가속 |
| 04 Object Detection | YOLO 클래스 지정 탐지(사람·폰), TensorRT INT8, FPS |
| 05 LLM & Gemma | 시스템 프롬프트(Role/Constraint/Output Format), Sliding Window 메모리 |
| 06 Vision-LLM | JSON → 자연어, ROI → Gemma, 파이프라인 분리, Whisper STT, Piper TTS |

### 확장성
- **공부 리포트**: 하루 기록을 저장해서 Gemma가 하루 요약을 말해주기
- **졸음 감지**: MediaPipe 얼굴 인식으로 눈 감김 감지

<br>

---

## 기타 참고 사항

### 한계
- CPU에서는 Gemma 답변에 30초 가량 걸려 실용이 떨어집니다.
- Gemma가 답하는 중에 생긴 이벤트는 덮어써져서 하나만 처리됩니다.
- 휴대폰을 들고 있지 않아도 감지가 되면 이벤트가 될 수 있습니다.
