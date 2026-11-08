"""
assistant.py - event.json 이 바뀌면 Gemma 지적 터미널 + 음성 출력
"""
import base64
import json
import os
import subprocess
import sys
import time

import cv2
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Gemma4ChatHandler


# ===================== 설정 =====================
JETSON = True  # 젯슨에서 실행할 때 True

EVENT_JSON_PATH = "src/output/event.json"
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"
WHISPER_MODEL = "whisper.cpp/models/ggml-base.bin"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
AUDIO_FILE = "src/audio/input.wav"
OUTPUT_FILE = "src/audio/response.wav"
RECORD_SECONDS = 5
MEMORY_SIZE = 6  # 05 Sliding Window: 최근 대화 6개만 기억

if JETSON:
    WHISPER_PATH = "whisper.cpp/build-cpu/bin/whisper-cli"
    PIPER_PYTHON = ".piper_venv/bin/python"
else:
    WHISPER_PATH = "whisper.cpp/whisper-cli.exe"
    PIPER_PYTHON = sys.executable

SYSTEM_PROMPT = """
Role: 너는 학생 옆에서 공부를 감시하고 도와주는 무서운 공부 코치 열품카야.
Constraint: 확인되는 내용만 말해. 이모지나 특수문자는 사용 금지.
Output Format: 한국어 한 두 문장 내외로.
"""


# ===================== Gemma 불러오기 (05, 06) =====================
llm = Llama(
    model_path=GEMMA_MODEL_PATH,
    chat_handler=Gemma4ChatHandler(clip_model_path=MMPROJ_PATH),
    n_gpu_layers=-1 if JETSON else 0,  # 젯슨은 GPU, 노트북은 CPU
    n_ctx=2048,
    n_batch=32,
    n_ubatch=32,
    verbose=False,
)


# ===================== STT / TTS (06) =====================
def speech_to_text():
    print("[듣는 중...] 5초 동안 말하세요")
    if JETSON:
        subprocess.run(["pasuspender", "--", "arecord", "-D", "plughw:2,0", "-f", "S16_LE",
                        "-r", "16000", "-c", "1", "-d", str(RECORD_SECONDS), AUDIO_FILE])
    else:
        import sounddevice as sd
        import wave
        audio = sd.rec(RECORD_SECONDS * 16000, samplerate=16000, channels=1, dtype="int16")
        sd.wait()
        with wave.open(AUDIO_FILE, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(16000)
            f.writeframes(audio.tobytes())

    result = subprocess.run([WHISPER_PATH, "-m", WHISPER_MODEL, "-f", AUDIO_FILE,
                             "-l", "ko", "--no-timestamps"], capture_output=True)
    return result.stdout.decode("utf-8", errors="ignore").strip()


def text_to_speech(text):
    print("[버디]", text)
    subprocess.run([PIPER_PYTHON, "-m", "piper", "-m", PIPER_MODEL, "-f", OUTPUT_FILE, "--", text])
    if JETSON:
        subprocess.run(["aplay", OUTPUT_FILE])
    else:
        import winsound
        winsound.PlaySound(OUTPUT_FILE, winsound.SND_FILENAME)


# ===================== JSON -> 자연어 (06) =====================
def event_to_text(event):
    if event["type"] == "phone":
        return f"학생이 휴대폰을 {event['seconds']}초째 보고 있다. 휴대폰 그만하고 공부하라고 따끔하게 말해라."
    if event["type"] == "away":
        return f"학생이 자리를 {event['seconds']}초째 비웠다. 어서 돌아와서 공부하라고 따끔하게 말해라."
    if event["type"] == "dark":
        return f"책상이 {event['seconds']}초째 어둡다. 불 키고 공부하자고 따끔하게 말해라."

    count = event["count"]
    return (f"[공부 기록] {event['study_minutes']}분 공부함. 휴대폰 경고 {count['phone']}회, "
            f"자리 비움 경고 {count['away']}회, 조명 경고 {count['dark']}회.")


# ===================== Gemma 호출 (06 Image 기반 Vision-LLM) =====================
memory = []


def ask_gemma(text, image_path):
    _, buffer = cv2.imencode(".jpg", cv2.imread(image_path))
    image_data = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + memory[-MEMORY_SIZE:]
    messages.append({"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": image_data}},
    ]})

    response = llm.create_chat_completion(messages=messages, max_tokens=80)
    return response["choices"][0]["message"]["content"].strip()


# ===================== 메인 루프 (06: 파일 수정 시간 감시) =====================
print("공부 감시 중.")
last_modified = os.path.getmtime(EVENT_JSON_PATH) if os.path.exists(EVENT_JSON_PATH) else 0

while True:
    time.sleep(0.1)
    if not os.path.exists(EVENT_JSON_PATH) or os.path.getmtime(EVENT_JSON_PATH) == last_modified:
        continue
    last_modified = os.path.getmtime(EVENT_JSON_PATH)

    with open(EVENT_JSON_PATH, "r", encoding="utf-8") as f:
        event = json.load(f)
    text = event_to_text(event)

    if event["type"] == "question":
        text_to_speech("네, 말하세요.")
        question = speech_to_text()
        print("[나]", question)
        answer = ask_gemma(text + "\n학생의 질문: " + question, event["roi_path"])
        memory.append({"role": "user", "content": question})
        memory.append({"role": "assistant", "content": answer})
    else:
        answer = ask_gemma(text, event["roi_path"])

    text_to_speech(answer)