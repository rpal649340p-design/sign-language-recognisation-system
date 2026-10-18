
import cv2
import mediapipe as mp
import pickle
import pandas as pd
import subprocess
import threading
import queue
import os
import base64

# =============== SETTINGS ===============

MODEL_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "model.pkl"
)

CONFIDENCE_THRESHOLD = 0.45
STABLE_FRAMES = 8

# ========================================

# Load trained model
if not os.path.exists(MODEL_FILE):
    print("ERROR: model.pkl not found!")
    raise SystemExit

with open(MODEL_FILE, "rb") as file:
    model = pickle.load(file)

print("Model loaded successfully!")
print("Available signs:", list(model.classes_))

FEATURE_COLUMNS = [
    name
    for i in range(21)
    for name in (f"x{i}", f"y{i}")
]

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# Speech queue
speech_queue = queue.Queue()


def get_spoken_text(sign):
    words = {
        "YES": "Yes",
        "NO": "No",
        "STOP": "Stop",
        "OKAY": "Okay",
        "PEACE": "Peace",
        "THANKYOU": "Thank you",
        "QUESTION": "Question"
    }
    return words.get(str(sign).upper(), str(sign))


# =============== SPEECH WORKER ===============

def speak_worker():
    ps_script = r"""
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speaker.Rate = 0

while ($true) {
    $line = [Console]::In.ReadLine()

    if ($null -eq $line -or $line -eq "__EXIT__") {
        break
    }

    if (-not [string]::IsNullOrWhiteSpace($line)) {
        $speaker.Speak($line)
    }
}

$speaker.Dispose()
"""

    encoded_script = base64.b64encode(
        ps_script.encode("utf-16le")
    ).decode("ascii")

    process = None

    try:
        process = subprocess.Popen(
            [
                "powershell.exe",
                "-NoProfile",
                "-EncodedCommand",
                encoded_script
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1
        )

        while True:
            message = speech_queue.get()

            if message is None:
                break

            if process.poll() is not None:
                print("Speech process stopped.")
                break

            process.stdin.write(message + "\n")
            process.stdin.flush()

    except Exception as error:
        print("Speech error:", error)

    finally:
        if process is not None and process.poll() is None:
            try:
                process.stdin.write("__EXIT__\n")
                process.stdin.flush()
                process.stdin.close()
                process.wait(timeout=5)
            except Exception:
                process.terminate()


speech_thread = threading.Thread(
    target=speak_worker,
    daemon=True
)
speech_thread.start()


# =============== OPEN CAMERA ===============

camera = None

backends = [
    ("Default", cv2.CAP_ANY),
    ("MSMF", cv2.CAP_MSMF),
    ("DirectShow", cv2.CAP_DSHOW)
]

for backend_name, backend in backends:
    for camera_index in range(4):

        print(
            "Trying camera index:",
            camera_index,
            "| Backend:",
            backend_name
        )

        test_camera = cv2.VideoCapture(
            camera_index,
            backend
        )

        if test_camera.isOpened():
            success, test_frame = test_camera.read()

            if success and test_frame is not None:
                camera = test_camera
                print("Camera opened successfully!")
                print("Camera index:", camera_index)
                print("Backend:", backend_name)
                break

        test_camera.release()

    if camera is not None:
        break


if camera is None:
    print("ERROR: Could not open any camera.")
    print("Check Windows camera permissions.")
    print("Try opening the Windows Camera app.")

    speech_queue.put(None)
    speech_thread.join(timeout=5)
    hands.close()
    raise SystemExit


# =============== MAIN PROGRAM ===============

print("Sign Language Converter started!")
print("Show your hand and hold each sign steady.")
print("Press Q to quit.")

stable_count = 0
previous_sign = None
last_spoken_sign = None

try:
    while True:
        success, frame = camera.read()

        if not success or frame is None:
            print("Could not read camera frame.")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            features = []

            for landmark in hand_landmarks.landmark:
                features.extend([
                    landmark.x,
                    landmark.y
                ])

            input_data = pd.DataFrame(
                [features],
                columns=FEATURE_COLUMNS
            )

            prediction = str(model.predict(input_data)[0])

            confidence = 1.0

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_data)[0]
                confidence = float(max(probabilities))

            if confidence >= CONFIDENCE_THRESHOLD:
                if prediction == previous_sign:
                    stable_count += 1
                else:
                    previous_sign = prediction
                    stable_count = 1

                if (
                    stable_count >= STABLE_FRAMES
                    and prediction != last_spoken_sign
                ):
                    spoken_text = get_spoken_text(prediction)

                    print(
                        "Detected:",
                        prediction,
                        "| Confidence:",
                        round(confidence * 100, 1),
                        "%"
                    )

                    speech_queue.put(spoken_text)
                    last_spoken_sign = prediction

                display_text = (
                    f"{prediction} "
                    f"({confidence * 100:.1f}%)"
                )

            else:
                previous_sign = None
                stable_count = 0
                display_text = "Low confidence"

            cv2.putText(
                frame,
                display_text,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:
            previous_sign = None
            stable_count = 0
            last_spoken_sign = None

            cv2.putText(
                frame,
                "Show your hand",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow(
            "Sign Language to Text and Speech",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    hands.close()

    speech_queue.put(None)
    speech_thread.join(timeout=15)

    print("Program closed.")
