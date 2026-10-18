
import cv2
import mediapipe as mp
import csv
import os
import pickle
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ================ SETTINGS ================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SIGNS = [
    "YES",
    "NO",
    "STOP",
    "OKAY",
    "PEACE",
    "QUESTION",
    "THANKYOU"
]

DATA_FILE = os.path.join(BASE_DIR, "dataset.csv")
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")

# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

# Create dataset if it does not exist
if not os.path.exists(DATA_FILE):
    header = ["label"]

    for i in range(21):
        header.extend([f"x{i}", f"y{i}"])

    with open(DATA_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)


# ================ SELECT SIGN ================

print("\nSIGN LANGUAGE DATA COLLECTION")
print("--------------------------------")

for i, sign in enumerate(SIGNS, start=1):
    print(i, "-", sign)

try:
    choice = int(input("\nEnter sign number (1-7): "))

    if choice < 1 or choice > len(SIGNS):
        raise ValueError

except ValueError:
    print("Invalid choice. Run the program again.")
    hands.close()
    raise SystemExit

label = SIGNS[choice - 1]

print("\nSelected sign:", label)
print("Show this sign clearly to the camera.")
print("Vary your hand position and angle slightly.")
print("Press SPACE to save a sample.")
print("Press Q when finished.")


# ================ OPEN CAMERA ================

camera = None

for backend in [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]:
    for index in range(4):
        test_camera = cv2.VideoCapture(index, backend)

        if test_camera.isOpened():
            success, frame = test_camera.read()

            if success and frame is not None:
                camera = test_camera
                print("Camera opened. Index:", index)
                break

        test_camera.release()

    if camera is not None:
        break

if camera is None:
    print("ERROR: Could not open camera.")
    hands.close()
    raise SystemExit


# ================ COLLECT SAMPLES ================

saved_count = 0

try:
    while True:
        success, frame = camera.read()

        if not success or frame is None:
            print("Could not read camera frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

        cv2.putText(
            frame,
            "SIGN: " + label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Saved: " + str(saved_count),
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "SPACE = Save | Q = Finish",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow("Sign Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 32:
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                row = [label]

                for point in hand.landmark:
                    row.extend([point.x, point.y])

                with open(DATA_FILE, "a", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(row)

                saved_count += 1
                print(label, "- Sample saved:", saved_count)

            else:
                print("No hand detected. Try again.")

        elif key == ord("q"):
            break

finally:
    camera.release()
    cv2.destroyAllWindows()
    hands.close()

print("\nSamples added for", label, ":", saved_count)


# ================ TRAIN MODEL ================

print("\nChecking dataset and training model...")

data = pd.read_csv(DATA_FILE)
data = data.dropna()

if "label" not in data.columns:
    print("ERROR: Dataset must contain a label column.")
    raise SystemExit

feature_columns = [
    column for column in data.columns if column != "label"
]

if len(feature_columns) != 42:
    print("ERROR: Expected 42 landmark features.")
    print("Found:", len(feature_columns))
    print("Check your dataset.csv file.")
    raise SystemExit

data["label"] = data["label"].astype(str).str.strip().str.upper()

# Show sample counts
print("\nSamples for each sign:")
counts = data["label"].value_counts()

for sign in SIGNS:
    print(f"{sign}: {counts.get(sign, 0)} samples")

missing = [sign for sign in SIGNS if counts.get(sign, 0) == 0]

if missing:
    print("\nERROR: Missing samples for:", missing)
    print("Collect samples for these signs first.")
    raise SystemExit

X = data[feature_columns]
y = data["label"]

# Check that each sign has enough samples
if counts.min() < 2:
    print("ERROR: Each sign needs at least 2 samples.")
    raise SystemExit

# Diagnostic evaluation
if counts.min() >= 5:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    test_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    )

    test_model.fit(X_train, y_train)
    predictions = test_model.predict(X_test)

    print("\nTest accuracy:",
          round(accuracy_score(y_test, predictions) * 100, 2), "%")

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

else:
    print("\nNot enough samples per sign for a reliable test report.")

# Train final model using all available data
model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

model.fit(X, y)

with open(MODEL_FILE, "wb") as file:
    pickle.dump(model, file)

print("\nMODEL TRAINED SUCCESSFULLY!")
print("Model saved to:", MODEL_FILE)
print("Now run main.py to test your signs.")
