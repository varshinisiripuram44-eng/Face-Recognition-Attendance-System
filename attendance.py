
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os

def mark_attendance(name, csv_path="attendance.csv"):
    """
    Marks attendance in a CSV file, avoiding duplicates for the same day.
    """
    now = datetime.now()
    time_str = now.strftime("%I:%M:%S %p")
    date_str = now.strftime("%d-%m-%Y")

    if name == "Unknown":
        return

    columns = ["Name", "Time", "Date"]

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(columns=columns)

    exists = ((df["Name"] == name) & (df["Date"] == date_str)).any() if not df.empty else False

    if not exists:
        new_entry = pd.DataFrame([[name, time_str, date_str]], columns=columns)
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Attendance marked for {name} at {time_str}")

def run_attendance(model_path="trainer.yml", label_map_path="label_map.npy"):
    if not os.path.exists(model_path) or not os.path.exists(label_map_path):
        print("[ERROR] Model files not found!")
        print("[INFO] Please run 'train_faces.py' first to create the model.")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    label_map = np.load(label_map_path, allow_pickle=True).item()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print("[INFO] Starting webcam... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            id_, conf = recognizer.predict(roi_gray)

            confidence_threshold = 70

            if conf < confidence_threshold:
                name = label_map.get(id_, "Unknown")
            else:
                name = "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({conf:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            mark_attendance(name)

        cv2.imshow("LBPH Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam stopped.")

if __name__ == "__main__":
    run_attendance()
