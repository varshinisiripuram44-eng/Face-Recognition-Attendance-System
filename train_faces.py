import cv2
import os
import numpy as np

def train_faces(dataset_path="known_faces", model_path="trainer.yml"):
    """
    Reads images from the dataset, trains the LBPH recognizer,
    and saves the model.
    """
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = []
    labels = []
    label_map = {} # Dictionary to map numeric ID to string name
    current_id = 0

    print("[INFO] Starting training... This will take a moment.")
    
    for person_name in os.listdir(dataset_path):
        person_folder = os.path.join(dataset_path, person_name)
        
        # We only care about directories (the sub-folders for each person)
        if not os.path.isdir(person_folder):
            continue
            
        print(f"[INFO] Training on person: {person_name} (ID: {current_id})")
        label_map[current_id] = person_name
        
        # Loop through all images for this person
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                print(f"[WARN] Could not read image {img_name}, skipping.")
                continue
            
            # Detect face in the image
            faces_detected = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=4)
            
            for (x, y, w, h) in faces_detected:
                face = image[y:y+h, x:x+w] # The face region
                faces.append(face)
                labels.append(current_id)
                
        current_id += 1 # Next person gets the next ID

    if len(faces) == 0:
        print("[ERROR] No faces found for training. Did you follow the folder structure?")
        print("[ERROR] Expected: known_faces / person_name / image.jpg")
        return

    print(f"[INFO] {len(faces)} faces found. Training LBPH model...")
    
    # Train the recognizer and save the model
    recognizer.train(faces, np.array(labels))
    recognizer.save(model_path)

    # Save the ID-to-name mapping
    np.save("label_map.npy", label_map)
    
    print(f"[INFO] Training completed!")
    print(f"[INFO] Model saved to {model_path}")
    print(f"[INFO] Label map saved to label_map.npy")

if __name__ == "__main__":
    if not os.path.exists("known_faces"):
        os.makedirs("known_faces")
        print("[ERROR] 'known_faces' directory created.")
        print("[INFO] Please add sub-folders with images for each person inside 'known_faces' and run again.")
    else:
        train_faces()