'''required steps and libraries

py -3.11 -m venv face_env
.\ face_env\Scripts\Activate.ps1

pip install dlib-19.24.1-cp311-cp311-win_amd64.whl


pip uninstall numpy -y

pip install opencv-python face_recognition numpy==1.26.4

'''
import cv2
import face_recognition
import numpy as np
import os


known_face_encodings = []
known_face_names = []
known_faces_dir = r"C:\FaceSystems\known_faces"


if not os.path.exists(known_faces_dir):
    print(f"Error: The folder '{known_faces_dir}' does not exist.")
    print("Please create a folder named 'known_faces' next to this script.")
    exit()

print("Loading face database, please wait...")


for filename in os.listdir(known_faces_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(known_faces_dir, filename)
        
        try:
            
            image = face_recognition.load_image_file(image_path)
         
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                
                name = os.path.splitext(filename)[0].capitalize()
                known_face_names.append(name)
                print(f" Loaded face data for: {name}")
            else:
                print(f" Warning: No face found in {filename}. Skipping.")
        except Exception as e:
            print(f" Could not process file {filename}: {e}")

if not known_face_encodings:
    print("\n Database is empty! Put some face images into the 'known_faces' folder first.")
    exit()

print(f"\n Successfully loaded {len(known_face_names)} person(s). Starting webcam...")


video_capture = cv2.VideoCapture(0)


video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
   
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Failed to grab frame from your laptop's webcam.")
        break

   
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "Unknown person"

        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

        face_names.append(name)

    
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale coordinates back up by 4x to match original webcam video scale
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        
        box_color = (0, 255, 0) if name != "Unknown person" else (0, 0, 255)
        
        
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)

        
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), box_color, cv2.FILLED)
        
        
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 8, bottom - 8), font, 0.7, (255, 255, 255), 1)

    
    cv2.imshow('Face Recognition AI (Press Q to Close)', frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nShutting down system...")
        break


video_capture.release()
cv2.destroyAllWindows()