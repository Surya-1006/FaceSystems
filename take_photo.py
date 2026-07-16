import cv2
import os

name = input("Enter name for the database: ").strip()
if not name:
    print("Error: Name cannot be empty.")
    exit()

output_dir = r"C:\FaceSystems\known_faces"
filename = f"{name}.jpg"
output_path = os.path.join(output_dir, filename)

os.makedirs(output_dir, exist_ok=True)

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break
        
    cv2.imshow("Capture Window", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 32 or key == 13:  
        try:
            cv2.imwrite(output_path, frame)
            test_img = cv2.imread(output_path)
            if test_img is not None:
                cv2.imwrite(output_path, test_img)
        except Exception:
            pass
        break
        
    elif key == ord('q') or key == ord('Q'):  
        break

cam.release()
cv2.destroyAllWindows()