import cv2
import os
import time


name = input("Enter the name of the person to fix (e.g., surya): ").strip()
if not name:
    print("Error: Name cannot be empty.")
    exit()

folder_dir = r"C:\FaceSystems\known_faces"
image_path = os.path.join(folder_dir, f"{name}.jpg")


if not os.path.exists(image_path):
    print(f"Error: Could not find '{image_path}'.")
    print("Please make sure the file is in your 'known_faces' folder.")
    exit()

try:
    
    img = cv2.imread(image_path)
    
    if img is None:
        print("Error: OpenCV could not read the image. It might be corrupted.")
        exit()
        
    
    time.sleep(0.2)
    os.remove(image_path)
    
  
    cv2.imwrite(image_path, img)
    
    print(f"\n Success '{name}.jpg' has been saved.")

except Exception as e:
    print(f"\n An error occurred: {e}")