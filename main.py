import cv2
import math

def main():
    face_cascade = cv2.CascadeClassifier('./opencv_things/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv_things/haarcascade_eye.xml')
    cap = cv2.VideoCapture(0)
    
    focal_length = 1000
    
    eye_dist = 2.5

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            (x, y, w, h) = faces[0]  

            face_roi_gray = gray[y:y+h, x:x+w]
            face_roi_color = frame[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(
                face_roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(15, 15)
            )
            
            if len(eyes) >= 2:
                (ex1, ey1, ew1, eh1) = eyes[0]
                (ex2, ey2, ew2, eh2) = eyes[1]

                eye1_center_x = x + ex1 + ew1 // 2
                eye1_center_y = y + ey1 + eh1 // 2
                eye2_center_x = x + ex2 + ew2 // 2
                eye2_center_y = y + ey2 + eh2 // 2
                
                between_eyes_pixels = math.dist(
                    (eye1_center_x, eye1_center_y), 
                    (eye2_center_x, eye2_center_y)
                )
                
                if between_eyes_pixels != 0:
                    distance_in_inches = (focal_length * eye_dist) / between_eyes_pixels
                    print(f"Estimated distance to camera: {distance_in_inches} inches")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
