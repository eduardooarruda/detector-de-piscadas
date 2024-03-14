import cv2
import mediapipe as mp
import time
import numpy as np
import pyautogui

pyautogui.FAILSAFE = False

cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

closed_eye_time =  time.time()
open_eye_time = time.time()

flag = True 

def eye_aspect_ratio(p1, p2, p3, p4, p5, p6):

    A = np.linalg.norm(p2.y - p6.y)
    B = np.linalg.norm(p3.y - p5.y)

    C = np.linalg.norm(p1.x - p4.x)
    
    ear = (A + B) / (2.0 * C)
    return ear

while True:
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape
    
    if landmark_points:
        landmarks = landmark_points[0].landmark
        
        right_p1 = landmarks[33]
        right_p2 = landmarks[160]
        right_p3 = landmarks[158]
        right_p4 = landmarks[133]
        right_p5 = landmarks[153]
        right_p6 = landmarks[144]

        left_p1 = landmarks[362]
        left_p2 = landmarks[385]
        left_p3 = landmarks[387]
        left_p4 = landmarks[263]
        left_p5 = landmarks[373]
        left_p6 = landmarks[380]

        right_eye_landmarks = [right_p1, right_p2,right_p3, right_p4,  right_p5, right_p6]

        left_eye_landmarks = [left_p1, left_p2, left_p3, left_p4, left_p5, left_p6]

        left_right_eye_landmarks = [left_p1, left_p2, left_p3, left_p4, left_p5, left_p6, right_p1, right_p2,right_p3, right_p4,  right_p5, right_p6]

        for landmark in left_right_eye_landmarks:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 2, (0, 255, 255))

        right_ear = eye_aspect_ratio( right_p1, right_p2, right_p3, right_p4,  right_p5, right_p6)
            
        left_ear = eye_aspect_ratio(left_p1, left_p2, left_p3, left_p4,  left_p5, left_p6)

        # print(ear)

        if left_ear < 0.25 or right_ear < 0.25:
            if flag:
                closed_eye_time = time.time()
                flag = False
           
       
        else:
            open_eye_time = time.time()
            if 0.5 <= open_eye_time - closed_eye_time <= 1.5:  
                pyautogui.click()
                print("Click")
            closed_eye_time = time.time()
            flag = True
             
    cv2.imshow('blink detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
