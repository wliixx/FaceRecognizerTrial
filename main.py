import cv2
import numpy as np
from scipy.spatial import Delaunay
import urllib.request
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



MODEL_PATH = 'face_landmarker.task'
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Скачиваю модель face_landmarker.task...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
detector = vision.FaceLandmarker.create_from_options(options)

def draw_face_mesh(frame, face_landmarks):
    h, w, _ = frame.shape
    
    points = np.array(
        [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
    )
    
    tri = Delaunay(points)
    
    for triangle in tri.simplices:
        p1, p2, p3 = points[triangle[0]], points[triangle[1]], points[triangle[2]]
        cv2. line(frame, tuple(p1), tuple(p2), (0,255,0), cv2.LINE_AA)
        cv2. line(frame, tuple(p2), tuple(p3), (0,255,0), cv2.LINE_AA)
        cv2. line(frame, tuple(p3), tuple(p1), (0,255,0), cv2.LINE_AA)
        
    for (x,y) in points:
        cv2. line(frame, (x,y),1, (0,255,0), -1,cv2.LINE_AA)
        
    return frame


cap = cv2.VideoCapture(0)
frame_timestamps_ms = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame - cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    frame_timestamps_ms += 33
    detection_result = detection.detect_for_video(mp_image, frame_timestamps_ms)
    
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            frame = draw_face_mesh(frame, face_landmarks)
            
    cv2.imshow("Face Skeleton", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
detector.close()
