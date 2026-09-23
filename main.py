import cv2
import numpy as np
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SILLY_CAT_NAME = 'Silly car.jpy'
SILLY_CAT_PATH = os.path.join(SCRIPT_DIR, SILLY_CAT_NAME)

sillycat_image = cv2.imread(SILLY_CAT_PATH)
if sillycat_image is None:
    print(f'не удалось загрузить картинку по пути: {SILLY_CAT_PATH}')


WINK_THRESHOLD = 0.5
WINK_OPEN_THRESHOLD = 0.3
JAW_OPEN_THRESHOLD = 0.35

def get_blendshapes_dict(detection_result, face_index=0):
    if not detection_result.face_blendshapes:
        return {}
    categories = detection_result.face_blendshapes[face_index]
    return {c.category_name: c.score for c in categories}

def is_winking(blendshapes):
    left = blendshapes.get('eyeBlinkLeft', 0.0)
    right = blendshapes.get('eyeBlinkRight', 0.0)
    left_wink = left > WINK_THRESHOLD and right < WINK_OPEN_THRESHOLD
    right_wink =  right > WINK_THRESHOLD and left < WINK_OPEN_THRESHOLD
    return left_wink or right_wink

def is_tongue_out(frame, face_landmarks, blendshapes):
    jaw_open = blendshapes.get("jawOpen", 0.0)
    if jaw_open < JAW_OPEN_THRESHOLD:
        return False
    
FACE_OVAL = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162),
    (162, 21), (21, 54), (54, 103), (103, 67), (67, 109), (109, 10),
]

LEFT_EYE = [
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381),
    (381, 382), (382, 362), (263, 466), (466, 388), (388, 387), (387, 386),
    (386, 385), (385, 384), (384, 398), (398, 362),
]
 
RIGHT_EYE = [
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154),
    (154, 155), (155, 133), (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133),
]
 
LEFT_EYEBROW = [
    (276, 283), (283, 282), (282, 295), (295, 285),
    (300, 293), (293, 334), (334, 296), (296, 336),
]
 
RIGHT_EYEBROW = [
    (46, 53), (53, 52), (52, 65), (65, 55),
    (70, 63), (63, 105), (105, 66), (66, 107),
]
 
LIPS = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314),
    (314, 405), (405, 321), (321, 375), (375, 291), (61, 185), (185, 40),
    (40, 39), (39, 37), (37, 0), (0, 267), (267, 269), (269, 270),
    (270, 409), (409, 291), (78, 95), (95, 88), (88, 178), (178, 87),
    (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308),
    (78, 191), (191, 80), (80, 81), (81, 82), (82, 13), (13, 312),
    (312, 311), (311, 310), (310, 415), (415, 308),
]

ALL_CONNECTIONS = FACE_OVAL + LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW + LIPS

def draw_face_mesh(frame, face_landmarks):
    h, w, _ = frame.shape
    
    points = np.array(
        [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks]
    )
    for start_idx, end_idx in ALL_CONNECTIONS:
        cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 255), 1, cv2.LINE_AA)

    for (x,y) in points:
         cv2.circle(frame, (x, y), 1, (0, 0, 255), -1, cv2.LINE_AA)
        
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
    detection_result = detector.detect_for_video(mp_image, frame_timestamps_ms)
    
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            frame = draw_face_mesh(frame, face_landmarks)
            
    cv2.imshow("Face Skeleton", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
detector.close()

