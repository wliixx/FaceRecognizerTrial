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
    num_faces=10,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=False,
)
detector = vision.FaceLandmarker.create_from_options(options)
 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REACTION_IMAGE_BASENAME = "Silly cat"
 
 
def find_reaction_image(folder, basename):
    """Ищет в папке файл с именем basename и любым расширением-картинкой."""
    if not os.path.isdir(folder):
        return None
    for filename in os.listdir(folder):
        name_without_ext, ext = os.path.splitext(filename)
        if name_without_ext.lower() == basename.lower() and ext.lower() in (
            ".jpg", ".jpeg", ".jpe", ".png", ".bmp", ".webp",
        ):
            return os.path.join(folder, filename)
    return None
 
 
REACTION_IMAGE_PATH = find_reaction_image(SCRIPT_DIR, REACTION_IMAGE_BASENAME)
 
if REACTION_IMAGE_PATH is None:
    reaction_image = None
    print(f"⚠ Не найден файл \"{REACTION_IMAGE_BASENAME}.*\" в папке {SCRIPT_DIR}")
    print("Проверь, что картинка лежит рядом со скриптом и имя начинается так же.")
else:
    reaction_image = cv2.imread(REACTION_IMAGE_PATH)
    if reaction_image is None:
        print(f"⚠ Файл найден ({REACTION_IMAGE_PATH}), но не читается как изображение.")
 
# ПОРОГИ — подбери их под себя, глядя на реальные цифры в углу окна (DEBUG-строки)
WINK_THRESHOLD = 0.3
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
    right_wink = right > WINK_THRESHOLD and left < WINK_OPEN_THRESHOLD
    return left_wink or right_wink
 
 
def get_mouth_color_stats(frame, face_landmarks):
    """Возвращает (avg_hue, avg_sat) внутренней области рта, либо (None, None)."""
    h, w, _ = frame.shape
    inner_lip_idx = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                      95, 88, 178, 87, 14, 317, 402, 318, 324]
    xs = [int(face_landmarks[i].x * w) for i in inner_lip_idx]
    ys = [int(face_landmarks[i].y * h) for i in inner_lip_idx]
    x1, x2 = max(min(xs), 0), min(max(xs), w - 1)
    y1, y2 = max(min(ys), 0), min(max(ys), h - 1)
 
    if x2 <= x1 or y2 <= y1:
        return None, None
 
    mouth_roi = frame[y1:y2, x1:x2]
    if mouth_roi.size == 0:
        return None, None
 
    hsv_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2HSV)
    avg_hue = float(np.mean(hsv_roi[:, :, 0]))
    avg_sat = float(np.mean(hsv_roi[:, :, 1]))
    return avg_hue, avg_sat
 
 
def is_tongue_out(jaw_open, avg_hue, avg_sat):
    if jaw_open < JAW_OPEN_THRESHOLD:
        return False
    if avg_hue is None:
        return False
    is_pinkish = (avg_hue < 15 or avg_hue > 135) and avg_sat > 60
    return is_pinkish
 
 
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
 
    for (x, y) in points:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1, cv2.LINE_AA)
 
    return frame
 
 
cap = cv2.VideoCapture(0)
frame_timestamps_ms = 0
reaction_window_open = False
 
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
 
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
 
    frame_timestamps_ms += 33
    detection_result = detector.detect_for_video(mp_image, frame_timestamps_ms)
 
    expression_triggered = False
 
    if detection_result.face_landmarks:
        for face_landmarks in detection_result.face_landmarks:
            frame = draw_face_mesh(frame, face_landmarks)
 
        blendshapes = get_blendshapes_dict(detection_result)
        face_landmarks = detection_result.face_landmarks[0]
 
        eye_blink_left = blendshapes.get('eyeBlinkLeft', 0.0)
        eye_blink_right = blendshapes.get('eyeBlinkRight', 0.0)
        jaw_open = blendshapes.get('jawOpen', 0.0)
        avg_hue, avg_sat = get_mouth_color_stats(frame, face_landmarks)
 
        winking = is_winking(blendshapes)
        tongue = is_tongue_out(jaw_open, avg_hue, avg_sat)
        expression_triggered = winking and tongue
 
        # Итоговый статус
        cv2.putText(
            frame,
            f'wink={winking} tongue={tongue}',
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )
 
        # DEBUG: реальные цифры, по которым принимается решение.
        # Смотри на них, пока моргаешь/открываешь рот, и подбирай пороги выше.
        debug_lines = [
            f"eyeBlinkLeft={eye_blink_left:.2f} eyeBlinkRight={eye_blink_right:.2f}",
            f"jawOpen={jaw_open:.2f}  hue={avg_hue if avg_hue is None else round(avg_hue,1)} sat={avg_sat if avg_sat is None else round(avg_sat,1)}",
        ]
        for i, line in enumerate(debug_lines):
            cv2.putText(
                frame,
                line,
                (10, 60 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
            )
 
    if expression_triggered and reaction_image is not None:
        if not reaction_window_open:
            reaction_window_open = True
        cv2.imshow('Reaction', reaction_image)
    elif reaction_window_open:
        cv2.destroyWindow('Reaction')
        reaction_window_open = False
 
    cv2.imshow("Face Skeleton", frame)
 
    if cv2.waitKey(1) & 0xFF == 27:
        break
 
cap.release()
cv2.destroyAllWindows()
detector.close()
 