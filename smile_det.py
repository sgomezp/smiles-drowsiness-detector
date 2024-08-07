from imutils import face_utils
import dlib
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av

# Función para calcular la distancia euclidiana entre dos puntos
def dist(a, b):
    x1, y1 = a
    x2, y2 = b
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

# Función para calcular el Mouth Aspect Ratio (MAR)
def calculate_mar(mouth_points):
    "Calculate the Mouth Aspect Ratio (MAR)"

    # Distance between mouth corner points (48 and 54)
    A = dist(mouth_points[0], mouth_points[6])

    # Distance between top and bottom outer- left lip points (50 and 58)
    B = dist(mouth_points[2], mouth_points[10])

    # Distance between top and bottom outer -right lip points (52 and 56)
    C = dist(mouth_points[4], mouth_points[8])

    # Distance between top and bottom mid-lip points (51 and 57)
    D = dist(mouth_points[3], mouth_points[9])

    # Calculate the mouth aspect ratio
    mar = (B + C + D) / (3.0 * A)
    return mar

# Definición de umbrales
THRES = 0.27
required_smiling_frames = 10
required_not_smiling_frames = 10

# Inicialización de Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

faces_data = {}

def video_frame_callback(frame):
    global faces_data

    image = frame.to_ndarray(format="bgr24")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceRects = detector(gray, 0)

    for (i, rect) in enumerate(faceRects):
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        mouth_points = shape[48:67]

        for (mx, my) in mouth_points:
            cv2.circle(image, (mx, my), 2, (0, 255, 0), -1)

        calculated_mar = calculate_mar(mouth_points)
        face_id = i

        if face_id not in faces_data:
            faces_data[face_id] = {
                "smiling_frames_count": 0,
                "not_smiling_frames_count": 0,
                "smiling": False
            }

        face_info = faces_data[face_id]

        if calculated_mar < THRES:
            face_info["smiling_frames_count"] += 1
            face_info["not_smiling_frames_count"] = 0

            if face_info["smiling_frames_count"] >= required_smiling_frames:
                face_info["smiling"] = True
        else:
            face_info["not_smiling_frames_count"] += 1
            face_info["smiling_frames_count"] = 0

            if face_info["not_smiling_frames_count"] >= required_not_smiling_frames:
                face_info["smiling"] = False

        if face_info["smiling"]:
            cv2.putText(image, "Smiling", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        else:
            cv2.putText(image, "Not Smiling", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2)

    return av.VideoFrame.from_ndarray(image, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

st.title("Smile Detection with Streamlit")

webrtc_streamer(key="example",
                video_frame_callback=video_frame_callback,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False})
