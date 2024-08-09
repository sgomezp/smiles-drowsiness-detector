from imutils import face_utils
import dlib
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import av
from pygame import mixer


# Initialize the sound mixer
mixer.init()
sound = mixer.Sound('alarm.wav')

# Thresholds definition
THRES = 0.24
THRES_EAR = 0.2
required_smiling_frames = 7
required_not_smiling_frames = 7
MAXIMUM_FRAME_COUNT = 10

# Inicialización de Dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class BaseDetector(VideoProcessorBase):
    def dist(selfself, a, b):
        x1, y1 = a
        x2, y2 = b
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def get_face_id(self, rect):
        x,y,w,h = face_utils.rect_to_bb(rect)
        return f"{x}-{y}-{w}-{h}"

class SmileDetector(BaseDetector):
    def __init__(self):
        self.faces_data = {}

    def calculate_mar(self, mouth_points):
        "Calculate the Mouth Aspect Ratio (MAR)"

        # Distance between mouth corner points (48 and 54)
        A = self.dist(mouth_points[0], mouth_points[6])

        # Distance between top and bottom outer- left lip points (50 and 58)
        B = self.dist(mouth_points[2], mouth_points[10])

        # Distance between top and bottom outer -right lip points (52 and 56)
        C = self.dist(mouth_points[4], mouth_points[8])

        # Distance between top and bottom mid-lip points (51 and 57)
        D = self.dist(mouth_points[3], mouth_points[9])

        # Calculate the mouth aspect ratio
        mar = (B + C + D) / (3.0 * A)
        return mar


    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faceRects = detector(gray, 0)

        current_faces = set()

        for rect in faceRects:
            face_id = self.get_face_id(rect)
            current_faces.add(face_id)
            if face_id not in self.faces_data:
                self.faces_data[face_id] = {
                    "smiling_frames_count": 0,
                    "not_smiling_frames_count": 0,
                    "smiling": False
                }
            face_info = self.faces_data[face_id]

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            mouth_points = shape[48:67]

            for (mx, my) in mouth_points:
                cv2.circle(image, (mx, my), 2, (0, 255, 0), -1)

            calculated_mar = self.calculate_mar(mouth_points)
            (x,y,w,h) = face_utils.rect_to_bb(rect)

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
                cv2.putText(
                    image, "Smiling", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
            else:
                cv2.putText(
                    image, "Not Smiling", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)

        # Remove faces that are not detected
        self.faces_data = {face_id: data for face_id, data in self.faces_data.items() if face_id in current_faces}

        return av.VideoFrame.from_ndarray(image, format="bgr24")

class DrownsinessDetector(BaseDetector):
    def __init__(self):
        self.faces_data = {}

    def eye_aspect_ratio(self, eye):
        "Calculate the Eye Aspect Ratio (EAR)"

        # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
        A = self.dist(eye[1], eye[5])
        B = self.dist(eye[2], eye[4])

        # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
        C = self.dist(eye[0], eye[3])

        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)

        # Return the eye aspect ratio
        return ear


    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faceRects = detector(gray, 0)

        current_faces = set()

        leftEyeStart, leftEyeEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        rightEyeStart, rightEyeEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        for rect in faceRects:
            face_id = self.get_face_id(rect)
            current_faces.add(face_id)

            if face_id not in self.faces_data:
                self.faces_data[face_id] = {
                    "eye_closed_counter": 0
                }

            face_info = self.faces_data[face_id]

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[leftEyeStart:leftEyeEnd]
            rightEye = shape[rightEyeStart:rightEyeEnd]

            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            #cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
            #cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < THRES_EAR:
                self.eye_closed_counter += 1
            else:
                self.eye_closed_counter = 0

            if self.eye_closed_counter >= MAXIMUM_FRAME_COUNT:
                cv2.putText(image, "DROWSINESS DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                try:
                    sound.play()
                except:
                    pass
            else:
                try:
                    sound.stop()
                except:
                    pass

        # Remove faces that are not detected
        self.faces_data = {face_id: data for face_id, data in self.faces_data.items() if face_id in current_faces}

        return av.VideoFrame.from_ndarray(image, format="bgr24")


RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

st.title("Smile and Drowsiness Detection")
detector_type = st.radio("Select detector type", ("Smile", "Drowsiness"))

if detector_type == "Smile":
    webrtc_streamer(key="streamlit-example",
                video_processor_factory=SmileDetector,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False})
elif detector_type == "Drowsiness":
    webrtc_streamer(key="drowsiness-detection",
                video_processor_factory=DrownsinessDetector,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False})


