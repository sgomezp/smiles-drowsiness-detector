from imutils import face_utils
import dlib
import cv2
from pygame import mixer

MINIMUM_EAR = 0.2 # More than this the eyes are open
MAXIMUM_FRAME_COUNT = 10  # More than this consecutive frames the eyes are closed
EYE_CLOSED_COUNTER = 0  # Counter of consecutive frames the eyes are closed



def dist(a,b):
    "Calculate the euclidean distance between two points"
    x1,y1 = a
    x2,y2 = b
    return ((x1-x2)**2 + (y1-y2)**2)**0.5


def eye_aspect_ratio(eye):
    "Calculate the Eye Aspect Ratio (EAR)"

    # Compute the euclidean distances between the two sets of vertical eye landmarks (x, y)-coordinates
    A = dist(eye[1], eye[5])
    B = dist(eye[2], eye[4])

    # Compute the euclidean distance between the horizontal eye landmark (x, y)-coordinates
    C = dist(eye[0], eye[3])

    # Compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # Return the eye aspect ratio
    return ear

mixer.init()
sound = mixer.Sound('alarm.wav')

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

leftEyeStart, leftEyeEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
rightEyeStart, rightEyeEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


while True:
    # Getting out image by webcam
    _, image = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces into webcam's image
    faceRects = detector(gray, 0)

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(faceRects):
        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[leftEyeStart:leftEyeEnd]
        rightEye = shape[rightEyeStart:rightEyeEnd]


        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        print(f"leftEAR: {leftEAR}")
        print(f"rightEAR: {rightEAR}")

        ear = (leftEAR + rightEAR) / 2.0
        print(f"EAR: {ear}")

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < MINIMUM_EAR:
            EYE_CLOSED_COUNTER += 1
        else:
            EYE_CLOSED_COUNTER = 0

        if EYE_CLOSED_COUNTER >= MAXIMUM_FRAME_COUNT:
            try:
                sound.play()
            except:
                pass
        else:
            try:
                sound.stop()
            except:
                pass



    # Show the image
    cv2.imshow("Output", image)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
