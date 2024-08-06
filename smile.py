from imutils import face_utils
import dlib
import cv2
import time
import streamlit as st

def dist(a,b):
    "Calcula la distancia euclidiana entre dos puntos"
    x1,y1 = a
    x2,y2 = b
    return ((x1-x2)**2 + (y1-y2)**2)**0.5

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


# defining thresholds
THRES = 0.27 # Por ensayo y error este umbral es el que mejor se ajusta
required_smiling_frames = 10  # Numbers of frames required to confirm a smile
required_not_smiling_frames = 10  # Numbers of frames required to confirm a not smile


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


# Save the detected faces data
faces_data = {}
print(f"Faces data: {faces_data}")

while True:
    # Getting out image by webcam
    _, image = cap.read()
    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get faces into webcam's image
    faceRects = detector(gray, 0)

    # For each detected face, find the landmark.
    for (i, rect) in enumerate(faceRects):

        # obteining coordinates of the detected face
        (x, y, w, h) = face_utils.rect_to_bb(rect)

        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extract the mouth region landmarks (points 48-67)
        mouth_points = shape[48:67]  # Index in mouth_points starts from 0

        # Draw on our image, all the finded mouth points cordinate points (x,y)
        for (mx, my) in mouth_points:
            cv2.circle(image, (mx, my), 2, (0, 255, 0), -1)

        calculated_mar = calculate_mar(mouth_points)
        print(f"MAR: {calculated_mar}")

        # Identify the face by its index
        face_id = i
        print(f"Face ID: {face_id}")

        # Initialize the face data if it is not in the dictionary
        if face_id not in faces_data:
            faces_data[face_id] = {
                "smiling_frames_count": 0,
                "not_smiling_frames_count": 0,
                "smiling": False
            }

        # Obteining the current face data
        face_info = faces_data[face_id]
        print(f"Faces data: {faces_data}")

        if calculated_mar < THRES:
            # Increment the smiling frames count
            face_info["smiling_frames_count"] += 1
            face_info["not_smiling_frames_count"] = 0
            print(f"Smiling frames count: {face_info['smiling_frames_count']}")

            # Check if the required smiling frames are reached
            if face_info["smiling_frames_count"] >= required_smiling_frames:
                face_info["smiling"] = True
                print("Entré en sonrisa")
                #cv2.putText(image, "Smiling", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            #1, (0, 255, 0), 2)
        else:
            # Increment the not smiling frames count
            face_info["not_smiling_frames_count"] += 1
            face_info["smiling_frames_count"] = 0

            # Check if the required not smiling frames are reached
            if face_info["not_smiling_frames_count"] >= required_not_smiling_frames:
                face_info["smiling"] = False
                #cv2.putText(image, "Not Smiling", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            #1, (0, 0, 255), 2)

    # show the state of the smiles below the face
    if face_info["smiling"]:
        cv2.putText(image, "Smiling", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)
    else:
        cv2.putText(image, "Not Smiling", (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 2)


    # Show the image
    cv2.imshow("Output", image)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()
