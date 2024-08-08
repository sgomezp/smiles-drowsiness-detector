# Smile and Drowsiness Detection

This project uses computer vision techniques to detect smiles and drowsiness in real-time using a webcam. Implemented in Python, the project utilizes the streamlit library for the user interface, dlib for facial detection and landmark points, and pygame for sound alerts.

## Installation
Ensure you have the following dependencies installed:

- Python 3.7+
- pip install imutils dlib opencv-python streamlit streamlit-webrtc av pygame

- Additionally, you will need the dlib facial landmark predictor file, which you can download from here.

## Usage
**1. Run the application**

In the terminal, navigate to the folder containing this file and execute:

```bash
streamlit run app.py
```

**2. User Interface**
- Smile Detection: Select "Smile" to activate smile detection. The application will detect if you are smiling and display a message on the screen if so.
- Drowsiness Detection: Select "Drowsiness" to activate drowsiness detection. The application will detect if you are drowsy based on your eye appearance 
and will play an alarm if drowsiness is detected.

# How It Works

## Smile Detection
- **Method**: Uses the Mouth Aspect Ratio (MAR) to determine if the person is smiling.
- **Threshold**: THRES = 0.24 (can be adjusted as needed).
- **Requirements**: Requires a series of frames to confirm a smile.
## Drowsiness Detection
- **Method**: Uses the Eye Aspect Ratio (EAR) to detect if the eyes are closed for an extended period.
- **EAR Threshold**: THRES_EAR = 0.2 (can be adjusted as needed).
- **Frame Counter**: If the eyes are closed for more than MAXIMUM_FRAME_COUNT frames, an alarm is triggered.

## Files
- alarm.wav: Sound file used for the drowsiness alert.
- shape_predictor_68_face_landmarks.dat: dlib model file for facial landmark detection.

