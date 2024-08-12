# Smile and Drowsiness Detection

This project uses computer vision techniques to detect smiles and drowsiness in real-time using a webcam. 
It can detect more than one face at a time. Implemented in Python, the project utilizes the Streamlit library for 
the user interface, dlib for facial detection and landmark points, and Pygame for sound alerts.

# File Structure Description:
- **detection.py**: Main file for the Streamlit application.
- **.gitignore**: File to specify which files and directories to ignore in the repository.
- **requirements.txt**: File containing the project dependencies.
- **package.txt**: This file installs system dependencies needed in deployment environments like Streamlit Share. It 
  ensures OpenCV works by including libraries like libGL1 for graphics.
- **alarm.wav**: Sound file used for the drowsiness alert.
- **shape_predictor_68_face_landmarks.dat**: dlib model file for facial landmark detection. [link](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat)

# Usage Instructions
1. Clone the repository to your local machine. [link](https://github.com/sgomezp/smiles-drowsiness-detector.git)
2. Navigate to the project directory.
3. Optionally, create a virtual environment for the project. (highly recommended)
4. Install dlip using the instructions below.
5. Install the project dependencies using the following command:
   `pip install -r requirements.txt`
6. Download the dlib facial landmark predictor file from the link provided below.
7. Run the Streamlit application using the following command:
   `streamlit run detection.py`
8. Select the desired detection mode (Smile or Drowsiness) from the dropdown menu.
9. Smile Detection: The application will detect if you are smiling and display a message on the screen if so.
10. Drowsiness Detection: The application will detect if you are drowsy based on your eye appearance and will play an alarm if drowsiness is detected.


## Installing dlib
Before installing the dependencies from requirements.txt, make sure to install dlib, as it requires 
CMake and a C++ compiler.

### Install CMake:

##### Windows

Download and install CMake from cmake.org.
If you don't have a C++ compiler installed, download and install Visual Studio (the free "Community" edition is 
sufficient) and select "Desktop development with C++" during the installation.

##### Ubuntu 
`sudo apt-get install cmake`

##### macOS
`brew install cmake`

### Install dlib:

Run the following command in your terminal:

`pip install dlib`

If you encounter any issues during the installation, refer to the official dlib installation guide for

# How It Works

## Smile Detection
- **Method**: Uses the Mouth Aspect Ratio (MAR) to determine if the person is smiling.
- **Threshold**: THRES = 0.24 (can be adjusted as needed).
- **Requirements**: Requires a series of frames to confirm a smile.
## Drowsiness Detection
- **Method**: Uses the Eye Aspect Ratio (EAR) to detect if the eyes are closed for an extended period.
- **EAR Threshold**: THRES_EAR = 0.15 (can be adjusted as needed).
- **Frame Counter**: If the eyes are closed for more than MAXIMUM_FRAME_COUNT frames, an alarm is triggered.

# Limitations
- **Smile Detection**: The smile detection algorithm may not work well with people wearing masks or with facial hair.
- **Drowsiness Detection**: The drowsiness detection algorithm may not work well with people wearing glasses or with certain eye shapes.
- **Lighting Conditions**: The performance of the detection algorithms may vary based on lighting conditions.
- **Background Noise**: The drowsiness alarm may not be effective in noisy environments.
- **Accuracy**: The accuracy of the detection algorithms may vary based on the quality of the webcam and the resolution of the video feed.
- **Speed**: The speed of the detection algorithms may vary based on the processing power of the system.


# Future Improvements
- **Dockerization**: Dockerize the application for easier deployment and scalability.

# Observations
For optimal performance of the facial detection and recognition features, it is highly recommended
to ensure good lighting conditions. Proper lighting enhances the accuracy of face detection and 
improves the effectiveness of smile and drowsiness detection algorithms.

## Performance Variability
Please be aware that detection performance may vary based on environmental conditions, such as lighting 
and background. In low-light conditions or with significant background distractions, the system 
may struggle to accurately detect faces, smiles, or signs of drowsiness. For best results, 
conduct testing and use the system in controlled lighting environments.

