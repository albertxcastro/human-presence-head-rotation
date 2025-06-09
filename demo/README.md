# Head Rotation Classification Demo

This project demonstrates a novel method for human presence detection in video recordings by analyzing head rotation patterns. It uses a Convolutional Neural Network (CNN) to identify temporal patterns of yaw (horizontal head rotation) movements to differentiate genuine human presence from absence.

This work aims to showcase the effectiveness of yaw-based temporal pattern analysis in human presence detection, highlighting a practical and scalable technique for real-time video applications.

## How it Works

The application follows these steps to classify a video:
1.  A short video is recorded via the user's webcam.
2.  The video is processed to extract individual frames.
3.  A CNN-based yaw regressor model (`yaw_regressor_attention_cnn.keras`) predicts the yaw angle (horizontal head rotation) for each frame.
4.  The time-series data of yaw angles is transformed into a feature vector, capturing characteristics like the number of peaks and troughs in movement.
5.  A classification model (`rotation_classifier_pipeline.joblib`) analyzes these features to determine if the rotation pattern corresponds to genuine human presence.
6.  The final classification (e.g., "Human Detected") is displayed to the user.

## Features

- Live webcam display and camera selection.
- Video recording functionality.
- Classification of recorded video to detect human presence based on head rotation patterns.
- Visual feedback of the classification result on the UI.

## Requirements

- Python 3.6+
- OpenCV
- Tkinter
- PIL (Pillow)
- Pandas
- Numpy
- Scikit-learn
- Tensorflow
- ffmpeg

## Installation

First, install `ffmpeg`. On macOS, you can use [Homebrew](https://brew.sh/):
```
brew install ffmpeg
```
For other operating systems, please refer to the official [ffmpeg documentation](https://ffmpeg.org/download.html).

Then, install the required Python packages:
```
pip install opencv-python pillow pandas numpy scikit-learn tensorflow
```

## Models

This application uses two pre-trained models:
1.  `yaw_regressor_attention_cnn.keras`: A TensorFlow/Keras model to predict the yaw rotation from an image.
2.  `rotation_classifier_pipeline.joblib`: A Scikit-learn pipeline to classify head rotation patterns.

These models must be located in a `trained_models` directory at the root of this project.

## Usage

Run the application from the `demo` directory:
```
python app.py
```

1. Select a camera from the dropdown menu.
2. Click "Record" to start recording a short video of your head movements.
3. Click "Stop" to end the recording.
4. Click "Classify" to process the recorded video. The application will analyze the head rotation and display the classification result.

## Notes

- The application automatically detects available cameras.
- The `recordings` folder is cleared each time the application starts.
- Recordings are saved in the `recordings` folder in MP4 format.