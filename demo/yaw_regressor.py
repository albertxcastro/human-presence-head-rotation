import subprocess
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import fnmatch
import pandas as pd
from tensorflow.keras import backend as K

def count_jpg_files(directory):
    count = 0
    for filename in os.listdir(directory):
        if fnmatch.fnmatch(filename, '*.jpg'):
            count += 1
    return count

def extract_frames_ffmpeg(video_path, output_folder, interval=0.2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    fps = 1 / interval
    command = [
        'ffmpeg',
        '-i', video_path,
        '-vf', f'fps={fps}',
        os.path.join(output_folder, 'frame_%04d.jpg')
    ]

    subprocess.run(command, check=True)
    print(f"Frames extracted to {output_folder}")

def mean_channel(x, axis=-1, keepdims=True):
    return K.mean(x, axis=axis, keepdims=keepdims)

def max_channel(x, axis=-1, keepdims=True):
    return K.max(x, axis=axis, keepdims=keepdims)

def load_model():
    # Try to load the saved model to confirm it works
    try:
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, '../trained_models/yaw_regressor_attention_cnn.keras')
        custom_objects = {
            "mean_channel": mean_channel,
            "max_channel": max_channel
        }
        loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False)
        print("Model loaded successfully!")
        return loaded_model
    except Exception as e:
        print(f"Error loading SavedModel: {e}")
        import traceback
        traceback.print_exc()
        return None

def predict_yaw_from_image(image_path, model):
    # Load the image
    img = cv2.imread(image_path)

    # Check if image was loaded successfully
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return None

    # Convert BGR to RGB (OpenCV loads as BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize to 128x128 (same as your training data)
    img = cv2.resize(img, (128, 128))

    # Normalize to [0, 1] range
    img = img.astype('float32') / 255.0

    # Prepare for prediction (add batch dimension)
    img_batch = np.expand_dims(img, axis=0)

    # Make prediction
    yaw_prediction = model.predict(img_batch, verbose=0)[0][0]

    return yaw_prediction

def process_video(model):
    # Extract frames if they don't exist
    if not os.path.exists(output_frames_path) or count_jpg_files(output_frames_path) == 0:
        extract_frames_ffmpeg(video_input_path, output_frames_path)

    # Create predictions map for this video
    predictions_map = {}
    number_of_frames = count_jpg_files(output_frames_path)
    print(f'Total frames extracted: {number_of_frames}')
    starting_index = 0.2

    # Process each frame
    for i in range(1, number_of_frames + 1):
        if (i < 10):
            path = f'{output_frames_path}/frame_000{i}.jpg'
        elif (i < 100):
            path = f'{output_frames_path}/frame_00{i}.jpg'
        else:
            path = f'{output_frames_path}/frame_0{i}.jpg'
        
        # Check if file exists
        if not os.path.exists(path):
            print(f"Warning: Frame not found at {path}")
            continue
            
        # Preprocess and predict
        preprocessed_image = preprocess_image(path)
        yaw_prediction = model.predict(preprocessed_image, verbose=0)
        
        # Store result
        index = starting_index * i
        predictions_map[index] = yaw_prediction[0][0]
        
    # Save to CSV
    predictions_df = pd.DataFrame(list(predictions_map.items()), columns=['Time', 'Yaw'])
    csv_path = f'{folder}/{video_name_no_extension}_yaw_data.csv'
    predictions_df.to_csv(csv_path, index=False)
    print(f"Saved yaw data to {csv_path}")