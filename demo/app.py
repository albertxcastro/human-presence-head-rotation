import os
import cv2
import tkinter as tk
from tkinter import ttk
import PIL.Image, PIL.ImageTk
import shutil
from datetime import datetime
import threading
import queue
import time
import joblib
import numpy as np
import pandas as pd
import rotation_classifier
import yaw_regressor

class App:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.resizable(False, False)
        
        # Create folder for recordings if it doesn't exist
        self.recordings_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
        self.cleanup_recordings()
        
        # Camera variables
        self.available_cameras = self.get_available_cameras()
        self.current_camera_index = 0 if self.available_cameras else None
        self.vid = None
        self.camera_lock = threading.Lock()
        self.connect_to_camera()
        
        # Recording variables
        self.recording = False
        self.video_writer = None
        self.output_file = None
        
        # Threading variables
        self.frame_queue = queue.Queue(maxsize=2)
        self.processing_thread = None
        self.is_running = True
        self.last_valid_frame = None
        self.camera_available = False
        self.last_update_time = 0
        self.update_interval = 1.0 / 30.0  # 30 FPS for UI updates
        
        # Models
        self.yaw_model = None
        self.classification_model = None
        self.classification_thread = None
        
        # Create layout
        self.create_widgets()
        
        # Bind window focus event
        self.window.bind('<FocusIn>', self.on_focus_in)
        
        # Start video processing thread
        self.start_processing_thread()
        
        # Initialize update
        self.update()
        self.window.mainloop()
    
    def get_available_cameras(self):
        """Find all available camera devices"""
        available_cameras = []
        max_cameras_to_check = 5  # Check first 5 camera indices
        
        for i in range(max_cameras_to_check):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                camera_name = f"Camera {i}"
                available_cameras.append((i, camera_name))
                cap.release()
        
        return available_cameras
    
    def connect_to_camera(self):
        """Connect to the currently selected camera"""
        with self.camera_lock:
            # Release any existing camera
            if self.vid is not None and self.vid.isOpened():
                self.vid.release()
                self.vid = None
            
            # Connect to selected camera if available
            if self.current_camera_index is not None:
                self.vid = cv2.VideoCapture(self.current_camera_index)
                
                # Check if connection was successful
                if self.vid.isOpened():
                    # Set resolution to 1280x720 (720p)
                    self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    # Set zoom to minimum (zoomed out)
                    self.vid.set(cv2.CAP_PROP_ZOOM, 0)
                    # Set focus to auto
                    self.vid.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                else:
                    self.vid = None
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.window)
        main_frame.pack(padx=10, pady=10)
        
        # Camera frame
        self.camera_frame = ttk.Frame(main_frame)
        self.camera_frame.grid(row=0, column=0, padx=10, pady=10)
        
        # Camera canvas - increased size for higher resolution
        self.canvas = tk.Canvas(self.camera_frame, width=1280, height=720)
        self.canvas.pack()
        
        # Control panel frame
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=0, column=1, padx=10, pady=10, sticky="n")
        
        # Camera selection
        camera_frame = ttk.LabelFrame(control_frame, text="Camera Selection")
        camera_frame.pack(fill="x", pady=5, padx=10)
        
        # Camera dropdown
        self.camera_var = tk.StringVar()
        self.camera_dropdown = ttk.Combobox(camera_frame, textvariable=self.camera_var, state="readonly")
        self.camera_dropdown.pack(fill="x", pady=5, padx=10)
        
        # Populate camera dropdown
        self.update_camera_dropdown()
        
        # Add dropdown selection callback - bind to both selection and focus out
        self.camera_dropdown.bind("<<ComboboxSelected>>", self.on_camera_selected)
        self.camera_dropdown.bind("<FocusOut>", self.on_camera_selected)
        
        # Create a style for buttons
        style = ttk.Style()
        style.configure('Custom.TButton', padding=5)
        
        # Record button
        self.record_btn = ttk.Button(control_frame, text="Record", style='Custom.TButton', command=self.toggle_recording)
        self.record_btn.pack(fill="x", pady=5, padx=10)
        
        # Classify button
        self.classify_btn = ttk.Button(control_frame, text="Classify", style='Custom.TButton', state="disabled", command=self.classify)
        self.classify_btn.pack(fill="x", pady=5, padx=10)
        
        # Status label
        self.status_label = ttk.Label(control_frame, text="", wraplength=250, justify=tk.LEFT)
        self.status_label.pack(fill="x", pady=5, padx=10)
        
        # Add a spacer to push the close button to the bottom
        spacer = ttk.Frame(control_frame)
        spacer.pack(fill="y", expand=True)
        
        # Close button
        self.close_btn = ttk.Button(control_frame, text="Close", style='Custom.TButton', command=self.close_app)
        self.close_btn.pack(fill="x", pady=5, padx=10)
        
        # Update UI based on camera availability
        self.update_ui_camera_status()
    
    def update_camera_dropdown(self):
        """Update the camera dropdown with available cameras"""
        if not self.available_cameras:
            self.camera_dropdown.set("No cameras available")
            self.camera_dropdown["values"] = []
        else:
            camera_names = [name for _, name in self.available_cameras]
            self.camera_dropdown["values"] = camera_names
            if self.current_camera_index is not None:
                camera_idx = [idx for idx, (i, _) in enumerate(self.available_cameras) if i == self.current_camera_index]
                if camera_idx:
                    self.camera_dropdown.current(camera_idx[0])
    
    def on_camera_selected(self, event):
        """Handle camera selection from dropdown"""
        if not self.available_cameras:
            return
            
        selected_idx = self.camera_dropdown.current()
        print(f"Selected camera index: {selected_idx}")  # Debug log
        
        if selected_idx >= 0 and selected_idx < len(self.available_cameras):
            new_camera_idx = self.available_cameras[selected_idx][0]
            print(f"Switching to camera index: {new_camera_idx}")  # Debug log
            
            # Stop the processing thread
            self.is_running = False
            if self.processing_thread is not None:
                self.processing_thread.join(timeout=1.0)
            
            # Clear the frame queue
            while not self.frame_queue.empty():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Switch camera
            self.current_camera_index = new_camera_idx
            self.connect_to_camera()
            
            # Restart the processing thread
            self.is_running = True
            self.start_processing_thread()
            
            self.update_ui_camera_status()
    
    def update_ui_camera_status(self):
        """Update UI based on camera availability"""
        camera_available = self.vid is not None and self.vid.isOpened()
        
        # Enable/disable recording based on camera availability
        if camera_available:
            self.record_btn["state"] = "normal"
            self.status_label.config(text="Camera ready")
        else:
            self.record_btn["state"] = "disabled"
            self.classify_btn["state"] = "disabled"
            self.status_label.config(text="No camera available")
    
    def start_processing_thread(self):
        """Start the video processing thread"""
        self.processing_thread = threading.Thread(target=self.process_video, daemon=True)
        self.processing_thread.start()
    
    def process_video(self):
        """Process video frames in a separate thread"""
        frame_count = 0
        while self.is_running:
            with self.camera_lock:
                if self.vid is not None and self.vid.isOpened():
                    ret, frame = self.vid.read()
                    if ret:
                        self.camera_available = True
                        # Mirror the frame horizontally
                        frame = cv2.flip(frame, 1)
                        
                        # Write frame to video file if recording
                        if self.recording and self.video_writer is not None:
                            try:
                                self.video_writer.write(frame)
                                frame_count += 1
                            except Exception as e:
                                print(f"Error writing frame: {e}")
                                self.recording = False
                                self.record_btn.config(text="Record")
                                self.status_label.config(text="Recording error - stopped")
                        
                        # Convert frame to PIL image for the queue
                        pil_img = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                        # Put frame in queue if there's space
                        try:
                            self.frame_queue.put_nowait(pil_img)
                        except queue.Full:
                            # If queue is full, remove old frame and add new one
                            try:
                                self.frame_queue.get_nowait()
                                self.frame_queue.put_nowait(pil_img)
                            except queue.Empty:
                                pass
                    else:
                        self.camera_available = False
                else:
                    self.camera_available = False
                    # If no camera, sleep briefly to prevent CPU spinning
                    threading.Event().wait(0.1)
    
    def update(self):
        """Update the UI with the latest frame"""
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval:
            self.last_update_time = current_time
            
            try:
                # Get the latest frame from the queue
                pil_img = self.frame_queue.get_nowait()
                self.last_valid_frame = pil_img
                
                # Convert to format suitable for tkinter
                self.photo = PIL.ImageTk.PhotoImage(image=pil_img)
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                
            except queue.Empty:
                if self.camera_available and self.last_valid_frame is not None:
                    # If camera is available but queue is empty, use last valid frame
                    self.photo = PIL.ImageTk.PhotoImage(image=self.last_valid_frame)
                    self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
                else:
                    # Only show "No camera available" if we're sure there's no camera
                    self.canvas.delete("all")
                    self.canvas.create_text(640, 360, text="No camera available", font=("Arial", 20))

            # Add or remove overlay text based on recording state
            if self.camera_available:
                if self.recording:
                    self.add_overlay_text()
                else:
                    self.canvas.delete("overlay_text")

        # Call update method after 15ms (about 60 fps)
        self.window.after(15, self.update)
    
    def toggle_recording(self):
        # Check if camera is available
        if self.vid is None or not self.vid.isOpened():
            self.status_label.config(text="No camera available")
            return
            
        if not self.recording:
            # Clean up existing recordings before starting new one
            self.cleanup_recordings()
            
            # Start recording
            self.recording = True
            self.record_btn.config(text="Stop")
            
            # Create output file
            os.makedirs(self.recordings_folder, exist_ok=True)
            self.output_file = os.path.join(self.recordings_folder, f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            
            # Get camera properties
            width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(self.vid.get(cv2.CAP_PROP_FPS))
            if fps <= 0:
                fps = 30  # Default to 30 fps if not available
            
            # Create video writer with H.264 codec
            fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
            self.video_writer = cv2.VideoWriter(
                self.output_file, 
                fourcc, 
                fps,
                (width, height)
            )
            
            # Disable camera selection and classify button during recording
            self.camera_dropdown["state"] = "disabled"
            self.classify_btn.config(state="disabled")
            
            self.status_label.config(text="Recording...")
            
        else:
            # Stop recording
            self.recording = False
            self.record_btn.config(text="Record")
            
            # Release video writer
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            
            # Re-enable camera selection
            self.camera_dropdown["state"] = "readonly"
            
            # Check if video was saved and update UI
            if self.output_file and os.path.exists(self.output_file) and os.path.getsize(self.output_file) > 0:
                self.status_label.config(text=f"Saved to {os.path.basename(self.output_file)}")
                self.classify_btn.config(state="normal")
            else:
                self.status_label.config(text="Recording failed or was empty.")
                self.classify_btn.config(state="disabled")
    
    def classify(self):
        """Classify the recorded video"""
        if self.classification_thread and self.classification_thread.is_alive():
            self.status_label.config(text="Classification is already in progress.")
            return

        self.classify_btn.config(state="disabled")
        self.status_label.config(text="Starting classification...")

        self.classification_thread = threading.Thread(target=self._classify_task, daemon=True)
        self.classification_thread.start()

    def _classify_task(self):
        try:
            # PART 1: YAW PREDICTION
            self.window.after(0, lambda: self.status_label.config(text="Loading Yaw Regressor model..."))
            if self.yaw_model is None:
                self.yaw_model = yaw_regressor.load_model()
                if self.yaw_model is None:
                    self.window.after(0, lambda: self.status_label.config(text="Failed to load yaw model."))
                    return

            if not self.output_file or not os.path.exists(self.output_file):
                self.window.after(0, lambda: self.status_label.config(text="No recording found to classify."))
                return

            video_path = self.output_file
            video_name_no_ext = os.path.splitext(os.path.basename(video_path))[0]
            frames_folder = os.path.join(self.recordings_folder, f"{video_name_no_ext}_frames")

            self.window.after(0, lambda: self.status_label.config(text="Extracting frames from video..."))
            if os.path.exists(frames_folder):
                shutil.rmtree(frames_folder)

            yaw_regressor.extract_frames_ffmpeg(video_path, frames_folder)

            frame_files = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])
            if not frame_files:
                self.window.after(0, lambda: self.status_label.config(text="No frames extracted from video."))
                return

            self.window.after(0, lambda: self.status_label.config(text=f"Predicting yaw angles for {len(frame_files)} extracted frames..."))
            
            predictions = {}
            interval = 0.2
            for i, frame_file in enumerate(frame_files):
                frame_path = os.path.join(frames_folder, frame_file)
                yaw_prediction = yaw_regressor.predict_yaw_from_image(frame_path, self.yaw_model)
                if yaw_prediction is not None:
                    predictions[i * interval] = yaw_prediction

            csv_path = os.path.join(self.recordings_folder, f"{video_name_no_ext}_yaw_data.csv")
            pd.DataFrame(list(predictions.items()), columns=['Time', 'Yaw']).to_csv(csv_path, index=False)
            self.window.after(0, lambda: self.status_label.config(text=f"Yaw data saved to {os.path.basename(csv_path)}"))

            # PART 2: CLASSIFICATION
            self.window.after(0, lambda: self.status_label.config(text="Loading classification model..."))
            if self.classification_model is None:
                model_path = os.path.join(os.path.dirname(__file__), '..', 'trained_models', 'time_series_logistic_regression_classifier.joblib')
                if not os.path.exists(model_path):
                    self.window.after(0, lambda: self.status_label.config(text="Classification model not found."))
                    return
                
                # The pickled model expects `_raw_to_feats` to be in the `__main__` module.
                # We import it from our classifier module and inject it into `sys.modules`
                # so that `joblib.load` can find it.
                import sys
                from rotation_classifier import _raw_to_feats
                sys.modules['__main__']._raw_to_feats = _raw_to_feats
                
                self.classification_model = joblib.load(model_path)

            self.window.after(0, lambda: self.status_label.config(text="Classifying rotation pattern..."))
            sequence = rotation_classifier.load_and_preprocess_csv(csv_path)
            
            sequence_batch = np.expand_dims(sequence, axis=0)
            
            prediction = self.classification_model.predict(sequence_batch)
            probability = self.classification_model.predict_proba(sequence_batch)
            
            result_label = "Correct" if prediction[0] == 1 else "Incorrect"
            prob_percent = probability[0][1] * 100 if prediction[0] == 1 else probability[0][0] * 100

            result_color = "green" if result_label == "Correct" else "red"
            self.window.after(0, lambda: self.status_label.config(
                text=f"Result: {result_label} Rotation (Confidence: {prob_percent:.2f}%)",
                foreground=result_color
            ))
            
        except Exception as e:
            self.window.after(0, lambda: self.status_label.config(text=f"Error during classification: {e}"))
            import traceback
            traceback.print_exc()
        finally:
            if self.output_file and os.path.exists(self.output_file):
                 self.window.after(0, lambda: self.classify_btn.config(state="normal"))
            else:
                 self.window.after(0, lambda: self.classify_btn.config(state="disabled"))
    
    def cleanup_recordings(self):
        """Clean up recordings folder by removing it and recreating it."""
        if os.path.exists(self.recordings_folder):
            try:
                shutil.rmtree(self.recordings_folder)
            except Exception as e:
                print(f"Error cleaning up recordings folder: {e}")
        
        os.makedirs(self.recordings_folder, exist_ok=True)
    
    def close_app(self):
        """Properly close the application"""
        # Stop recording if active
        if self.recording:
            self.toggle_recording()
        
        # Stop the processing thread
        self.is_running = False
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=1.0)
        
        # Release camera and video writer
        if hasattr(self, 'vid') and self.vid is not None:
            self.vid.release()
        if hasattr(self, 'video_writer') and self.video_writer is not None:
            self.video_writer.release()
        
        # Destroy the window
        self.window.destroy()

    def add_overlay_text(self):
        """Adds an overlay text on the canvas."""
        self.canvas.delete("overlay_text")
        text = "Please rotate your head horizontally"
        font_config = ("Arial", 28, "bold")
        
        canvas_width = 1280
        x = canvas_width / 2
        y = 50  # Position at the top

        # Text with a black background for readability
        # Create a background rectangle
        text_width = len(text) * 14  # Approximate width
        self.canvas.create_rectangle(
            x - text_width / 2 - 20, y - 25, x + text_width / 2 + 20, y + 25,
            fill='black', outline='', tags='overlay_text'
        )
        self.canvas.create_text(x, y, text=text, font=font_config, fill="white", tags="overlay_text")

    def on_focus_in(self, event):
        """Handle window gaining focus to process pending UI events."""
        self.window.update_idletasks()

# Create application window
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root, "ML Model Testing UI")