# import general dependencies
import cv2
import numpy as np
import time
from datetime import datetime
import os
from collections import deque

# import config
from config import VIDEO_PATH, VIDEO_URL, FLIP_CODE, TARGET_FPS, MODEL_PATH

# import setup functions
from setup import define_crop_area, define_traffic_light
from setup.manage_config import save_config, load_config

# import functions that return stream 
# 1. Generate Stream
from stream import get_raw_stream_generator, get_camera_stream_generator 
# 2. Optimize Stream
from stream import crop_generator, reduce_framerate, flip_stream_generator, enhance_light_generator
# 3. Analyze Stream
from stream import detect_redlight_stream, detect_people_stream
# 4. Draw Stream
from stream import draw_objects_stream, show_stream
from stream import invert_colors_generator

# import tracking class
from analysis import CentroidTracker





# we have following toggleble options:
    # 1. Gamma Correction
    # 2. CLAHE
    # 3. Emojis
    # 4. Sound
    # 5. Image Inversion
    # 6. 


# --- UNIVERSAL IMPORT BLOCK ---
try:
    # TFLITE RUNTIME (For Raspberry Pi)
    import tflite_runtime.interpreter as tflite
    print("Using TFLite Runtime")
except ImportError:
    # FULL TENSORFLOW (For Mac / PC)
    try:
        import tensorflow.lite as tflite
        print("Using Full TensorFlow")
    except ImportError:
        print("Warning: TensorFlow not found. AI features will be disabled.")
        tflite = None


def main_modular(use_enhancement=True, invert_colors=False, use_emojis=True, sample_source=False, url_source=False, target_fps=TARGET_FPS, record_session=False, record_jaywalking=False, use_predefined_setup=False):
    print(f"--- Starting Modular Surveillance ---")
    print(f"Features: Enhance={use_enhancement}, Invert={invert_colors}, Emojis={use_emojis}, SAMPLE_SOURCE={sample_source}, FPS={target_fps}")
    print(f"Recording: Session={record_session}, Jaywalking={record_jaywalking}")
    print(f"Setup: Predefined={use_predefined_setup}")

    if url_source:
        print("1. Initializing URL Source...")
        fps, raw_stream = get_raw_stream_generator(VIDEO_URL, loop=True)
    elif sample_source:
        print("1. Initializing Sample Source...")
        fps, raw_stream = get_raw_stream_generator(VIDEO_PATH, loop=True)
    else: 
        print("1. Initializing Camera Source...")
        # Use device index 0 (or 1 if 0 is built-in webcam)
        try:
            fps, raw_stream = get_camera_stream_generator(0)
        except ValueError:
            print("Camera 0 failed, trying Camera 1...")
            fps, raw_stream = get_camera_stream_generator(1)

    stream = reduce_framerate(raw_stream, fps, target_fps)

    print("2. Define Crop Area...")
    crop_rect = None
    tl_configs = []
    
    if use_predefined_setup:
        crop_rect, tl_configs = load_config()
        if crop_rect is None or not tl_configs:
            print("Predefined setup not found or invalid. Falling back to manual setup.")
            use_predefined_setup = False # Fallback

    if not use_predefined_setup:
         crop_rect = define_crop_area(stream)
    
    stream = crop_generator(stream, crop_rect)

    stream = flip_stream_generator(stream, FLIP_CODE)

    print("3. Starting Setup Phase...")
    if not use_predefined_setup:
        # This now returns a LIST of configs
        tl_configs = define_traffic_light(stream)
        # Save for next time
        save_config(crop_rect, tl_configs)

    if use_enhancement:
        print("4. Optimizing Stream (Enhance Light)...")
        stream = enhance_light_generator(stream)

    print("5. Starting Modular Surveillance Phase...")
    
    # --- Modular Pipeline ---
    
    # 1. Initialize State needed for modules
    street_masks = []
    
    # Pre-calculate masks for each config
    if crop_rect:
        w_crop = crop_rect[2] - crop_rect[0]
        h_crop = crop_rect[3] - crop_rect[1]
    else:
        from config import FULL_W, FULL_H
        w_crop = FULL_W
        h_crop = FULL_H

    for config in tl_configs:
        street_pts = np.array(config.get("street", []), np.int32)
        mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
        if len(street_pts) > 0:
            cv2.fillPoly(mask, [street_pts], 255)
        street_masks.append(mask)
    
    # 2. Init AI & Tracker
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()
    tracker = CentroidTracker()

    # 3. Build Pipeline
    
    # Stream -> Red Light Analysis
    # stream_data yields: (frame, results_list)
    # results_list: [(status, color), ...]
    stream_data = detect_redlight_stream(stream, tl_configs)
    
    # Red Light -> People Detection
    # stream_data yields: (frame, results_list, people)
    stream_data = detect_people_stream(stream_data, interpreter, tracker)
    
    # Invert Colors
    if invert_colors:
        from stream.optimize_stream import invert_stream_data_generator
        stream_data = invert_stream_data_generator(stream_data)

    # People -> Drawing
    # We pass the LISTs now
    stream = draw_objects_stream(stream_data, tl_configs, street_masks, use_emojis=use_emojis)
    
    # 4. Display & Record Loop
    
    # --- RECORDING SETUP ---
    session_writer = None
    jaywalk_writer = None
    
    # Ensure recordings directory exists
    # Ensure recordings directory exists
    if record_session or record_jaywalking:
        if not os.path.exists("recordings"):
            os.makedirs("recordings")
        
        # Define codec for both recording types
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Session Recording
    if record_session:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"recordings/rec_session_{timestamp}.mp4"
        print(f"Session Recording enabled. Saving to: {filename}")
        # Writer initialized on first frame

    # Jaywalking Recording State
    frame_buffer_len = int(10 * target_fps) # 10 seconds before
    post_event_frames = int(3 * target_fps) # 3 seconds after
    frame_buffer = deque(maxlen=frame_buffer_len)
    
    jaywalk_recording_counter = 0 # How many frames left to record
    is_recording_jaywalk = False
    
    window_name = "Modular Surveillance"
    
    try:
        for frame, is_jaywalking in stream:
             # Get dimensions and init writers if needed
            h, w = frame.shape[:2]
            
            if record_session and session_writer is None:
                session_writer = cv2.VideoWriter(filename, fourcc, target_fps, (w, h))

            if record_session:
                session_writer.write(frame)

            # --- JAYWALKING LOGIC ---
            if record_jaywalking:
                 frame_buffer.append(frame)
                 
                 if is_jaywalking:
                     # Extend recording timer if new event happens or start it
                     if not is_recording_jaywalk:
                         print("TRIGEGR: Jaywalking Detected! Starting recording...")
                         is_recording_jaywalk = True
                         
                         ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                         jw_filename = f"recordings/jaywalk_{ts}.mp4"
                         jaywalk_writer = cv2.VideoWriter(jw_filename, fourcc, target_fps, (w, h))
                         
                         # Dump buffer (past 10s)
                         for old_frame in frame_buffer:
                             jaywalk_writer.write(old_frame)
                             
                     # Reset/Extend counter to 3s after LAST detection
                     jaywalk_recording_counter = post_event_frames 
            
            # If we are in recording state
            if is_recording_jaywalk and jaywalk_recording_counter > 0:
                 # We already wrote the buffer. Now we write the live frames.
                 # Avoid writing the same frame twice if it was in buffer? 
                 # Actually, buffer logic dumped Past. Now we are in Live.
                 # Buffer acts as delay? No, buffer stores Copy.
                 # We just write the current frame to the writer.
                 if jaywalk_writer:
                     jaywalk_writer.write(frame)
                 
                 jaywalk_recording_counter -= 1
                 
                 if jaywalk_recording_counter == 0:
                     print("Jaywalking recording ended.")
                     if jaywalk_writer:
                        jaywalk_writer.release()
                        jaywalk_writer = None
                     is_recording_jaywalk = False

            
            # Show live
            cv2.imshow(window_name, frame)
            
            if cv2.waitKey(1) == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred during loop: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if session_writer:
            session_writer.release()
        if jaywalk_writer:
            jaywalk_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage:
    # main_modular(use_enhancement=True, invert_colors=True, use_emojis=True, target_fps=30, record_session=True)
    main_modular(use_enhancement=True, invert_colors=False, use_emojis=False, sample_source=False, url_source=True, target_fps=15, record_session=False, record_jaywalking=True, use_predefined_setup=True)