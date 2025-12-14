# import general dependencies
import cv2
import numpy as np

# import config
from config import VIDEO_PATH, VIDEO_URL, FLIP_CODE, TARGET_FPS, MODEL_PATH

# import setup functions
from setup import define_crop_area, define_traffic_light

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

def main_modular_sample():
    print("1. Initializing sample Source...")
    # Use device index 0 (or 1 if 0 is built-in webcam)
    fps, raw_stream = get_raw_stream_generator(VIDEO_PATH, loop=True)
    print(f"sample FPS: {fps}")

    stream = reduce_framerate(raw_stream, fps, TARGET_FPS)

    print("2. Define Crop Area...")
    crop_rect = define_crop_area(stream)
    
    stream = crop_generator(stream, crop_rect)

    stream = flip_stream_generator(stream, FLIP_CODE)

    print("3. Starting Setup Phase...")
    tl_config = define_traffic_light(stream)

    print("4. Optimizing Stream...")
    stream = enhance_light_generator(stream)

    print("5. Starting Modular Surveillance Phase...")
    
    # --- Modular Pipeline ---
    
    # 1. Initialize State needed for modules
    street_mask = None
    street_pts = np.array(tl_config.get("street", []), np.int32)

    w_crop = crop_rect[2] - crop_rect[0]
    h_crop = crop_rect[3] - crop_rect[1]
    
    street_mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
    if len(street_pts) > 0:
        cv2.fillPoly(street_mask, [street_pts], 255)
    
    # 2. Init AI & Tracker
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()
    tracker = CentroidTracker()

    # 3. Build Pipeline
    # Stream -> Red Light Analysis
    # stream data contains frame, status, status_color
    stream_data = detect_redlight_stream(stream, tl_config)
    
    # Red Light -> People Detection
    # stream data contains frame, status, status_color, people
    stream_data = detect_people_stream(stream_data, interpreter, tracker)
    
    # People -> Drawing
    # Toggle emojis with a boolean (optional, hardcoded for now or parameterize)
    stream = draw_objects_stream(stream_data, tl_config, street_mask, use_emojis=True)
    
    # 4. Display
    show_stream(stream, window_name="Modular Surveillance")


    show_stream(stream, window_name="Modular Surveillance")


def main_modular(use_enhancement=False, invert_colors=False, use_emojis=True):
    print(f"--- Starting Modular Surveillance ---")
    print(f"Features: Enhance={use_enhancement}, Invert={invert_colors}, Emojis={use_emojis}")

    print("1. Initializing Camera Source...")
    # Use device index 0 (or 1 if 0 is built-in webcam)
    try:
        fps, raw_stream = get_camera_stream_generator(0)
    except ValueError:
         print("Camera 0 failed, trying Camera 1...")
         fps, raw_stream = get_camera_stream_generator(1)

    #fps, raw_stream = get_raw_stream_generator(VIDEO_PATH, loop=True)


    print(f"Camera FPS: {fps}")

    stream = reduce_framerate(raw_stream, fps, TARGET_FPS)

    print("2. Define Crop Area...")
    crop_rect = define_crop_area(stream)
    
    stream = crop_generator(stream, crop_rect)

    stream = flip_stream_generator(stream, FLIP_CODE)

    print("3. Starting Setup Phase...")
    tl_config = define_traffic_light(stream)

    if use_enhancement:
        print("4. Optimizing Stream (Enhance Light)...")
        stream = enhance_light_generator(stream)

    print("5. Starting Modular Surveillance Phase...")
    
    # --- Modular Pipeline ---
    
    # 1. Initialize State needed for modules
    street_mask = None
    street_pts = np.array(tl_config.get("street", []), np.int32)

    w_crop = crop_rect[2] - crop_rect[0]
    h_crop = crop_rect[3] - crop_rect[1]
    
    street_mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
    if len(street_pts) > 0:
        cv2.fillPoly(street_mask, [street_pts], 255)
    
    # 2. Init AI & Tracker
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()
    tracker = CentroidTracker()

    # 3. Build Pipeline
    # Stream -> Red Light Analysis
    # stream data contains frame, status, status_color
    stream_data = detect_redlight_stream(stream, tl_config)
    
    # Red Light -> People Detection
    # stream data contains frame, status, status_color, people
    stream_data = detect_people_stream(stream_data, interpreter, tracker)
    
    # Invert Colors (After analysis, before drawing)
    if invert_colors:
        from stream.optimize_stream import invert_stream_data_generator
        stream_data = invert_stream_data_generator(stream_data)

    # People -> Drawing
    # Toggle emojis with a boolean (optional, hardcoded for now or parameterize)
    stream = draw_objects_stream(stream_data, tl_config, street_mask, use_emojis=use_emojis)
    
    # 4. Display
    show_stream(stream, window_name="Modular Surveillance")

if __name__ == "__main__":
    # Example usage:
    # main_modular(use_enhancement=True, invert_colors=True, use_emojis=True)
    main_modular(use_enhancement=True, invert_colors=False, use_emojis=True)