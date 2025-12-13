import cv2
import numpy as np
import config
import time
from get_stream import get_raw_stream_generator
import surveillance
from tracker_class import CentroidTracker

# Try to import TFLite just like in surveillance.py
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None

def run_people_detection_only():
    """
    Runs only people detection and plotting, skipping all setup.
    """
    print("--- Starting People Detection Test (No Setup) ---")

    # 1. Get Stream
    print(f"Loading video from: {config.VIDEO_PATH}")
    # Note: get_raw_stream_generator returns (fps, generator)
    fps, stream = get_raw_stream_generator(config.VIDEO_PATH, loop=True)
    print(f"Input FPS: {fps}")

    print(f"Reducing analysis FPS to {config.TARGET_FPS}")
    stream = surveillance.reduce_framerate(stream, fps, config.TARGET_FPS, True)

    print(f"Loading model from: {config.MODEL_PATH}")

    # 2. Init AI
    if tflite is None:
        print("Error: TensorFlow Lite not found. Cannot run detection.")
        return

    interpreter = tflite.Interpreter(model_path=config.MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()

    tracker = CentroidTracker()

    # 3. Dummy Config (Required by draw_objects)
    # We provide dummy values because draw_objects unpacks them.
    dummy_config = {
        "red_rect": (0, 0, 1, 1),      # Dummy coordinates
        "shape": [(0,0), (1,0), (1,1)],# Dummy shape points
        "street": []                   # No street defined
    }




    # Target frame interval in seconds
    target_interval = 1.0 / config.TARGET_FPS

    # 4. Main Loop
    for frame in stream:
        loop_start = time.time()
        
        frame = cv2.resize(frame, (config.FULL_W, config.FULL_H))

        # Detect People
        t0 = time.time()
        people = surveillance.detect_people(frame, interpreter, tracker)
        t1 = time.time()
        
        # Plot Objects
        surveillance.draw_objects("GREEN", (0, 255, 0), people, dummy_config, frame, None)
        t2 = time.time()

        print(f"Detect: {(t1-t0)*1000:.1f}ms | Draw: {(t2-t1)*1000:.1f}ms")

        cv2.imshow("TEST - People Detection Only", frame)

        # Calculate how long processing took
        processing_time = time.time() - loop_start
        
        # Calculate remaining time to wait to match target FPS
        wait_time = target_interval - processing_time
        delay_ms = max(1, int(wait_time * 1000))
        
        # Quit on 'q'
        if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def reduce_framerate_only():
    print("1. Initializing Video Source...")
    fps, stream = get_raw_stream_generator(config.VIDEO_PATH, loop=True)
    print(f"Input FPS: {fps}")
    print(f"Reducing analysis FPS to {config.TARGET_FPS}")
    # Enable real-time playback simulation since we are reading from file
    stream = surveillance.reduce_framerate(stream, fps, config.TARGET_FPS, real_time_playback=False)

    # 4. Main Loop
    for frame in stream:
        cv2.imshow("TEST - Reduce Framerate Only", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

def playback_raw_stream():
    print("1. Initializing Video Source...")
    fps, stream = get_raw_stream_generator(config.VIDEO_PATH, loop=True)
    print(f"Input FPS: {fps}")
    
    # 4. Main Loop
    for frame in stream:
        cv2.imshow("TEST - Raw Stream", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_people_detection_only()
