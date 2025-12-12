import cv2
import numpy as np
import config
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

    # 2. Init AI
    if tflite is None:
        print("Error: TensorFlow Lite not found. Cannot run detection.")
        return

    interpreter = tflite.Interpreter(model_path=config.MODEL_PATH)
    interpreter.allocate_tensors()

    tracker = CentroidTracker()

    # 3. Dummy Config (Required by draw_objects)
    # We provide dummy values because draw_objects unpacks them.
    dummy_config = {
        "red_rect": (0, 0, 1, 1),      # Dummy coordinates
        "shape": [(0,0), (1,0), (1,1)],# Dummy shape points
        "street": []                   # No street defined
    }

    stream = surveillance.reduce_framerate(stream, fps, config.TARGET_FPS)



    # 4. Main Loop
    for frame in stream:
        # Resize to standard size (important for consistent detection)
        # surveillance.detect_people expects the frame, usually resized or original?
        # In run_surveillance, frames come from the stream.
        # If we use main.py logic, frames are cropped/resized before.
        # But here we skip crop, so we should at least resize to FULL_W, FULL_H 
        # to match the canvas size expected by the UI/Config (even if dummy).
        
        frame = cv2.resize(frame, (config.FULL_W, config.FULL_H))

        # Detect People
        # status="GREEN" -> Traffic light is assumed green, so no jaywalking check.
        # street_mask=None -> No street mask needed for Green status.
        
        people = surveillance.detect_people(frame, interpreter, tracker)
        
        # Plot Objects
        # We pass "GREEN" so it plots boxes in Green (unless configured otherwise in draw_objects)
        # and skips jaywalking logic.
        surveillance.draw_objects("GREEN", (0, 255, 0), people, dummy_config, frame, None)

        cv2.imshow("TEST - People Detection Only", frame)

        # Quit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_people_detection_only()
