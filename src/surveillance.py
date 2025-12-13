# General
import cv2
import numpy as np
import time

# Detection
from analysis import detect_red_light, detect_people

# Tracking
from analysis.tracker_class import CentroidTracker

# Drawing
from plotting import draw_objects, draw_border_layer, draw_banner, draw_border_banner

# Config
from config import MODEL_PATH, FRAME_DELAY_MS, SCROLL_SPEED


# --- CORE FUNCTIONS ---

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

def run_surveillance(stream,config):
    """
    Consumes the stream (or the looping stream) using the config.
    """
    
    # Init AI
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()


    tracker = CentroidTracker()

    scroll_dist = 0

    # Precompute Street Mask
    street_mask = None
    street_pts = np.array(config.get("street", []), np.int32)
    if len(street_pts) > 0:

        pass

    first_frame = True

    for frame in stream: 
        if first_frame:
            h, w = frame.shape[:2]
            street_mask = np.zeros((h, w), dtype=np.uint8)
            if len(street_pts) > 0:
                cv2.fillPoly(street_mask, [street_pts], 255)
            first_frame = False

        key = cv2.waitKey(FRAME_DELAY_MS)


        status, status_color = detect_red_light(frame,config)

        people = detect_people(frame, interpreter, tracker)
        
        # Pass traffic status to draw_objects
        frame = draw_objects(status, status_color, people, config, frame, street_mask)



        draw_border_banner(frame,w,h,scroll_dist)

        #draw_banner(frame, scroll_dist, status, status_color)

        #draw_border_layer(frame,scroll_dist)

        cv2.imshow("SURVEILLANCE", frame)

        scroll_dist += SCROLL_SPEED

        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()



