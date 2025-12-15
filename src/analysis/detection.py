# General
import cv2
import numpy as np

# Model handling
from .model_outputs import smart_model_processor

# Object tracking
from .tracker_class import CentroidTracker

# config
from config import TRAFFIC_LIGHT_RED_DIFF_THRESHOLD, TRAFFIC_LIGHT_RED_MIN_VALUE, PERSON_CLASS_ID, THRESHOLD



def detect_red_light(frame, config):
    """
    Decides whether traffic light is red or green
    """ 
    # Config Unpack
    rx1, ry1, rx2, ry2 = config["red_rect"]
    shape_pts = np.array(config["shape"], np.int32)
    shape_pts = shape_pts.reshape((-1, 1, 2))

        
    # --- 1. TRAFFIC LIGHT CHECK ---
    # Using the coords from Setup (which are relative to the cropped frame)
    roi = frame[ry1:ry2, rx1:rx2]
        
    traffic_status = "UNKNOWN"
    status_color = (100, 100, 100)

    if roi.size > 0:
        b, g, r = cv2.split(roi)
        mean_r = np.mean(r)
        mean_g = np.mean(g)
            
        # Simple Logic: If Red is dominant
        if mean_r > mean_g + TRAFFIC_LIGHT_RED_DIFF_THRESHOLD and mean_r > TRAFFIC_LIGHT_RED_MIN_VALUE:
            traffic_status = "RED"
            status_color = (203, 192, 255)
        else:
            traffic_status = "GREEN"
            status_color = (0, 255, 0)
    
    return traffic_status, status_color

def detect_people(frame, interpreter, tracker, model_output_processor = smart_model_processor):
    """
    Detects people in the frame using TFLite model, using a pluggable 
    function for output processing.
    
    Args:
        frame (np.array): The input image frame (BGR format).
        interpreter: The TFLite interpreter instance.
        tracker: The object tracker instance.
        model_output_processor (function): A function (interpreter, frame) -> dict 
                                           to handle model-specific output extraction.
                                           
    Returns:
        dict: A dictionary of detected and tracked objects.
    """
    if frame is None: return {}

    # Init AI (Input Preprocessing is model-agnostic for this example)
    input_details = interpreter.get_input_details()

    h, w = frame.shape[:2]

    # --- 1. PRE-PROCESSING (Standard) ---
    max_dim = max(h, w)
    square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    square_img[0:h, 0:w] = frame
    
    # Resize to model input size (300x300, or whatever the input_details specify)
    # The mock interpreter is 300x300.
    input_h, input_w = input_details[0]['shape'][1], input_details[0]['shape'][2]
    input_img = cv2.resize(square_img, (input_w, input_h), interpolation=cv2.INTER_AREA)
    
    # Convert BGR to RGB and normalize (CRITICAL FIX)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    
    # Convert to float32 and normalize (if the model expects it, often 0-1 or -1 to 1)
    # Assuming 0-255 uint8 input as is common for MobileNet/SSD models
    input_data = np.expand_dims(input_img, axis=0)
    
    # Check expected dtype (e.g., if model expects float32)
    if input_details[0]['dtype'] == np.float32:
         input_data = input_data.astype(np.float32) / 255.0 # Simple normalization example

    # --- 2. AI INFERENCE ---
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
        
    # --- 3. MODEL-SPECIFIC OUTPUT PROCESSING ---
    # This is the pluggable part!
    try:
        outputs = model_output_processor(interpreter, frame)
        boxes = outputs['boxes']
        classes = outputs['classes']
        scores = outputs['scores']
    except Exception as e:
        print(f"Error processing model output: {e}")
        return {}
    
    # --- 4. POST-PROCESSING (Model-Agnostic Rects and Tracking) ---
    rects = []
    # Iterate through detections using the length of the scores array
    for i in range(len(scores)):
        # Check against threshold and ensure the detected class is 'person'
        if scores[i] > THRESHOLD and int(classes[i]) == PERSON_CLASS_ID:
            ymin, xmin, ymax, xmax = boxes[i]
            
            # Map normalized coordinates back from square space to original space
            # Note: TFLite boxes are typically normalized [0, 1]
            rects.append((
                int(xmin * max_dim), 
                int(ymin * max_dim), 
                int(xmax * max_dim), 
                int(ymax * max_dim)
            ))
    
    # Update tracker with the list of bounding box rectangles
    objects = tracker.update(rects)
    return objects
