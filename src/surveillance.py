import cv2
import numpy as np

from model_outputs import smart_model_processor
from border_text import draw_border_layer

from tracker_class import CentroidTracker
from config import FULL_W, FULL_H, PERSON_CLASS_ID, THRESHOLD, BANNER_TEXT, FRAME_DELAY_MS, MODEL_PATH, SCROLL_SPEED, CROP_H, TRAFFIC_LIGHT_RED_DIFF_THRESHOLD, TRAFFIC_LIGHT_RED_MIN_VALUE

# --- CORE FUNCTIONS ---

# --- UNIVERSAL IMPORT BLOCK ---
try:
    # TFLITE RUNTIME (For Raspberry Pi)
    import tflite_runtime.interpreter as tflite
    print("Using TFLite Runtime")
except ImportError:
    # FULL TENSORFLOW (For Mac / PC)
    # This works with tensorflow-macos too!
    import tensorflow.lite as tflite
    print("Using Full TensorFlow")

def run_surveillance(stream,config):
    """
    Consumes the stream (or the looping stream) using the config.
    """
    
    # Init AI
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()


    tracker = CentroidTracker()

    scroll_dist = 0

    for frame in stream: 

        key = cv2.waitKey(FRAME_DELAY_MS)


        status, status_color = detect_red_light(frame,config)

        people = detect_people(frame, interpreter, tracker)
        
        # Pass traffic status to draw_objects
        frame = draw_objects(status, status_color, people, config, frame)



        draw_banner(frame, scroll_dist, status, status_color)

        #draw_border_layer(frame,scroll_dist)

        cv2.imshow("SURVEILLANCE", frame)

        scroll_dist += SCROLL_SPEED

        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()

def crop_generator(input_stream):
    """
    Consumes a stream, crops it, and yields cropped frames.
    """
    # Define Crop Coordinates (Zone 1)
    # Logic: Right half of the image
    A_H, B_H = 0, CROP_H
    A_V, B_V = int(FULL_W - 1.5*CROP_H), FULL_W

    for frame in input_stream:
        # Resize to standard if needed
        frame = cv2.resize(frame, (FULL_W, FULL_H))
        
        # Crop
        cropped = frame[A_H:B_H, A_V:B_V]
        
        yield cropped



# -- layers of run surveillance

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
            status_color = (0, 0, 255)
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

def draw_objects(status, status_color, objects, config, frame):
    # Config Unpack
    rx1, ry1, rx2, ry2 = config["red_rect"]
    shape_pts = np.array(config["shape"], np.int32)
    shape_pts = shape_pts.reshape((-1, 1, 2))
    
    # Street Polygon
    street_pts = np.array(config.get("street", []), np.int32)
    if len(street_pts) > 0:
        street_pts = street_pts.reshape((-1, 1, 2))
        # Draw street area (optional, maybe semi-transparent or just outline)
        cv2.polylines(frame, [street_pts], isClosed=True, color=(255, 255, 0), thickness=2)


    # Persons
    for (objectID, box) in objects.items():
        (xmin, ymin, xmax, ymax) = box
        
        # Default color
        box_color = (0, 255, 0) # Green
        
        # Check intersection with street if traffic light is RED
        if status == "RED" and len(street_pts) > 0:
            # Check if the bottom center of the box is inside the street polygon
            # Using bottom center is better for "feet" position
            foot_x = int((xmin + xmax) / 2)
            foot_y = int(ymax)
            
            # pointPolygonTest returns positive if inside, negative if outside, 0 if on edge
            dist = cv2.pointPolygonTest(street_pts, (foot_x, foot_y), False)
            
            if dist >= 0:
                box_color = (0, 0, 255) # Red (Jaywalking!)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
        cv2.putText(frame, f"ID {objectID}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Traffic Light Shape
    cv2.polylines(frame, [shape_pts], isClosed=True, color=status_color, thickness=3)

    return(frame)



def draw_banner(frame, scroll_dist, traffic_status, status_color):

    h,w,_ = frame.shape

    # Scrolling Banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
    info_text = f"{BANNER_TEXT} | SIGNAL: {traffic_status}"
    # Simple scrolling logic
    text_x = 10 - (scroll_dist % (w + 400)) + w
    cv2.putText(frame, info_text, (text_x, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    # Second copy for seamless scroll
    if text_x < 0:
         cv2.putText(frame, info_text, (text_x + w + 400, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

