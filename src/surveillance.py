import cv2
import numpy as np

from model_outputs import smart_model_processor
from border_text import draw_border_layer, draw_banner
from complex_border import draw_border_banner

from tracker_class import CentroidTracker
from config import FULL_W, FULL_H, PERSON_CLASS_ID, THRESHOLD, BANNER_TEXT, FRAME_DELAY_MS, MODEL_PATH, SCROLL_SPEED, CROP_H, TRAFFIC_LIGHT_RED_DIFF_THRESHOLD, TRAFFIC_LIGHT_RED_MIN_VALUE, TARGET_FPS, STREET_VISIBLE

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
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()


    tracker = CentroidTracker()

    scroll_dist = 0

    # Precompute Street Mask
    street_mask = None
    street_pts = np.array(config.get("street", []), np.int32)
    if len(street_pts) > 0:
        # Assuming frame size is constant (FULL_W, FULL_H) after crop/resize logic
        # But we need the actual frame size. We can init it on the first frame.
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

def crop_generator(input_stream, crop_rect=None):
    """
    Consumes a stream, crops it, and yields cropped frames.
    If crop_rect is None, it yields the full frame (resized to standard).
    """
    
    for frame in input_stream:
        # Crop if defined
        if crop_rect:
            x1, y1, x2, y2 = crop_rect
            # Ensure coordinates are valid
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            frame = frame[y1:y2, x1:x2]
        else:
            # Resize to standard if needed (to keep UI consistent) ONLY if not cropping
            frame = cv2.resize(frame, (FULL_W, FULL_H))
        
        yield frame


def reduce_framerate(stream, fps, target_fps=25):
    """
    Yields frames from the stream at a reduced framerate.
    """
    skip_frames = max(1, int(fps / target_fps))
    print(f"Reducing Framerate: Skipping every {skip_frames} frames (Input: {fps:.2f} -> Target: {target_fps})")
    
    count = 0
    for frame in stream:
        count += 1
        if count % skip_frames == 0:
            yield frame


def flip_stream_generator(stream, flip_code):
    """
    Yields flipped frames from the stream.
    """
    for frame in stream:
        yield cv2.flip(frame, flip_code)



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

def draw_objects(status, status_color, objects, config, frame, street_mask):
    # Config Unpack
    rx1, ry1, rx2, ry2 = config["red_rect"]
    shape_pts = np.array(config["shape"], np.int32)
    shape_pts = shape_pts.reshape((-1, 1, 2))
    
    # Street Polygon
    street_pts = np.array(config.get("street", []), np.int32)
    if len(street_pts) > 0:
        street_pts = street_pts.reshape((-1, 1, 2))
        # Draw street area (optional, maybe semi-transparent or just outline)
        if STREET_VISIBLE:
            cv2.polylines(frame, [street_pts], isClosed=True, color=(255, 255, 0), thickness=2)


    # Persons
    h, w = frame.shape[:2]
    
    for (objectID, box) in objects.items():
        (xmin, ymin, xmax, ymax) = box
        
        # Default color
        box_color = (0, 255, 0) # Green
        
        # Check intersection with street if traffic light is RED
        if status == "RED" and street_mask is not None:
            is_jaywalking = False
            
            # Check if touching border
            touches_border = (xmin <= 0 or ymin <= 0 or xmax >= w or ymax >= h)
            
            if touches_border:
                # Case 2: Touching border -> Check if ANY part overlaps
                # ROI of the box in the mask
                # Ensure coords are within bounds
                b_y1, b_y2 = max(0, ymin), min(h, ymax)
                b_x1, b_x2 = max(0, xmin), min(w, xmax)
                
                if b_y2 > b_y1 and b_x2 > b_x1:
                    mask_roi = street_mask[b_y1:b_y2, b_x1:b_x2]
                    non_zero = cv2.countNonZero(mask_roi)
                    if non_zero > 0:
                        is_jaywalking = True
            else:
                # Case 1: Not touching border -> Check if LOWER SIDE touches street
                # We check a thin strip at the bottom of the box
                b_y1 = max(0, ymax - 5) # Check last 5 pixels? Or just the line? User said "lower side".
                b_y2 = min(h, ymax)
                b_x1, b_x2 = max(0, xmin), min(w, xmax)
                
                if b_y2 > b_y1 and b_x2 > b_x1:
                    mask_roi = street_mask[b_y1:b_y2, b_x1:b_x2]
                    non_zero = cv2.countNonZero(mask_roi)
                    if non_zero > 0:
                        is_jaywalking = True

            if is_jaywalking:
                box_color = (0, 0, 255) # Red (Jaywalking!)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
        cv2.putText(frame, f"ID {objectID}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Traffic Light Shape
    cv2.polylines(frame, [shape_pts], isClosed=True, color=status_color, thickness=3)

    return(frame)



