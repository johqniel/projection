import numpy as np
import cv2


from config import PERSON_CLASS_ID, THRESHOLD, NMS_THRESHOLD

# --- Functions return the model outputs for different models ---
"""
Module: model_outputs.py

This module handles the extraction of model outputs from different types of models.
It provides functions to extract output tensors for standard TFLite detection models (e.g., MobileNet SSD).

Key Functions:
- `smart_model_processor(...)`: Extracts output tensors for different models.
    supported models: 
    - standard tflite
    - yolo tflite

Dependencies:
- `config.py`: specific constants (e.g., PERSON_CLASS_ID, THRESHOLD, NMS_THRESHOLD).
- `cv2` (OpenCV): for video capture and frame processing.
- `yt_dlp`: for downloading video streams from URLs.
"""

def smart_model_processor(interpreter, frame):
    """
    Automatically detects the model type based on output details
    and routes to the correct processor.

    Args:
        interpreter: The TFLite interpreter instance.
        frame (np.ndarray): The input frame to be processed.

    Returns:
        dict: A dictionary containing detection results ('boxes', 'classes', 'scores').
    """
    output_details = interpreter.get_output_details()
    
    # Logic: SSD models usually have 4 output tensors. YOLO has 1.
    if len(output_details) == 1:
        # Likely YOLO or similar raw-output model
        return get_yolo_tflite_outputs(interpreter, frame)
    elif len(output_details) >= 3:
        # Likely SSD MobileNet (Boxes, Classes, Scores, [Count])
        return get_standard_tflite_outputs(interpreter, frame)
    else:
        print(f"Warning: Unknown model output structure ({len(output_details)} outputs). Defaulting to SSD.")
        return get_standard_tflite_outputs(interpreter, frame)



def get_standard_tflite_outputs(interpreter, frame):
    """
    Extracts output tensors for standard TFLite detection models (e.g., MobileNet SSD).

    Args:
        interpreter: The TFLite interpreter instance.
        frame (np.ndarray): The input frame (unused here, but kept for function signature consistency).

    Returns:
        dict: A dictionary containing 'boxes' (normalized), 'classes', and 'scores' arrays.
    """
    output_details = interpreter.get_output_details()
    
    # The first three tensors usually represent normalized boxes, classes, and scores.
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
    # The output format is standardized to a dictionary for easy, model-agnostic access.
    return {
        'boxes': boxes,
        'classes': classes,
        'scores': scores
    }


def get_yolo_tflite_outputs(interpreter, frame, conf_threshold=THRESHOLD):
    """
    Handles YOLO style models with Vectorized Processing.
    Fixes 'Slowness' by removing Python loops.
    Fixes 'No Detections' by auto-detecting coordinate format.

    Args:
        interpreter: The TFLite interpreter instance.
        frame (np.ndarray): The input frame (unused here, but kept for function signature consistency).
        conf_threshold (float): Confidence threshold for filtering. Defaults to config.THRESHOLD.

    Returns:
        dict: A dictionary containing 'boxes' (normalized [ymin, xmin, ymax, xmax]),
              'classes', and 'scores' arrays.
    """
    output_details = interpreter.get_output_details()
    input_details = interpreter.get_input_details()
    
    input_h = input_details[0]['shape'][1]
    input_w = input_details[0]['shape'][2]

    output_data = interpreter.get_tensor(output_details[0]['index'])[0] 
    
    # Handle Transpose (YOLOv8 output is often [Classes+4, Anchors])
    if output_data.shape[-1] > output_data.shape[0] and output_data.ndim == 2:
        output_data = output_data.T 

    # --- 1. Efficient Data Extraction ---
    rows = output_data.shape[0]
    cols = output_data.shape[1]
    
    # Heuristic: v5 (5+classes), v8 (4+classes)
    is_v8 = (cols == 4 + 80) # Assuming standard 80 classes
    
    # Extract coordinates
    x_center = output_data[:, 0]
    y_center = output_data[:, 1]
    w = output_data[:, 2]
    h = output_data[:, 3]
    
    # Calculate scores efficiently
    if cols > 5: # YOLOv5
        obj_conf = output_data[:, 4]
        class_scores = output_data[:, 5:]
        # Broadcast multiplication: obj_conf is (N,), class_scores is (N, 80)
        # We only care about the PERSON column for speed
        person_scores = class_scores[:, PERSON_CLASS_ID]
        final_scores = obj_conf * person_scores
    else: # YOLOv8
        class_scores = output_data[:, 4:]
        final_scores = class_scores[:, PERSON_CLASS_ID]

    # --- 2. Vectorized Filtering (Replaces Slow Loop) ---
    # Create a boolean mask for everything above threshold
    mask = final_scores > conf_threshold
    
    # If nothing detected, return early
    if not np.any(mask):
        return {'boxes': [], 'classes': [], 'scores': []}
    
    # Filter arrays using the mask (Only keep good candidates)
    # This reduces data size from ~8400 to just a few (e.g., 5 or 10)
    det_xc = x_center[mask]
    det_yc = y_center[mask]
    det_w = w[mask]
    det_h = h[mask]
    det_scores = final_scores[mask]
    
    # --- 3. Coordinate Auto-Correction (Fixes "No Detections") ---
    # Check if model outputs normalized (0-1) or pixels (0-640)
    # If max coordinate value is small (~1.0), it's normalized.
    if np.max(det_xc) <= 1.05:
        # Scale up to pixels for NMS
        det_xc *= input_w
        det_yc *= input_h
        det_w *= input_w
        det_h *= input_h
        
    # --- 4. Convert to Top-Left for NMS (Vectorized) ---
    # From Center-WH to TopLeft-WH
    det_x = det_xc - (det_w / 2)
    det_y = det_yc - (det_h / 2)
    
    # Stack into (N, 4) array
    boxes_for_nms = np.column_stack((det_x, det_y, det_w, det_h))
    
    # --- 5. Non-Maximum Suppression ---
    # NMSBoxes expects list of boxes, list of scores. 
    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms.tolist(), 
        det_scores.tolist(), 
        conf_threshold, 
        NMS_THRESHOLD
    )
    
    final_boxes = []
    final_classes = []
    final_scores_out = []
    
    if len(indices) > 0:
        # Flatten indices if needed
        indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
        
        for i in indices:
            box = boxes_for_nms[i] # [x, y, w, h] in pixels
            score = det_scores[i]
            
            # Normalize to [0, 1] for the main 'detect_people' function
            xmin = box[0] / input_w
            ymin = box[1] / input_h
            xmax = (box[0] + box[2]) / input_w
            ymax = (box[1] + box[3]) / input_h
            
            final_boxes.append([ymin, xmin, ymax, xmax])
            final_classes.append(PERSON_CLASS_ID)
            final_scores_out.append(score)
            
    return {
        'boxes': np.array(final_boxes),
        'classes': np.array(final_classes),
        'scores': np.array(final_scores_out)
    }

