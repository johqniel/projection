import cv2
import argparse
import os
import sys
import numpy as np

try:
    from moviepy.editor import VideoFileClip
except ImportError:
    print("Error: 'moviepy' is required to preserve audio.")
    sys.exit(1)

# TFLite Import
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Error: tensorflow or tflite_runtime required.")
        sys.exit(1)

# Import existing helpers
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from analysis.model_outputs import get_yolo_tflite_outputs
from analysis.tracker_class import CentroidTracker
from config import PERSON_CLASS_ID

# Direct Path to Model
MODEL_FILE = "yolov5s_f16.tflite"

def blur_regions_pixels(frame, pixels_boxes, w_img, h_img):
    """
    frame: RGB numpy array
    pixels_boxes: list of [x1, y1, x2, y2]
    """
    out_frame = frame.copy()
    
    for box in pixels_boxes:
        # Tracker returns floats/ints usually
        xmin, ymin, xmax, ymax = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        
        # --- IMPROVED HEURISTIC: "Top Center" ---
        box_w = xmax - xmin
        box_h = ymax - ymin
        
        # 1. Height: Top 15%
        face_h = int(box_h * 0.15)
        face_y2 = ymin + face_h
        
        # 2. Width: Center the face box (Head is narrower than shoulders)
        face_w = int(box_w * 0.4)
        center_x = (xmin + xmax) // 2
        
        fx1 = center_x - (face_w // 2)
        fx2 = center_x + (face_w // 2)
        
        # Clamp
        fx1, fy1 = max(0, fx1), max(0, ymin)
        fx2, fy2 = min(w_img, fx2), min(h_img, face_y2)
        
        fw = fx2 - fx1
        fh = fy2 - fy1
        
        if fw > 0 and fh > 0:
            roi = out_frame[fy1:fy2, fx1:fx2]
            ksize = (fw // 3) | 1
            if ksize < 3: ksize = 3
            roi = cv2.GaussianBlur(roi, (ksize, ksize), 0)
            out_frame[fy1:fy2, fx1:fx2] = roi
            
    return out_frame

def process_video_tflite(input_path, output_path=None, threshold=0.5):
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found.")
        return
        
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file '{MODEL_FILE}' not found.")
        return

    if output_path is None:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_blurred_tracker.mp4"
    
    print(f"Processing: {input_path}")
    print(f"Output: {output_path}")
    print(f"Model: {MODEL_FILE}")
    print(f"Threshold: {threshold}")
    print("Using CentroidTracker for stability.")

    # Init Interpreter
    interpreter = tflite.Interpreter(model_path=MODEL_FILE, num_threads=4)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape'] # [1, 640, 640, 3] usually
    target_h, target_w = input_shape[1], input_shape[2]
    
    # Init Tracker
    # We can tune params if needed
    tracker = CentroidTracker(maxDisappeared=20, smoothing_factor=0.5)
    
    clip = VideoFileClip(input_path)
    
    def process_frame_wrapper(frame):
        # Frame is RGB (MoviePy)
        h_orig, w_orig = frame.shape[:2]
        
        # Resize for Model
        resized = cv2.resize(frame, (target_w, target_h))
        input_data = np.expand_dims(resized / 255.0, axis=0).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # Raw Results
        results = get_yolo_tflite_outputs(interpreter, frame, conf_threshold=threshold)
        
        # Filter Person & Convert to Pixels
        # results['boxes'] is [ymin, xmin, ymax, xmax] (Normalized)
        
        raw_boxes = results['boxes']
        raw_classes = results['classes']
        
        pixel_rects = [] # [xmin, ymin, xmax, ymax]
        
        for i in range(len(raw_boxes)):
            if int(raw_classes[i]) == PERSON_CLASS_ID:
                box = raw_boxes[i]
                # Convert normalized [ymin, xmin, ymax, xmax] -> pixels [xmin, ymin, xmax, ymax]
                # Note: tracker expects (startX, startY, endX, endY)
                
                ymin = int(box[0] * h_orig)
                xmin = int(box[1] * w_orig)
                ymax = int(box[2] * h_orig)
                xmax = int(box[3] * w_orig)
                
                pixel_rects.append((xmin, ymin, xmax, ymax))
        
        # Update Tracker
        # objects is dict: ID -> [xmin, ymin, xmax, ymax]
        objects = tracker.update(pixel_rects)
        
        # We only care about values (boxes) for blurring
        tracked_boxes = list(objects.values())
        
        # Apply Blur
        return blur_regions_pixels(frame, tracked_boxes, w_orig, h_orig)

    new_clip = clip.fl_image(process_frame_wrapper)
    
    print("Writing video file with audio...")
    new_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Blur pseudo-faces using YOLOv5 TFLite + Tracker.")
    parser.add_argument("input_video", help="Path to input video file")
    parser.add_argument("--output", help="Path to output video file", default=None)
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection confidence threshold")
    
    args = parser.parse_args()
    
    process_video_tflite(args.input_video, args.output, threshold=args.threshold)

if __name__ == "__main__":
    main()
