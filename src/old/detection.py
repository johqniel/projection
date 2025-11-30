import cv2
import numpy as np
import os
import datetime 
from collections import OrderedDict
import yt_dlp 

# --- UNIVERSAL IMPORT BLOCK ---
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# --- CONFIGURATION ---
MODEL_PATH = "detect.tflite" 
THRESHOLD = 0.3 
YOUTUBE_URL = "https://www.youtube.com/watch?v=6dp-bvQ7RWo" 
VIDEO_PATH = "/Users/nielo/Desktop/projection/samples/A.MOV"
VIDEO_PATH = ""
USB_DEVICE_INDEX = 0 

PERSON_CLASS_ID = 0
TRAFFIC_LIGHT_CLASS_ID = 9 

# --- RECORDING CONFIGURATION ---
RECORD_VIDEO = True     
OUTPUT_FOLDER = "recordings"

# --- VISUAL CONFIGURATION ---
BANNER_TEXT = "ZONE 1 ACTIVE: ASPECT RATIO CORRECTED *** SCANNING... "
SCROLL_SPEED = 6        
TEXT_COLOR = (0, 255, 255) 
FONT_SCALE = 0.9        
THICKNESS = 2
MARGIN = 20             
BG_ALPHA = 0.6          
BG_PADDING_X = 10       
BG_PADDING_Y = 10 
FULL_W, FULL_H = 1280, 720
CROP_W, CROP_H = FULL_W // 2, FULL_H // 2

# --- GLOBAL STATE FOR MOUSE SELECTION ---
roi_start = None       # Start point (x,y)
roi_end = None         # Current/End point (x,y)
roi_selected = False   # Is the box finalized?
manual_roi_coords = None # Final box: (xmin, ymin, xmax, ymax)

# --- GLOBAL TRAFFIC LIGHT STATE ---
CURRENT_AMPEL_STATE = "UNKNOWN" # Global variable updated by analysis

# --- HELPER CLASSES (Original Logic) ---
class CentroidTracker:
    def __init__(self, maxDisappeared=20):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.bbox_store = OrderedDict()
        self.maxDisappeared = maxDisappeared
    def register(self, centroid, rect):
        self.objects[self.nextObjectID] = centroid
        self.bbox_store[self.nextObjectID] = rect
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.bbox_store[objectID]
    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            return self.bbox_store
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)): self.register(inputCentroids[i], rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = []
            for i in range(len(objectCentroids)):
                row = []
                for j in range(len(inputCentroids)):
                    dist = np.linalg.norm(np.array(objectCentroids[i]) - np.array(inputCentroids[j]))
                    row.append(dist)
                D.append(row)
            D = np.array(D)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.bbox_store[objectID] = rects[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            else:
                for col in unusedCols: self.register(inputCentroids[col], rects[col])
        return self.bbox_store

def get_stream_url(yt_url):
    print(f"⏳ Extrahiere Stream-URL von: {yt_url} ...")
    ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(yt_url, download=False)
        return info['url']

# --- MOUSE CALLBACK & ROI LOGIC  ---
# ---------------------------------------------------------

def mouse_callback(event, x, y, flags, param):

    global roi_start, roi_end, roi_selected, manual_roi_coords
    
    # 1. User presses mouse button down
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
        roi_end = (x, y)
        roi_selected = False # Reset status while dragging

    # 2. User drags the mouse
    elif event == cv2.EVENT_MOUSEMOVE:
        if roi_start is not None and not roi_selected:
            roi_end = (x, y)

    # 3. User releases mouse button
    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)
        roi_selected = True
        
        # Calculate properly ordered coordinates (xmin, ymin, xmax, ymax)
        x1, y1 = roi_start
        x2, y2 = roi_end
        # We use min/max so it works even if you drag bottom-right to top-left
        manual_roi_coords = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        print(f"ROI Set: {manual_roi_coords}")

# ---------------------------------------------------------
# --- MODULE 1: IMAGE PROCESSING (Generators) ---
# ---------------------------------------------------------

def resize_stream(input_stream):
    """crops and resizes the stream"""
    for raw_frame in input_stream:
        # resize
        resized_frame = cv2.resize(raw_frame, (FULL_W, FULL_H))
        yield resized_frame

def crop_stream(input_stream,A_H = 0,B_H = CROP_H, A_V = int(FULL_W -1.5*CROP_H), B_V = FULL_W):
    """crops stream"""
    for raw_frame in input_stream:
        # crop frame: 
        cropped_frame = raw_frame[A_H:B_H, A_V: B_V]
        yield cropped_frame

def generate_frames_from_source(cap, loop_video=False):
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            if loop_video:
                print("Video ended. Restarting...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break 
        yield raw_frame

# ---------------------------------------------------------
# --- MODULE 2: AI INFERENCE ---
# ---------------------------------------------------------

def init_ai_engine():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Error: {MODEL_PATH} not found.")
    
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def perform_ai_analysis(frame, interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    h, w, _ = frame.shape
    max_dim = max(h, w)

    # Letterboxing
    square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    square_img[0:h, 0:w] = frame
    input_img = cv2.resize(square_img, (300, 300))
    input_data = np.expand_dims(input_img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    person_rects = [] 

    for i in range(len(scores)):
        if scores[i] > THRESHOLD:
            class_id = int(classes[i])
            ymin = int(boxes[i][0] * max_dim)
            xmin = int(boxes[i][1] * max_dim)
            ymax = int(boxes[i][2] * max_dim)
            xmax = int(boxes[i][3] * max_dim)
            
            if ymin > h: continue 
            ymax = min(ymax, h)
            xmax = min(xmax, w) 

            if class_id == PERSON_CLASS_ID:
                person_rects.append((xmin, ymin, xmax, ymax))
                
    return person_rects

# ---------------------------------------------------------
# --- MODULE 3: TRAFFIC LIGHT ANALYSIS ---
# ---------------------------------------------------------

def analyze_manual_roi_color(frame):
    global manual_roi_coords
    
    # If user hasn't drawn a box yet
    if manual_roi_coords is None:
        return (0, 165, 255), "DRAW BOX" 

    xmin, ymin, xmax, ymax = manual_roi_coords
    
    # Safety: Ensure box stays inside the image boundaries
    h, w, _ = frame.shape
    xmin, xmax = max(0, xmin), min(w, xmax)
    ymin, ymax = max(0, ymin), min(h, ymax)
    
    roi = frame[ymin:ymax, xmin:xmax]
    
    if roi.size == 0: return (0, 165, 255), "EMPTY"

    # Split channels (Blue, Green, Red)
    b, g, r = cv2.split(roi)
    
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    
    # Simple Dominance Logic: If Green is brighter than Red (+ buffer)
    if mean_g > mean_r + 5: 
        return (0, 255, 0), "GRUEN"
    else:
        return (0, 0, 255), "ROT"

# ---------------------------------------------------------
# --- MODULE 4: VISUALIZATION ---
# ---------------------------------------------------------

def calculate_letter_geometry(letter, center, angle, scale, thickness):
    (fw, fh), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    box_w = fw + (BG_PADDING_X * 2)
    box_h = fh + (BG_PADDING_Y * 2)
    corners = np.array([[-box_w/2, -box_h/2], [ box_w/2, -box_h/2], [ box_w/2,  box_h/2], [-box_w/2,  box_h/2]])
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((c, -s), (s, c)))
    return (np.dot(corners, rotation_matrix) + center).astype(np.int32), (fw, fh)

def draw_text_content(img, letter, center, angle, scale, color, thickness, size):
    fw, fh = size
    diag = int(np.sqrt(fw**2 + fh**2)) + 10
    txt_img = np.zeros((diag, diag, 3), dtype=np.uint8)
    cx, cy = diag // 2, diag // 2
    cv2.putText(txt_img, letter, (cx - fw // 2, cy + fh // 2), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(txt_img, M, (diag, diag))
    start_x, start_y = int(center[0] - cx), int(center[1] - cy)
    h, w, _ = img.shape
    if start_x < 0 or start_y < 0 or start_x+diag > w or start_y+diag > h: return 
    
    roi = img[start_y:start_y+diag, start_x:start_x+diag]
    mask = cv2.threshold(cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
    img[start_y:start_y+diag, start_x:start_x+diag] = cv2.add(cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask)), cv2.bitwise_and(rotated, rotated, mask=mask))

def get_perimeter_pos(dist, w, h, margin):
    eff_w, eff_h = w - 2*margin, h - 2*margin
    d = dist % (2 * eff_w + 2 * eff_h)
    if d < eff_h: return ((w - margin), h - margin - d), 90
    d -= eff_h
    if d < eff_w: return ((w - margin) - d, margin), 180
    d -= eff_w
    if d < eff_h: return (margin, margin + d), 270
    d -= eff_h
    return (w - margin - d, h - margin), 0

def add_labels_and_border(frame, person_rects, tracker, scroll_dist):
    h, w, _ = frame.shape
    
    # 1. Update Tracker
    objects = tracker.update(person_rects)
    for (objectID, box) in objects.items():
        (xmin, ymin, xmax, ymax) = box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {objectID}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 3. Draw Scrolling Text
    render_queue = []
    for i, char in enumerate(BANNER_TEXT):
        (cx, cy), angle = get_perimeter_pos(scroll_dist + (i * 20), w, h, MARGIN)
        box_pts, text_size = calculate_letter_geometry(char, (cx, cy), angle, FONT_SCALE, THICKNESS)
        render_queue.append({"char": char, "center": (cx,cy), "angle": angle, "box": box_pts, "size": text_size})

    overlay = frame.copy()
    for item in render_queue: cv2.fillPoly(overlay, [item["box"]], (0, 0, 0))
    cv2.addWeighted(overlay, BG_ALPHA, frame, 1 - BG_ALPHA, 0, frame)
    for item in render_queue: draw_text_content(frame, item["char"], item["center"], item["angle"], FONT_SCALE, TEXT_COLOR, THICKNESS, item["size"])
    return frame

# ---------------------------------------------------------
# --- MAIN LOOP (Source Agnostic) ---
# ---------------------------------------------------------

def run_camera_loop(frame_generator_stream):
    """
    This loop is now 'blind'. It doesn't know where the frame comes from 
    or that it has been cropped. It just takes the next image, runs AI, 
    draws labels, and shows it.
    """
    global roi_start, roi_end, roi_selected, manual_roi_coords, CURRENT_AMPEL_STATE
    
    interpreter = init_ai_engine()
    tracker = CentroidTracker(maxDisappeared=20)
    
    out = None
    if RECORD_VIDEO:
        if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{OUTPUT_FOLDER}/rec_{timestamp}.avi"
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (CROP_W, CROP_H))

    global_distance = 0
    print("System Active. Press 'q' to exit. Press 'r' to reset ROI.")

    # 1. SETUP MOUSE CALLBACK
    cv2.namedWindow("Quarter Detector")
    cv2.setMouseCallback("Quarter Detector", mouse_callback)

    # iterate over the generator
    for frame in frame_generator_stream:
        
        # 1. AI Analysis (People)
        persons = perform_ai_analysis(frame, interpreter)
        
        # 2. Visuals (People & Border)
        global_distance += SCROLL_SPEED
        frame = add_labels_and_border(frame, persons, tracker, global_distance)

        # -----------------------------------------------------------
        # 3. TRAFFIC LIGHT DRAWING & ANALYSIS
        # -----------------------------------------------------------
        
        # A. PREVIEW: Draw the "Dragging" Box (Yellow/Cyan)
        # We check if start exists, end exists, but NOT finalized yet
        if roi_start and roi_end and not roi_selected:
             cv2.rectangle(frame, roi_start, roi_end, (255, 255, 0), 1)

        # B. ACTIVE: Draw the "Final" Box + Analyze Color
        if manual_roi_coords is not None:
            # 1. Analyze color inside the box
            color_bgr, label_text = analyze_manual_roi_color(frame)
            
            # 2. Update Global State
            CURRENT_AMPEL_STATE = label_text 

            # 3. Draw the active box
            (x1, y1, x2, y2) = manual_roi_coords
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            
            # 4. Draw the label "Ampel"
            label = f"Ampel: {label_text}"
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        # -----------------------------------------------------------

        # 4. Output
        if out is not None: out.write(frame)
        cv2.imshow("Quarter Detector", frame)

        k = cv2.waitKey(1)
        
        if k == ord('q'): break

        # *** RESET TRAFFIC LIGHTS FUNCTION ***
        if k == ord('r'): 
            manual_roi_coords = None
            roi_start = None
            roi_end = None
            roi_selected = False
            CURRENT_AMPEL_STATE = "UNKNOWN"
            print("ROI Reset.")

    if out is not None: out.release() 
    cv2.destroyAllWindows()

# ---------------------------------------------------------
# --- CLEAN MAINS ---
# ---------------------------------------------------------

def main_sample():
    print(f"Loading Sample: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # We create the stream HERE
    stream = generate_frames_from_source(cap, loop_video=True)
    stream = resize_stream(stream)
    
    run_camera_loop(stream)
    cap.release()

def main_stream():
    url = get_stream_url(YOUTUBE_URL)
    print(f"Loading Stream: {url}")
    cap = cv2.VideoCapture(url)
    
    # loop_video=False for live streams
    stream = generate_frames_from_source(cap, loop_video=False)

    stream = resize_stream(stream)
    cropped_stream = crop_stream(stream)
    
    run_camera_loop(cropped_stream)
    cap.release()

def main_USB_Video():
    print(f"Loading USB Camera: {USB_DEVICE_INDEX}")
    cap = cv2.VideoCapture(USB_DEVICE_INDEX)
    
    stream = generate_frames_from_source(cap, loop_video=False)
    
    run_camera_loop(stream)
    cap.release()

if __name__ == "__main__":
    main_stream()
    #main_sample()
    # main_USB_Video()