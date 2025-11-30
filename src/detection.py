import cv2
import numpy as np
import os
import time
from collections import OrderedDict
import yt_dlp 

# --- IMPORT TFLITE ---
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# --- CONFIGURATION ---
MODEL_PATH = "detect.tflite" 
THRESHOLD = 0.3 
VIDEO_PATH = "/Users/nielo/Desktop/projection/samples/MOV001.mp4"
VIDEO_URL = "https://www.youtube.com/watch?v=6dp-bvQ7RWo" # Falls URL gewünscht

PERSON_CLASS_ID = 0

# Visual Config
FULL_W, FULL_H = 1280, 720
FRAME_DELAY_MS = 1
CROP_W, CROP_H = FULL_W // 2, FULL_H // 2
BANNER_TEXT = "ZONE 1 ACTIVE: SURVEILLANCE RUNNING"
SCROLL_SPEED = 6

# Config for border banner
BG_ALPHA = 0.6          # Transparency (0.6 = 60% opaque)
BG_BAR_WIDTH = 80
SHIFT = 40
TEXT_COLOR = (0, 255, 255) 
FONT_SCALE = 1.2
THICKNESS = 2
MARGIN = 40  
letter_spacing = 25

# --- HELPER: TRACKER ---
class CentroidTracker:
    def __init__(self, maxDisappeared=20):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
    
    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
    
    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            return self.objects
        
        inputCentroids = np.zeros((len(rects), 4), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            #inputCentroids[i] = (cX, cY)
            inputCentroids[i] = (startX,startY,endX,endY)
            
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)): self.register(inputCentroids[i])
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
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            for col in set(range(0, D.shape[1])).difference(usedCols):
                self.register(inputCentroids[col])
        return self.objects

# --- HELPER: SETUP SESSION (MOUSE LOGIC) ---
class SetupSession:
    def __init__(self):
        self.step = 0 # 0=Zoom, 1=Red Light, 2=Shape
        self.pt_start = None # Startpunkt (Mausklick)
        self.pt_end = None   # Endpunkt (Mausbewegung)
        self.current_pos = (0,0) # Aktuelle Mausposition
        self.dragging = False
        
        # Diese Listen speichern die Ergebnisse in SCREEN Koordinaten (0-640)
        # Wir rechnen sie erst am Ende um.
        self.zoom_rect_screen = None 
        self.red_rect_screen = None
        self.shape_points_screen = []
        
        self.finished = False

    def mouse_callback(self, event, x, y, flags, param):
        if self.finished: return
        
        self.current_pos = (x, y)

        # STEP 1 & 2: RECHTECK ZIEHEN
        if self.step in [0, 1]:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.pt_start = (x, y)
                self.pt_end = (x, y)
                self.dragging = True
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                self.pt_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.dragging = False
                self.pt_end = (x, y)
                # Ordnen: x1 < x2
                x1, x2 = sorted([self.pt_start[0], x])
                y1, y2 = sorted([self.pt_start[1], y])
                
                if self.step == 0:
                    self.zoom_rect_screen = (x1, y1, x2, y2)
                    print(f"Zoom Screen Coords: {self.zoom_rect_screen}")
                elif self.step == 1:
                    self.red_rect_screen = (x1, y1, x2, y2)
                    print(f"Red Light Screen Coords: {self.red_rect_screen}")

        # STEP 3: PUNKTE KLICKEN
        elif self.step == 2:
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.shape_points_screen) < 4:
                    self.shape_points_screen.append((x, y))
                    print(f"Point added: {(x, y)}")

# --- HELPER: DRAW BORDER TEXT ----

def draw_rotated_letter(img, letter, center, angle, scale, color, thickness):
    """
    Draws a single letter rotated around its center.
    """
    # 1. Get size of the letter
    (fw, fh), baseline = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    

    # B Draw Letters

    # 2. Create a temporary canvas large enough to hold the rotated letter
    # We make it square based on the diagonal to ensure no clipping
    diag = int(np.sqrt(fw**2 + fh**2)) + 10
    txt_img = np.zeros((diag, diag, 3), dtype=np.uint8)
    
    # 3. Draw the letter in the center of the temp canvas
    # (Note: y is tricky in CV2, we add fh to shift it down from top)
    cx, cy = diag // 2, diag // 2
    org = (cx - fw // 2, cy + fh // 2)
    cv2.putText(txt_img, letter, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    # 4. Rotate the temp canvas
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(txt_img, M, (diag, diag))

    # 5. Blend it onto the main image
    # Calculate where to place the top-left corner of our temp canvas on the main image
    start_x = int(center[0] - cx)
    start_y = int(center[1] - cy)
    
    # Clipping logic (don't draw if it goes off the main screen edges)
    h_main, w_main, _ = img.shape
    if start_x < 0 or start_y < 0 or start_x+diag > w_main or start_y+diag > h_main:
        return # Skip drawing if it clips edge to prevent crash

    # Create a mask for the letter pixels (where the temp image is not black)
    roi = img[start_y:start_y+diag, start_x:start_x+diag]
    mask = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    # Black-out the area of the letter in the ROI
    img_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    # Take only the region of the letter from the rotated text image
    img_fg = cv2.bitwise_and(rotated, rotated, mask=mask)
    # Put the two together
    dst = cv2.add(img_bg, img_fg)
    
    # Place it back into the main image
    img[start_y:start_y+diag, start_x:start_x+diag] = dst

def get_perimeter_pos(dist, w, h, margin):
    """
    Maps a linear distance (0 -> Perimeter) to (x, y, angle_degrees)
    Path: Bottom(R->L) -> Left(B->T) -> Top(L->R) -> Right(T->B)
    """
    # Define lengths of the 4 sides
    # We adjust for margin so text doesn't hit the absolute edge
    eff_w = w - 2*margin
    eff_h = h - 2*margin
    
    perimeter = 2 * eff_w + 2 * eff_h
    d = dist % perimeter # Loop the distance
    
    


    # 1. Right Edge (Moving Bottom to Top)
    if d < eff_h:
        x = (w - margin)
        y = h - margin - d
        angle = 270
        return (x, y), angle

    # 2. Top Edge (Moving Right to Left)
    d -= eff_h
    if d < eff_w:
        x = (w - margin) - d
        y = margin
        angle = 0 # Text reading up
        return (x, y), angle

    # 3. Left Edge (Moving Top to Bottom)
    d -= eff_w
    if d < eff_h:
        x = margin
        y = margin + d
        angle = 90 # Upside down (to face center) or 0. Let's do 180 for "crawling" effect.
        return (x, y), angle

    # 4. Bottom Edge (Moving Right to Left)
    d -= eff_h
    if d < eff_w:
        x = w - margin - d
        y = h - margin
        angle = 0 # Text reading down
        return (x, y), angle
        
    return (0,0), 0

def get_corner_pos(dist, w, h, margin):
    """
    Maps a linear distance (0 -> Perimeter) to (x, y, angle_degrees)
    Path: Bottom(R->L) -> Left(B->T) -> Top(L->R) -> Right(T->B)
    """
    # Define lengths of the 4 sides
    # We adjust for margin so text doesn't hit the absolute edge
    eff_w = w - 2*margin
    eff_h = h - 2*margin
    
    perimeter = 2 * eff_w + 2 * eff_h
    d = dist % perimeter # Loop the distance
    
    


    # 1. Right Edge (Moving Bottom to Top)
    if d < eff_h:
        x = w
        y = h - margin - d
        angle = 270
        return (x, y), angle

    # 2. Top Edge (Moving Right to Left)
    d -= eff_h
    if d < eff_w:
        x = (w - margin) - d
        y = 0
        angle = 0 # Text reading up
        return (x, y), angle

    # 3. Left Edge (Moving Top to Bottom)
    d -= eff_w
    if d < eff_h:
        x = 0
        y = margin + d
        angle = 90 # Upside down (to face center) or 0. Let's do 180 for "crawling" effect.
        return (x, y), angle

    # 4. Bottom Edge (Moving Right to Left)
    d -= eff_h
    if d < eff_w:
        x = w - margin - d
        y = h
        angle = 0 # Text reading down
        return (x, y), angle
        
    return (0,0), 0

def draw_background_border(frame, w, h, start_pos, end_pos):
    """Draws a semi-transparent frame around the image   """
    overlay = frame.copy()

    rect_A_1 = (0,0)
    rect_A_2 = (0,0)

    rect_B_1 = (0,0)
    rect_B_2 = (0,0)

    rect_C_1 = (0,0)
    rect_C_2 = (0,0)

    # ends left border
    if (end_pos[0] == 0):
        rect_A_1 = (0,0)
        rect_A_2 = (BG_BAR_WIDTH,end_pos[1])

        # starts top border
        if start_pos[1] == 0:
            rect_B_1 = (BG_BAR_WIDTH,0)
            rect_B_2 = (start_pos[0],BG_BAR_WIDTH)

        # start right border
        if start_pos[0] == w:
            rect_B_1 = (BG_BAR_WIDTH,0)
            rect_B_2 = (w-BG_BAR_WIDTH,BG_BAR_WIDTH)

            rect_C_1 = (w-BG_BAR_WIDTH,0)
            rect_C_2 = (w,start_pos[1])

    # ends right border
    if (end_pos[0] == w):
        rect_A_1 = (w-BG_BAR_WIDTH,end_pos[1])
        rect_A_2 = (w,h)
        #start bot border
        if (start_pos[1] == h):
            rect_B_1 = (start_pos[0],h-BG_BAR_WIDTH)
            rect_B_2 = (0,h)
        #starts left border
        if (start_pos[0] == 0):
            rect_B_1 = (BG_BAR_WIDTH, h -BG_BAR_WIDTH)
            rect_B_2 = (w-BG_BAR_WIDTH,h)

            rect_C_1 = (0,start_pos[1])
            rect_C_2 = (BG_BAR_WIDTH, h)

    # ends top border
    if (end_pos[1] == 0):
        rect_A_1 = (end_pos[0],0)
        rect_A_2 = (w,BG_BAR_WIDTH)
        # starts right border
        if (start_pos[0] == w):
            rect_B_1 = (w - BG_BAR_WIDTH, BG_BAR_WIDTH)
            rect_B_2 = (w,start_pos[1])
        #starts bottom border
        if (start_pos[1] == h):
            rect_B_1 = (w-BG_BAR_WIDTH,BG_BAR_WIDTH)
            rect_B_2 = (w,h)

            rect_C_1 = (0,h - BG_BAR_WIDTH)
            rect_C_2 = (start_pos[0], h)


    # ends bot border
    if (end_pos[1] == h):
        rect_A_1 = (end_pos[0],h-BG_BAR_WIDTH)
        rect_A_2 = (w,h)
        # starts left border
        if (start_pos[0] == 0):
            rect_B_1 = (0,start_pos[1])
            rect_B_2 = (BG_BAR_WIDTH,h)
        #starts top border
        if (start_pos[1] == 0):
            rect_B_1 = (0,BG_BAR_WIDTH)
            rect_B_2 = (BG_BAR_WIDTH,h)

            rect_C_1 = (0,0)
            rect_C_2 = (start_pos[0], BG_BAR_WIDTH)

    

    # Draw 3 rectangles (Non-overlapping to keep corners even)
    # 1. End bar
    cv2.rectangle(overlay, rect_A_1, rect_A_2, (0,0,0), -1)
    # 2. Second Bar 
    cv2.rectangle(overlay, rect_B_1, rect_B_2, (0,0,0), -1)
    # 3. Third Bar 
    cv2.rectangle(overlay,rect_C_1,rect_C_2, (0,0,0), -1)

    
    # Apply the transparency
    cv2.addWeighted(overlay, BG_ALPHA, frame, 1 - BG_ALPHA, 0, frame)


# --- GENERATORS (STREAMS) ---

def get_raw_stream_generator(source_path, loop=True):
    """
    Yields frames from file or URL. 
    If loop=True, restarts video when it ends (good for setup).
    """
    # 1. Handle URL
    if str(source_path).startswith("http"):
        print(f"⏳ Extracting URL: {source_path}")
        ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source_path, download=False)
            source_path = info['url']
        loop = False # Livestreams don't loop

    cap = cv2.VideoCapture(source_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            if loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            else:
                break # End of stream
        yield frame
    
    cap.release()

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

# --- CORE FUNCTIONS ---


def define_traffic_light(stream):
    session = SetupSession()
    window_name = "SETUP PHASE"
    # Make Window Resizable and Fullscreen-capable
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, session.mouse_callback)
    
    print("--- SETUP START ---")
    
    final_zoom_rect = None 
    final_red_rect = None
    final_shape = []
    
    # Safe Stream Iteration (Handles End-of-Stream for YouTube/Files)
    stream_iter = iter(stream)
    last_valid_frame = None

    while True:

        try:
            # Try getting a new frame
            frame = next(stream_iter)
            if frame is not None:
                last_valid_frame = frame
        except StopIteration:
            # Video ended? Reuse last frame.
            frame = last_valid_frame
        
        # If we still have no frame (e.g. stream start fail), wait/skip
        if frame is None: 
            if last_valid_frame is not None:
                frame = last_valid_frame
            else: 
                time.sleep(0.1)
                continue

        # 1. Get NATIVE dimensions (We do NOT resize the frame here)
        orig_h, orig_w = frame.shape[:2]
        
        # Display logic variables
        display_frame = None

        # STEP 1: ZOOM BOX AUF DEM GESAMTBILD
        if session.step == 0:
            # Zeige natives Bild (ohne Resize/Upscale)
            display_frame = frame.copy()
            
            cv2.putText(display_frame, "1. BOX UM AMPEL (ENTER)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if session.dragging and session.pt_start:
                cv2.rectangle(display_frame, session.pt_start, session.current_pos, (255, 255, 0), 1)
            elif session.zoom_rect_screen:
                r = session.zoom_rect_screen
                cv2.rectangle(display_frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)

        # STEP 2 & 3: ARBEITEN IM ZOOM
        elif session.step in [1, 2]:
            # Wir haben bereits ein final_zoom_rect (aus Step 1), da Step 0 fertig ist.
            # Um sicherzugehen, nutzen wir das gespeicherte Rechteck, nicht session.zoom_rect_screen direkt.
            if not final_zoom_rect: continue

            zx1, zy1, zx2, zy2 = final_zoom_rect
            # Sicherstellen, dass Koordinaten im Bild sind
            zx1, zy1 = max(0, zx1), max(0, zy1)
            zx2, zy2 = min(orig_w, zx2), min(orig_h, zy2)

            zoom_view = frame[zy1:zy2, zx1:zx2].copy()
            if zoom_view.size == 0: continue
            
            # HIER ist der einzige Ort, wo wir upscalen (für bessere Sichtbarkeit/Klickbarkeit)
            # Wir skalieren den kleinen Ausschnitt auf die VOLLE Fenstergröße
            display_frame = cv2.resize(zoom_view, (orig_w, orig_h))
            
            cv2.putText(display_frame, "ZOOM ANSICHT", (10, orig_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

            if session.step == 1:
                cv2.putText(display_frame, "2. BOX UM ROTES LICHT (ENTER)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                if session.dragging and session.pt_start:
                    cv2.rectangle(display_frame, session.pt_start, session.current_pos, (0, 0, 255), 1)
                elif session.red_rect_screen:
                    r = session.red_rect_screen
                    cv2.rectangle(display_frame, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 2)

            elif session.step == 2:
                cv2.putText(display_frame, "3. KLICKE 4 ECKPUNKTE (ENTER)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                pts = session.shape_points_screen
                for i, pt in enumerate(pts):
                    cv2.circle(display_frame, pt, 5, (255, 0, 0), -1)
                    if i > 0: cv2.line(display_frame, pts[i-1], pt, (255, 0, 0), 2)
                if len(pts) == 4: cv2.line(display_frame, pts[0], pts[3], (255, 0, 0), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1)
        
        # --- LOGIK BEI ENTER ---
        if key == 13: # ENTER
            if session.step == 0 and session.zoom_rect_screen:
                # Step 1 fertig: Da wir im Native Mode waren, sind die Screen-Koordinaten == Original Koordinaten
                # Wir müssen nichts skalieren.
                final_zoom_rect = session.zoom_rect_screen
                print(f"Original Zoom Rect: {final_zoom_rect}")
                
                # Reset Session für nächsten Schritt
                session.pt_start = None; session.dragging = False; session.step = 1; time.sleep(0.3)
                
            elif session.step == 1 and session.red_rect_screen:
                # Step 2 fertig: Wir waren im UPSCALED Zoom Mode.
                # Wir müssen die Klicks (auf orig_w/h) runterskalieren auf die Zoom-Box Größe
                
                # Echte Größe des Zoom-Ausschnitts
                real_zoom_w = final_zoom_rect[2] - final_zoom_rect[0]
                real_zoom_h = final_zoom_rect[3] - final_zoom_rect[1]
                
                # Klick Koordinaten (auf dem großen Bildschirm)
                rx1, ry1, rx2, ry2 = session.red_rect_screen
                
                # Skalierungsfaktor (Wie viel kleiner ist der Zoom in echt?)
                scale_x = real_zoom_w / orig_w
                scale_y = real_zoom_h / orig_h
                
                # Umrechnung: Zoom_Start + (Klick * Scale)
                final_red_rect = (
                    int(final_zoom_rect[0] + (rx1 * scale_x)),
                    int(final_zoom_rect[1] + (ry1 * scale_y)),
                    int(final_zoom_rect[0] + (rx2 * scale_x)),
                    int(final_zoom_rect[1] + (ry2 * scale_y))
                )
                print(f"Original Red Rect: {final_red_rect}")
                
                session.pt_start = None; session.dragging = False; session.step = 2; time.sleep(0.3)
                
            elif session.step == 2 and len(session.shape_points_screen) == 4:
                # Step 3 fertig: Gleiche Logik wie bei Step 2 (Upscaled -> Native)
                real_zoom_w = final_zoom_rect[2] - final_zoom_rect[0]
                real_zoom_h = final_zoom_rect[3] - final_zoom_rect[1]
                
                scale_x = real_zoom_w / orig_w
                scale_y = real_zoom_h / orig_h
                
                for pt in session.shape_points_screen:
                    real_x = int(final_zoom_rect[0] + (pt[0] * scale_x))
                    real_y = int(final_zoom_rect[1] + (pt[1] * scale_y))
                    final_shape.append((real_x, real_y))
                
                session.finished = True
                cv2.destroyWindow(window_name)
                print("Setup Complete.")
                return {"red_rect": final_red_rect, "shape": final_shape}
                
        elif key == ord('q'): 
            cv2.destroyAllWindows(); sys.exit(0)
        elif key == ord('f'):
            prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if prop < 1 else cv2.WINDOW_NORMAL)


def test_show_stream(stream):
    window_name = "Test."
    cv2.namedWindow(window_name)

    for frame in stream:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == ord('q'):
            break
    


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

        frame = draw_objects(status,status_color,people, config, frame)



        draw_banner(frame, scroll_dist, status, status_color)

        #draw_border_layer(frame,scroll_dist)

        cv2.imshow("SURVEILLANCE", frame)

        scroll_dist += SCROLL_SPEED

        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()


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
        if mean_r > mean_g + 20 and mean_r > 80:
            traffic_status = "RED"
            status_color = (0, 0, 255)
        else:
            traffic_status = "GREEN"
            status_color = (0, 255, 0)
    
    return traffic_status, status_color



def detect_people(frame, interpreter, tracker):
    """
    Detects people in the frame using TFLite model.
    Scales input to a square to avoid distortion.
    """
    if frame is None: return {}

    # Init AI
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    h, w = frame.shape[:2]

    # --- 2. AI DETECTION (Standard) ---
    max_dim = max(h, w)
    square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    square_img[0:h, 0:w] = frame
    
    # Resize to model input size (300x300)
    # Use INTER_AREA for better quality when downscaling
    input_img = cv2.resize(square_img, (300, 300), interpolation=cv2.INTER_AREA)
    
    # CRITICAL FIX: Convert BGR to RGB
    # OpenCV uses BGR, but TFLite models usually expect RGB. 
    # Without this, colors are inverted (Red<->Blue) which confuses the AI.
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    
    input_data = np.expand_dims(input_img, axis=0)
        
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
        
    boxes = interpreter.get_tensor(output_details[0]['index'])[0]
    classes = interpreter.get_tensor(output_details[1]['index'])[0]
    scores = interpreter.get_tensor(output_details[2]['index'])[0]

    rects = []
    for i in range(len(scores)):
        if scores[i] > THRESHOLD and int(classes[i]) == PERSON_CLASS_ID:
            ymin, xmin, ymax, xmax = boxes[i]
            # Map coordinates back from square space to original space
            rects.append((int(xmin * max_dim), int(ymin * max_dim), int(xmax * max_dim), int(ymax * max_dim)))
    
    objects = tracker.update(rects)
    return objects

def draw_objects(status, status_color, objects, config, frame):
    # Config Unpack
    rx1, ry1, rx2, ry2 = config["red_rect"]
    shape_pts = np.array(config["shape"], np.int32)
    shape_pts = shape_pts.reshape((-1, 1, 2))


    # Persons
    for (objectID, box) in objects.items():
        (xmin, ymin, xmax, ymax) = box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {objectID}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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

def draw_border_layer(frame, scroll_dist):
    """
    The orchestrator function you requested. 
    Calculates positions and calls the user's border/text functions.
    """
    if frame is None: return
    h, w = frame.shape[:2]

    # 1. Logic for snake positions
    # Calculate length of the text string in pixels
    text_len = len(BANNER_TEXT) * letter_spacing
    
    head_dist = scroll_dist
    tail_dist = scroll_dist - text_len
    
    # 2. Draw Background
    # We use margin=0 for corner pos because background logic expects exact edges (0 or w)
    head_corner_pos, _ = get_corner_pos(head_dist, w, h, 0)
    tail_corner_pos, _ = get_corner_pos(tail_dist, w, h, 0)
    
    draw_background_border(frame, w, h, head_corner_pos, tail_corner_pos)
    
    # 3. Draw Text
    # Iterate through the string and place each character
    for i, char in enumerate(BANNER_TEXT):
        # Distance of this specific character
        # (Assuming head is first char, tail is last char)
        dist = scroll_dist - (i * letter_spacing)
        
        # Get position for letter (using actual visual margin)
        center_pos, angle = get_perimeter_pos(dist, w, h, MARGIN)
        
        draw_rotated_letter(frame, char, center_pos, angle, FONT_SCALE, TEXT_COLOR, THICKNESS)


# --- MAIN ---

def main_sample():
    print("1. Initializing Video Source...")
    raw_stream = get_raw_stream_generator(VIDEO_PATH, loop=True)

    print("3. Starting Setup Phase...")
    # Let the user define the refference pixels for the traffic light
    tl_config = define_traffic_light(raw_stream)

    print("4. Starting Surveillance Phase...")

    run_surveillance(raw_stream, tl_config)

def main_url():
    print("1. Initializing Video Source...")
    # Erstelle den Roh-Stream (Webcam, Datei oder URL)
    raw_stream = get_raw_stream_generator(VIDEO_URL, loop=True)
    
    print("2. Applying Cropping Pipeline...")

    raw_stream = crop_generator(raw_stream)

    print("3. Starting Setup Phase...")
    # Let the user define the refference pixels for the traffic light
    tl_config = define_traffic_light(raw_stream)

    print("4. Starting Surveillance Phase...")

    run_surveillance(raw_stream, tl_config)

if __name__ == "__main__":
    main_sample()