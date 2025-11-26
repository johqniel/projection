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
THRESHOLD = 0.3 # Reduced threshold for better detection
YOUTUBE_URL = "https://www.youtube.com/watch?v=6dp-bvQ7RWo" 
VIDEO_PATH = "/Users/nielo/Desktop/projection/samples/A.MOV"

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

# --- HELPER & TRACKER KLASSEN ---
def get_stream_url(yt_url):
    print(f"⏳ Extrahiere Stream-URL von: {yt_url} ...")
    ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(yt_url, download=False)
        return info['url']

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

def analyze_traffic_light(frame, box):
    h_img, w_img, _ = frame.shape
    ymin = int(max(0, box[0] * h_img))
    xmin = int(max(0, box[1] * w_img))
    ymax = int(min(h_img, box[2] * h_img))
    xmax = int(min(w_img, box[3] * w_img))
    roi = frame[ymin:ymax, xmin:xmax]
    if roi.size == 0: return (0, 165, 255), "UNKNOWN"
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    h_roi, w_roi = gray.shape
    top_half = gray[0 : h_roi//2, :]
    bottom_half = gray[h_roi//2 : h_roi, :]
    brightness_top = cv2.mean(top_half)[0]
    brightness_bottom = cv2.mean(bottom_half)[0]
    sensitivity = 10 
    if brightness_top > brightness_bottom + sensitivity: return (0, 0, 255), "ROT"
    elif brightness_bottom > brightness_top + sensitivity: return (0, 255, 0), "GRUEN"
    else: return (0, 165, 255), "AMPEL"

def calculate_letter_geometry(letter, center, angle, scale, thickness):
    (fw, fh), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    box_w = fw + (BG_PADDING_X * 2)
    box_h = fh + (BG_PADDING_Y * 2)
    corners = np.array([[-box_w/2, -box_h/2], [ box_w/2, -box_h/2], [ box_w/2,  box_h/2], [-box_w/2,  box_h/2]])
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    rotation_matrix = np.array(((c, -s), (s, c)))
    rotated_corners = np.dot(corners, rotation_matrix) + center
    return rotated_corners.astype(np.int32), (fw, fh)

def draw_text_content(img, letter, center, angle, scale, color, thickness, size):
    fw, fh = size
    diag = int(np.sqrt(fw**2 + fh**2)) + 10
    txt_img = np.zeros((diag, diag, 3), dtype=np.uint8)
    cx, cy = diag // 2, diag // 2
    org = (cx - fw // 2, cy + fh // 2)
    cv2.putText(txt_img, letter, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(txt_img, M, (diag, diag))
    start_x = int(center[0] - cx)
    start_y = int(center[1] - cy)
    h_main, w_main, _ = img.shape
    if start_x < 0 or start_y < 0 or start_x+diag > w_main or start_y+diag > h_main: return 
    roi = img[start_y:start_y+diag, start_x:start_x+diag]
    mask = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    img_bg = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(mask))
    img_fg = cv2.bitwise_and(rotated, rotated, mask=mask)
    dst = cv2.add(img_bg, img_fg)
    img[start_y:start_y+diag, start_x:start_x+diag] = dst

def get_perimeter_pos(dist, w, h, margin):
    eff_w = w - 2*margin
    eff_h = h - 2*margin
    perimeter = 2 * eff_w + 2 * eff_h
    d = dist % perimeter 
    if d < eff_h: return ((w - margin), h - margin - d), 90
    d -= eff_h
    if d < eff_w: return ((w - margin) - d, margin), 180
    d -= eff_w
    if d < eff_h: return (margin, margin + d), 270
    d -= eff_h
    if d < eff_w: return (w - margin - d, h - margin), 0
    return (0,0), 0

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return
    if RECORD_VIDEO and not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    stream_url = get_stream_url(YOUTUBE_URL)
    cap = cv2.VideoCapture(stream_url)
    
    FULL_W, FULL_H = 1280, 720
    CROP_W, CROP_H = FULL_W // 2, FULL_H // 2

    ct = CentroidTracker(maxDisappeared=20)

    out = None
    if RECORD_VIDEO:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{OUTPUT_FOLDER}/crop_rec_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        out = cv2.VideoWriter(filename, fourcc, 20.0, (CROP_W, CROP_H))

    window_name = "Quarter Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    global_distance = 0
    letter_spacing = 20 

    print("Starting Top-Right Crop Stream... Press 'q' to exit.")

    while True:
        ret, raw_frame = cap.read()
        if not ret: break
        
        full_frame = cv2.resize(raw_frame, (FULL_W, FULL_H))
        
        # 1. Das Bild schneiden (wie vorher)
        frame = full_frame[0:CROP_H, int(FULL_W - 1.5*CROP_H) : FULL_W]
        h, w, _ = frame.shape # Das ist jetzt ca 640 x 360

        # --- 2. ASPECT RATIO FIX (Letterboxing) ---
        # Wir machen das Bild quadratisch mit schwarzen Balken, bevor wir es der KI geben.
        
        max_dim = max(h, w) # Das wird die Größe unseres Quadrats (640x640)
        
        # Quadratische Leinwand erstellen (Schwarz)
        square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        
        # Das originale Bild oben links einfügen
        square_img[0:h, 0:w] = frame
        
        # Jetzt das quadratische Bild für die KI verkleinern
        input_img = cv2.resize(square_img, (300, 300))
        input_data = np.expand_dims(input_img, axis=0)

        # --- KI DENKT JETZT ---
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        person_rects = [] 
        traffic_lights = [] 

        for i in range(len(scores)):
            if scores[i] > THRESHOLD:
                class_id = int(classes[i])
                
                # --- 3. KOORDINATEN UMRECHNEN (Wichtig!) ---
                # Die KI gibt uns Koordinaten relative zum QUADRAT (max_dim).
                # Wir rechnen also mit 'max_dim', nicht mit 'h' oder 'w' einzeln.
                
                ymin = int(boxes[i][0] * max_dim)
                xmin = int(boxes[i][1] * max_dim)
                ymax = int(boxes[i][2] * max_dim)
                xmax = int(boxes[i][3] * max_dim)
                
                # Check: Liegt die Box im schwarzen Bereich? Wenn ja, ignorieren.
                if ymin > h: continue # Box ist im schwarzen Balken
                
                # Clipping: Falls die Box halb im schwarzen Balken hängt, schneiden wir sie ab
                ymax = min(ymax, h) 

                if class_id == PERSON_CLASS_ID:
                    person_rects.append((xmin, ymin, xmax, ymax))
                elif class_id == TRAFFIC_LIGHT_CLASS_ID:
                    traffic_lights.append(boxes[i]) # Hier müssten wir eigentlich auch die korrigierten Boxen speichern

        # Kleiner Fix: Traffic Lights müssen jetzt auch die korrigierten Koordinaten nutzen
        # Wir müssen die Box-Logik für Ampeln anpassen, da 'boxes[i]' noch die rohen KI-Daten hat
        # Ich speichere hier direkt die fertigen Pixel-Koordinaten ab:
        traffic_lights_pixel_coords = []
        for i in range(len(scores)):
             if scores[i] > THRESHOLD and int(classes[i]) == TRAFFIC_LIGHT_CLASS_ID:
                ymin = int(boxes[i][0] * max_dim)
                xmin = int(boxes[i][1] * max_dim)
                ymax = int(min(h, boxes[i][2] * max_dim))
                xmax = int(min(w, boxes[i][3] * max_dim))
                if ymin < h: # Nur wenn nicht im schwarzen Bereich
                    traffic_lights_pixel_coords.append((xmin, ymin, xmax, ymax))


        objects = ct.update(person_rects)

        for (objectID, box) in objects.items():
            (xmin, ymin, xmax, ymax) = box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {objectID}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Ampeln zeichnen mit korrigierten Koordinaten
        for (xmin, ymin, xmax, ymax) in traffic_lights_pixel_coords:
            # Für die Analyse brauchen wir eine "Box" im Format [ymin, xmin, ymax, xmax] relativ zum Frame (0-1)
            # Das rechnen wir kurz zurück, damit die analyze-Funktion nicht umgeschrieben werden muss
            rel_box = [ymin/h, xmin/w, ymax/h, xmax/w]
            
            color, label = analyze_traffic_light(frame, rel_box)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
            cv2.putText(frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # TEXT
        global_distance += SCROLL_SPEED
        render_queue = []
        for i, char in enumerate(BANNER_TEXT):
            char_dist = global_distance + (i * letter_spacing)
            (cx, cy), angle = get_perimeter_pos(char_dist, w, h, MARGIN)
            box_pts, text_size = calculate_letter_geometry(char, (cx, cy), angle, FONT_SCALE, THICKNESS)
            render_queue.append({"char": char, "center": (cx,cy), "angle": angle, "box": box_pts, "size": text_size})

        overlay = frame.copy()
        for item in render_queue:
            cv2.fillPoly(overlay, [item["box"]], (0, 0, 0))
        cv2.addWeighted(overlay, BG_ALPHA, frame, 1 - BG_ALPHA, 0, frame)

        for item in render_queue:
            draw_text_content(frame, item["char"], item["center"], item["angle"], FONT_SCALE, TEXT_COLOR, THICKNESS, item["size"])

        if RECORD_VIDEO and out is not None:
            out.write(frame)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    if out is not None: out.release() 
    cv2.destroyAllWindows()


def main_sample():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found.")
        return
    if RECORD_VIDEO and not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    FULL_W, FULL_H = 1280, 720
    CROP_W, CROP_H = FULL_W // 2, FULL_H // 2

    ct = CentroidTracker(maxDisappeared=20)

    out = None
    if RECORD_VIDEO:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"{OUTPUT_FOLDER}/crop_rec_{timestamp}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
        out = cv2.VideoWriter(filename, fourcc, 20.0, (CROP_W, CROP_H))

    window_name = "Quarter Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    global_distance = 0
    letter_spacing = 20 

    print("Starting Top-Right Crop Stream... Press 'q' to exit.")

    while True:
        ret, raw_frame = cap.read()
        if not ret:
            print("Video ended. Restarting ...")
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            continue
        
        full_frame = cv2.resize(raw_frame, (FULL_W, FULL_H))
        
        # 1. Das Bild schneiden (wie vorher)
        frame = full_frame[0:CROP_H, int(FULL_W - 1.5*CROP_H) : FULL_W]
        h, w, _ = frame.shape # Das ist jetzt ca 640 x 360

        # --- 2. ASPECT RATIO FIX (Letterboxing) ---
        # Wir machen das Bild quadratisch mit schwarzen Balken, bevor wir es der KI geben.
        
        max_dim = max(h, w) # Das wird die Größe unseres Quadrats (640x640)
        
        # Quadratische Leinwand erstellen (Schwarz)
        square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        
        # Das originale Bild oben links einfügen
        square_img[0:h, 0:w] = frame
        
        # Jetzt das quadratische Bild für die KI verkleinern
        input_img = cv2.resize(square_img, (300, 300))
        input_data = np.expand_dims(input_img, axis=0)

        # --- KI DENKT JETZT ---
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        person_rects = [] 
        traffic_lights = [] 

        for i in range(len(scores)):
            if scores[i] > THRESHOLD:
                class_id = int(classes[i])
                
                # --- 3. KOORDINATEN UMRECHNEN (Wichtig!) ---
                # Die KI gibt uns Koordinaten relative zum QUADRAT (max_dim).
                # Wir rechnen also mit 'max_dim', nicht mit 'h' oder 'w' einzeln.
                
                ymin = int(boxes[i][0] * max_dim)
                xmin = int(boxes[i][1] * max_dim)
                ymax = int(boxes[i][2] * max_dim)
                xmax = int(boxes[i][3] * max_dim)
                
                # Check: Liegt die Box im schwarzen Bereich? Wenn ja, ignorieren.
                if ymin > h: continue # Box ist im schwarzen Balken
                
                # Clipping: Falls die Box halb im schwarzen Balken hängt, schneiden wir sie ab
                ymax = min(ymax, h) 

                if class_id == PERSON_CLASS_ID:
                    person_rects.append((xmin, ymin, xmax, ymax))
                elif class_id == TRAFFIC_LIGHT_CLASS_ID:
                    traffic_lights.append(boxes[i]) # Hier müssten wir eigentlich auch die korrigierten Boxen speichern

        # Kleiner Fix: Traffic Lights müssen jetzt auch die korrigierten Koordinaten nutzen
        # Wir müssen die Box-Logik für Ampeln anpassen, da 'boxes[i]' noch die rohen KI-Daten hat
        # Ich speichere hier direkt die fertigen Pixel-Koordinaten ab:
        traffic_lights_pixel_coords = []
        for i in range(len(scores)):
             if scores[i] > THRESHOLD and int(classes[i]) == TRAFFIC_LIGHT_CLASS_ID:
                ymin = int(boxes[i][0] * max_dim)
                xmin = int(boxes[i][1] * max_dim)
                ymax = int(min(h, boxes[i][2] * max_dim))
                xmax = int(min(w, boxes[i][3] * max_dim))
                if ymin < h: # Nur wenn nicht im schwarzen Bereich
                    traffic_lights_pixel_coords.append((xmin, ymin, xmax, ymax))


        objects = ct.update(person_rects)

        for (objectID, box) in objects.items():
            (xmin, ymin, xmax, ymax) = box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {objectID}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Ampeln zeichnen mit korrigierten Koordinaten
        for (xmin, ymin, xmax, ymax) in traffic_lights_pixel_coords:
            # Für die Analyse brauchen wir eine "Box" im Format [ymin, xmin, ymax, xmax] relativ zum Frame (0-1)
            # Das rechnen wir kurz zurück, damit die analyze-Funktion nicht umgeschrieben werden muss
            rel_box = [ymin/h, xmin/w, ymax/h, xmax/w]
            
            color, label = analyze_traffic_light(frame, rel_box)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 3)
            cv2.putText(frame, label, (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # TEXT
        global_distance += SCROLL_SPEED
        render_queue = []
        for i, char in enumerate(BANNER_TEXT):
            char_dist = global_distance + (i * letter_spacing)
            (cx, cy), angle = get_perimeter_pos(char_dist, w, h, MARGIN)
            box_pts, text_size = calculate_letter_geometry(char, (cx, cy), angle, FONT_SCALE, THICKNESS)
            render_queue.append({"char": char, "center": (cx,cy), "angle": angle, "box": box_pts, "size": text_size})

        overlay = frame.copy()
        for item in render_queue:
            cv2.fillPoly(overlay, [item["box"]], (0, 0, 0))
        cv2.addWeighted(overlay, BG_ALPHA, frame, 1 - BG_ALPHA, 0, frame)

        for item in render_queue:
            draw_text_content(frame, item["char"], item["center"], item["angle"], FONT_SCALE, TEXT_COLOR, THICKNESS, item["size"])

        if RECORD_VIDEO and out is not None:
            out.write(frame)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    if out is not None: out.release() 
    cv2.destroyAllWindows()

if __name__ == "__main__":

    if len(VIDEO_PATH) > 0:
        main_sample()
    else:
        main()
