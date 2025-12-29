

import cv2
import time
import sys
import numpy as np


# --- HELPER: SETUP SESSION (MOUSE LOGIC) ---
class SetupSession:
    def __init__(self):
        self.step = 0 # 0=Zoom, 1=Red Light, 2=Shape, 3=Street
        self.pt_start = None # Startpunkt (Mausklick)
        self.pt_end = None   # Endpunkt (Mausbewegung)
        self.current_pos = (0,0) # Aktuelle Mausposition
        self.dragging = False
        
        # Diese Listen speichern die Ergebnisse in SCREEN Koordinaten (0-640)
        # Wir rechnen sie erst am Ende um.
        self.zoom_rect_screen = None 
        self.red_rect_screen = None
        self.shape_points_screen = []
        self.street_points_screen = []
        
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
                if len(self.shape_points_screen) >= 4:
                    self.shape_points_screen = []
                
                self.shape_points_screen.append((x, y))
                print(f"Point added: {(x, y)}")

        # STEP 4: STREET DEFINITION (4 POINTS)
        elif self.step == 3:
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.street_points_screen) >= 4:
                    self.street_points_screen = []
                
                self.street_points_screen.append((x, y))
                print(f"Street Point added: {(x, y)}")




                
def define_traffic_light(stream):
    """
    Returns a LIST of config dicts:
    [
      {"red_rect": ..., "shape": ..., "street": ...},
      ...
    ]
    """
    configs = []
    
    while True:
        # Run one session
        cfg = _define_single_traffic_light(stream)
        configs.append(cfg)
        
        # Ask to continue using GUI (Console input is blocking and often invisible)
        if not _ask_to_continue():
            break
            
    print(f"Setup Complete. {len(configs)} zones defined.")
    return configs

def _ask_to_continue():
    """
    Shows a window asking user to continue or not.
    Returns True if 'y', False otherwise.
    """
    window_name = "Continue?"
    img = np.zeros((300, 600, 3), dtype=np.uint8)
    
    cv2.putText(img, "Add another Traffic Light?", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "y = YES, n = NO", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow(window_name, img)
    
    print("Waiting for user response (y/n) in 'Continue?' window...")
    
    res = False
    while True:
        key = cv2.waitKey(0)
        if key == ord('y'):
            res = True
            break
        elif key == ord('n') or key == 27: # ESC or n
            res = False
            break
            
    cv2.destroyWindow(window_name)
    return res

def _define_single_traffic_light(stream):
    session = SetupSession()
    window_name = "SETUP PHASE"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, session.mouse_callback)
    
    print("--- SETUP NEW ZONE ---")
    
    final_zoom_rect = None 
    final_red_rect = None
    final_shape = []
    final_street = []
    
    stream_iter = iter(stream)
    last_valid_frame = None

    while True:
        try:
            frame = next(stream_iter)
            if frame is not None: last_valid_frame = frame
        except StopIteration:
            frame = last_valid_frame
        
        if frame is None: 
            if last_valid_frame is not None: frame = last_valid_frame
            else: 
                time.sleep(0.1); continue

        orig_h, orig_w = frame.shape[:2]
        display_frame = None

        if session.step == 0:
            display_frame = frame.copy()
            cv2.putText(display_frame, "1. BOX UM AMPEL (ENTER)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if session.dragging and session.pt_start:
                cv2.rectangle(display_frame, session.pt_start, session.current_pos, (255, 255, 0), 1)
            elif session.zoom_rect_screen:
                r = session.zoom_rect_screen
                cv2.rectangle(display_frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)

        elif session.step in [1, 2]:
            if not final_zoom_rect: continue
            zx1, zy1, zx2, zy2 = final_zoom_rect
            zx1, zy1 = max(0, zx1), max(0, zy1)
            zx2, zy2 = min(orig_w, zx2), min(orig_h, zy2)

            zoom_view = frame[zy1:zy2, zx1:zx2].copy()
            if zoom_view.size == 0: continue
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

        elif session.step == 3:
            display_frame = frame.copy()
            cv2.putText(display_frame, "4. KLICKE 4 PUNKTE FUER STRASSE (ENTER)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            pts = session.street_points_screen
            for i, pt in enumerate(pts):
                cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)
                if i > 0: cv2.line(display_frame, pts[i-1], pt, (0, 0, 255), 2)
            if len(pts) == 4: cv2.line(display_frame, pts[0], pts[3], (0, 0, 255), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1)
        
        if key == 13: # ENTER
            if session.step == 0 and session.zoom_rect_screen:
                final_zoom_rect = session.zoom_rect_screen
                session.pt_start = None; session.dragging = False; session.step = 1; time.sleep(0.3)
            elif session.step == 1 and session.red_rect_screen:
                real_zoom_w = final_zoom_rect[2] - final_zoom_rect[0]
                real_zoom_h = final_zoom_rect[3] - final_zoom_rect[1]
                rx1, ry1, rx2, ry2 = session.red_rect_screen
                scale_x = real_zoom_w / orig_w
                scale_y = real_zoom_h / orig_h
                final_red_rect = (
                    int(final_zoom_rect[0] + (rx1 * scale_x)),
                    int(final_zoom_rect[1] + (ry1 * scale_y)),
                    int(final_zoom_rect[0] + (rx2 * scale_x)),
                    int(final_zoom_rect[1] + (ry2 * scale_y))
                )
                session.pt_start = None; session.dragging = False; session.step = 2; time.sleep(0.3)
            elif session.step == 2 and len(session.shape_points_screen) == 4:
                real_zoom_w = final_zoom_rect[2] - final_zoom_rect[0]
                real_zoom_h = final_zoom_rect[3] - final_zoom_rect[1]
                scale_x = real_zoom_w / orig_w
                scale_y = real_zoom_h / orig_h
                for pt in session.shape_points_screen:
                    real_x = int(final_zoom_rect[0] + (pt[0] * scale_x))
                    real_y = int(final_zoom_rect[1] + (pt[1] * scale_y))
                    final_shape.append((real_x, real_y))
                session.pt_start = None; session.dragging = False; session.step = 3; time.sleep(0.3)
            elif session.step == 3 and len(session.street_points_screen) == 4:
                final_street = session.street_points_screen
                session.finished = True
                cv2.destroyWindow(window_name)
                return {"red_rect": final_red_rect, "shape": final_shape, "street": final_street}

        elif key == ord('q'): 
            cv2.destroyAllWindows(); sys.exit(0)
        elif key == ord('f'):
            prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if prop < 1 else cv2.WINDOW_NORMAL)


def define_crop_area(stream):
    """
    Allows the user to define a crop area for the stream.
    Returns (x1, y1, x2, y2) or None.
    """
    session = SetupSession()
    # Reuse Step 0 logic (Zoom Rect) for Crop Area
    session.step = 0 
    
    window_name = "DEFINE CROP AREA"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, session.mouse_callback)
    
    print("--- CROP SETUP START ---")
    
    stream_iter = iter(stream)
    last_valid_frame = None
    final_crop_rect = None

    while True:
        try:
            frame = next(stream_iter)
            if frame is not None:
                last_valid_frame = frame
        except StopIteration:
            frame = last_valid_frame
        
        if frame is None:
            if last_valid_frame is not None:
                frame = last_valid_frame
            else:
                time.sleep(0.1)
                continue

        display_frame = frame.copy()
        
        cv2.putText(display_frame, "DRAW CROP AREA (ENTER to Confirm, ENTER without draw for FULL)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if session.dragging and session.pt_start:
            cv2.rectangle(display_frame, session.pt_start, session.current_pos, (0, 255, 0), 1)
        elif session.zoom_rect_screen:
            r = session.zoom_rect_screen
            cv2.rectangle(display_frame, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)

        cv2.imshow(window_name, display_frame)
        key = cv2.waitKey(1)

        if key == 13: # ENTER
            if session.zoom_rect_screen:
                final_crop_rect = session.zoom_rect_screen
                print(f"Crop Area Defined: {final_crop_rect}")
            else:
                print("No Crop Area Defined. Using Full Stream.")
                final_crop_rect = None
            
            break
        
        elif key == ord('q'):
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyWindow(window_name)
    return final_crop_rect

