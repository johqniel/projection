

import cv2
import time
import sys


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
            print("reuse last frame")
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

