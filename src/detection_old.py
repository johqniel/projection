import cv2
import numpy as np
import os

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

# --- CONFIGURATION ---
MODEL_PATH = "detect.tflite" 
LABEL_PATH = "labelmap.txt"
CAMERA_INDEX = 0  # Note: On Mac, 0 is usually the Webcam. 1 is usually the USB Capture Card.
THRESHOLD = 0.5       
PERSON_CLASS_ID = 0   

# --- BANNER CONFIGURATION ---
BANNER_TEXT = " *** La Li Lu nur der Mann im Mond schaut zu... ***   "
SCROLL_SPEED = 8        # Higher is faster
TEXT_COLOR = (0, 255, 255) 
FONT_SCALE = 1.2
THICKNESS = 2
MARGIN = 40             # How far from the edge the text floats

# --- BANNER BACKGROUND ---
BG_ALPHA = 0.6          # Transparency (0.6 = 60% opaque)
BG_BAR_WIDTH = 80
SHIFT = 40

# --- Text Animation Configuration ---
letter_spacing = 25 # Pixels between letters


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

def main():
    # Check if model file exists before starting
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Did you unzip the file? Is it inside a subfolder?")
        return

    # 1. Load the TFLite model
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # 2. Initialize Video Capture
    print(f"Attempting to open camera {CAMERA_INDEX}...")
    cap = cv2.VideoCapture(CAMERA_INDEX)
    
    if not cap.isOpened():
        print("Error: Could not open video device.")
        print("Try changing CAMERA_INDEX to 0.")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 3. Setup Full Screen Window
    window_name = "People Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # Mac specific fullscreen trick
    #cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)




    # 5. Set temp vars for text
    global_distance = 0


    print("Starting stream... Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # get frame shape 
        h_frame, w_frame, _ = frame.shape
        
        
        # AI Processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]

        # Draw Boxes
        for i in range(len(scores)):
            if scores[i] > THRESHOLD and int(classes[i]) == PERSON_CLASS_ID:
                ymin = int(max(1, (boxes[i][0] * frame.shape[0])))
                xmin = int(max(1, (boxes[i][1] * frame.shape[1])))
                ymax = int(min(frame.shape[0], (boxes[i][2] * frame.shape[0])))
                xmax = int(min(frame.shape[1], (boxes[i][3] * frame.shape[1])))

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 4)


        # --- Text LOGIC ---
        global_distance += SCROLL_SPEED

        start_dist = global_distance + (1 * letter_spacing)
        end_dist = global_distance + (len(BANNER_TEXT) * letter_spacing)

        start_pos, _ = get_corner_pos(start_dist, w_frame, h_frame, MARGIN)
        end_pos, _ = get_corner_pos(end_dist,w_frame,h_frame,MARGIN)

        draw_background_border(frame,w_frame,h_frame,start_pos, end_pos)



        # Iterate through every letter in the text
        for i, char in enumerate(reversed(BANNER_TEXT)):
            # Calculate how far along the track this specific letter is
            # We add (i * spacing) to the global timer
            char_dist = global_distance + (i * letter_spacing)
            
            # Get the coordinate and angle for this specific distance
            (cx, cy), angle = get_perimeter_pos(char_dist, w_frame, h_frame, MARGIN)
            
            # Draw the letter with rotation
            draw_rotated_letter(frame, char, (cx, cy), angle, FONT_SCALE, TEXT_COLOR, THICKNESS)
        





        # Show frame

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()