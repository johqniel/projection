import cv2
import numpy as np

from config import BG_BAR_WIDTH, BANNER_TEXT, letter_spacing, MARGIN, FONT_SCALE, TEXT_COLOR, THICKNESS, BG_ALPHA

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
