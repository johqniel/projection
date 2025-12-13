import cv2
import numpy as np

from config import BG_BAR_WIDTH, BANNER_TEXT, LETTER_SPACING, MARGIN, FONT_SCALE, TEXT_COLOR, THICKNESS, BG_ALPHA

# --- Functions to draw the text on the border of the frame ----
"""
Module: border_text.py

This module handles the dynamic rendering of text and borders around the perimeter of the video frame.
It is designed to create a "scrolling marquee" effect that moves along the frame edges.

Key Functions:
- `draw_border_layer(frame, scroll_dist)`: The main entry point. Orchestrates the drawing of the
  text and background border based on the current scroll distance. Call this once per frame.

- `draw_rotated_letter(...)`: Draws individual characters rotated to match the border orientation.
- `get_perimeter_pos(...)` & `get_corner_pos(...)`: Helper functions to map linear distance to (x, y) coordinates on the frame border.
- `draw_background_border(...)`: Draws the semi-transparent background behind the text.

Dependencies:
- `config.py`: specific constants (e.g., BG_BAR_WIDTH, MARGIN, FONT_SCALE).
- `cv2` (OpenCV): for drawing text and shapes.
- `numpy`: for image manipulation.
"""

# simple banner on top of the frame 
def draw_banner(frame, scroll_dist, traffic_status, status_color):
    """
    Draws a scrolling banner at the top of the frame.

    Args:
        frame (np.ndarray): The image frame to process.
        scroll_dist (int): The current scroll distance for the animation.
        traffic_status (str): The current traffic status to display.
        status_color (tuple): The color of the traffic status text (B, G, R).
    """
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

# Complex border with text around the frame 
def draw_border_layer(frame, scroll_dist):
    """
    The orchestrator function you requested.
    Calculates positions and calls the user's border/text functions.

    Args:
        frame (np.ndarray): The image frame to process.
        scroll_dist (int): The current scroll distance for the animation.
    """
    if frame is None: return
    h, w = frame.shape[:2]

    # 1. Logic for snake positions
    # Calculate positions of first and last character centers
    # Head (first char) is at scroll_dist
    last_char_dist = scroll_dist - (len(BANNER_TEXT)) * LETTER_SPACING
    
    # Add padding to cover the characters visually (half spacing on each side)
    padding = int(LETTER_SPACING)
    
    head_dist = scroll_dist + padding
    tail_dist = last_char_dist - padding
    
    # 2. Draw Background
    draw_background_border(frame, w, h, head_dist, tail_dist)
    
    # 3. Draw Text
    # Iterate through the string and place each character
    for i, char in enumerate(BANNER_TEXT):
        # Distance of this specific character
        # (Assuming head is first char, tail is last char)
        dist = scroll_dist - (i * LETTER_SPACING)
        
        # Get position for letter (using actual visual margin)
        center_pos, angle = get_perimeter_pos(dist, w, h, MARGIN)
        
        draw_rotated_letter(frame, char, center_pos, angle, FONT_SCALE, TEXT_COLOR, THICKNESS)


def draw_rotated_letter(img, letter, center, angle, scale, color, thickness):
    """
    Draws a single letter rotated around its center.

    Args:
        img (np.ndarray): The image to draw on.
        letter (str): The letter to draw.
        center (tuple): The (x, y) coordinates of the center of the letter.
        angle (float): The rotation angle in degrees.
        scale (float): The scale of the font.
        color (tuple): The color of the text (B, G, R).
        thickness (int): The thickness of the text.
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
    Maps a linear distance (0 -> Perimeter) to (x, y, angle_degrees).
    Path: Bottom(R->L) -> Left(B->T) -> Top(L->R) -> Right(T->B)

    Args:
        dist (int): The linear distance along the perimeter.
        w (int): The width of the area.
        h (int): The height of the area.
        margin (int): The margin to offset from the edge.

    Returns:
        tuple: A tuple containing ((x, y), angle).
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

    # 4. Bottom Edge (Moving Left to Right)
    d -= eff_h
    if d < eff_w:
        x = margin + d
        y = h - margin
        angle = 0 # Text reading down
        return (x, y), angle
        
    return (0,0), 0

def get_corner_pos(dist, w, h, margin):
    """
    Maps a linear distance (0 -> Perimeter) to (x, y, angle_degrees) for the corner.
    Path: Bottom(R->L) -> Left(B->T) -> Top(L->R) -> Right(T->B)

    Args:
        dist (int): The linear distance along the perimeter.
        w (int): The width of the area.
        h (int): The height of the area.
        margin (int): The margin to offset from the edge.

    Returns:
        tuple: A tuple containing ((x, y), angle).
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

    # 4. Bottom Edge (Moving Left to Right)
    d -= eff_h
    if d < eff_w:
        x = margin + d
        y = h
        angle = 0 # Text reading down
        return (x, y), angle
        
    return (0,0), 0

def draw_segment(overlay, start_d, end_d, w, h):
    """
    Helper to draw a continuous segment of the border background.
    """
    # Normalize lengths
    
    # Range of each side in the linear domain 0 -> Perimeter
    # Side 1: Right (0 -> h)
    s1_start, s1_end = 0, h
    # Side 2: Top (h -> h+w)
    s2_start, s2_end = h, h+w
    # Side 3: Left (h+w -> h+w+h)
    s3_start, s3_end = h+w, h+w+h
    # Side 4: Bottom (h+w+h -> h+w+h+w)
    s4_start, s4_end = h+w+h, h+w+h+w
    
    def intersect(range_start, range_end, seg_start, seg_end):
        return max(range_start, seg_start), min(range_end, seg_end)
    
    # 1. Check Side 1 (Right)
    i_start, i_end = intersect(s1_start, s1_end, start_d, end_d)
    if i_start < i_end:
        # Map back to coords. Side 1 goes y: h -> 0 (Bottom to Top)
        # 0 corresponds to bottom (h), h corresponds to top (0)
        # local d: 0 is bottom, h is top.
        y_bot = h - (i_start - s1_start)
        y_top = h - (i_end - s1_start)
        # Rect: Right edge. x is [w-BG_BAR_WIDTH, w]
        cv2.rectangle(overlay, (w - BG_BAR_WIDTH, y_top), (w, y_bot), (0,0,0), -1)

    # 2. Check Side 2 (Top)
    i_start, i_end = intersect(s2_start, s2_end, start_d, end_d)
    if i_start < i_end:
        # Side 2 goes x: w -> 0 (Right to Left)
        # 0 (local) is Right (w), w (local) is Left (0)
        x_right = w - (i_start - s2_start)
        x_left = w - (i_end - s2_start)
        # Rect: Top edge. y is [0, BG_BAR_WIDTH]
        cv2.rectangle(overlay, (x_left, 0), (x_right, BG_BAR_WIDTH), (0,0,0), -1)

    # 3. Check Side 3 (Left)
    i_start, i_end = intersect(s3_start, s3_end, start_d, end_d)
    if i_start < i_end:
        # Side 3 goes y: 0 -> h (Top to Bottom)
        y_top = (i_start - s3_start)
        y_bot = (i_end - s3_start)
        # Rect: Left edge. x is [0, BG_BAR_WIDTH]
        cv2.rectangle(overlay, (0, y_top), (BG_BAR_WIDTH, y_bot), (0,0,0), -1)

    # 4. Check Side 4 (Bottom)
    i_start, i_end = intersect(s4_start, s4_end, start_d, end_d)
    if i_start < i_end:
        # Side 4 goes x: 0 -> w (Left to Right)
        x_left = (i_start - s4_start)
        x_right = (i_end - s4_start)
        # Rect: Bottom edge. y is [h-BG_BAR_WIDTH, h]
        cv2.rectangle(overlay, (x_left, h-BG_BAR_WIDTH), (x_right, h), (0,0,0), -1)


def draw_background_border(frame, w, h, head_dist, tail_dist):
    """
    Draws a semi-transparent frame around the image based on start and end distances.
    Handles wrapping around the perimeter.

    Args:
        frame (np.ndarray): The image to draw on.
        w (int): The width of the frame.
        h (int): The height of the frame.
        head_dist (int): The linear distance of the head of the text.
        tail_dist (int): The linear distance of the tail of the text.
    """
    overlay = frame.copy()
    
    # We use margin=0 logic for background, so full perimeter
    perimeter = 2 * w + 2 * h
    
    # Normalize distances to [0, perimeter]
    d_start = tail_dist % perimeter
    d_end = head_dist % perimeter
    
    if d_start <= d_end:
        # Continuous segment
        draw_segment(overlay, d_start, d_end, w, h)
    else:
        # Wraps around
        # 1. d_start to Perimeter
        draw_segment(overlay, d_start, perimeter, w, h)
        # 2. 0 to d_end
        draw_segment(overlay, 0, d_end, w, h)
    
    # Apply the transparency
    cv2.addWeighted(overlay, BG_ALPHA, frame, 1 - BG_ALPHA, 0, frame)


