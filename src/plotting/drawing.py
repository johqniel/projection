import cv2
import numpy as np

from config import STREET_VISIBLE

def draw_objects(status, status_color, objects, config, frame, street_mask):
    # Config Unpack
    rx1, ry1, rx2, ry2 = config["red_rect"]
    shape_pts = np.array(config["shape"], np.int32)
    shape_pts = shape_pts.reshape((-1, 1, 2))
    
    # Street Polygon
    street_pts = np.array(config.get("street", []), np.int32)
    if len(street_pts) > 0:
        street_pts = street_pts.reshape((-1, 1, 2))
        # Draw street area (optional, maybe semi-transparent or just outline)
        if STREET_VISIBLE:
            cv2.polylines(frame, [street_pts], isClosed=True, color=(255, 255, 0), thickness=2)


    # Persons
    h, w = frame.shape[:2]
    
    for (objectID, box) in objects.items():
        (xmin, ymin, xmax, ymax) = box
        
        # Default color
        box_color = (0, 255, 0) # Green
        
        # Check intersection with street if traffic light is RED
        if status == "RED" and street_mask is not None:
            is_jaywalking = False
            
            # Check if touching border
            touches_border = (xmin <= 0 or ymin <= 0 or xmax >= w or ymax >= h)
            
            if touches_border:
                # Case 2: Touching border -> Check if ANY part overlaps
                # ROI of the box in the mask
                # Ensure coords are within bounds
                b_y1, b_y2 = max(0, ymin), min(h, ymax)
                b_x1, b_x2 = max(0, xmin), min(w, xmax)
                
                if b_y2 > b_y1 and b_x2 > b_x1:
                    mask_roi = street_mask[b_y1:b_y2, b_x1:b_x2]
                    non_zero = cv2.countNonZero(mask_roi)
                    if non_zero > 0:
                        is_jaywalking = True
            else:
                # Case 1: Not touching border -> Check if LOWER SIDE touches street
                # We check a thin strip at the bottom of the box
                b_y1 = max(0, ymax - 5) # Check last 5 pixels? Or just the line? User said "lower side".
                b_y2 = min(h, ymax)
                b_x1, b_x2 = max(0, xmin), min(w, xmax)
                
                if b_y2 > b_y1 and b_x2 > b_x1:
                    mask_roi = street_mask[b_y1:b_y2, b_x1:b_x2]
                    non_zero = cv2.countNonZero(mask_roi)
                    if non_zero > 0:
                        is_jaywalking = True

            if is_jaywalking:
                box_color = (0, 0, 255) # Red (Jaywalking!)

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
        cv2.putText(frame, f"ID {objectID}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Traffic Light Shape
    cv2.polylines(frame, [shape_pts], isClosed=True, color=status_color, thickness=3)

    return(frame)


# Global particle state
EMOJI_PARTICLES = []

def draw_cool_emoji(frame, x, y, size=20):
    """Draws a simple 'cool' emoji (yellow face + sunglasses)."""
    # 1. Yellow Face
    cv2.circle(frame, (x, y), size, (0, 255, 255), -1) # BGR: Yellow
    cv2.circle(frame, (x, y), size, (0, 0, 0), 1)      # Outline
    
    # 2. Sunglasses (Black)
    # Lenses
    off_x = int(size * 0.4)
    off_y = int(size * 0.1)
    len_rad = int(size * 0.35)
    
    cv2.circle(frame, (x - off_x, y - off_y), len_rad, (0, 0, 0), -1)
    cv2.circle(frame, (x + off_x, y - off_y), len_rad, (0, 0, 0), -1)
    
    # Bridge
    cv2.line(frame, (x - off_x, y - off_y), (x + off_x, y - off_y), (0, 0, 0), 2)
    
    # Smile (Simple curve)
    # cv2.ellipse(frame, center, axes, angle, startAngle, endAngle, color, thickness)
    cv2.ellipse(frame, (x, y + int(size*0.2)), (int(size*0.5), int(size*0.3)), 0, 20, 160, (0, 0, 0), 2)

def draw_objects_emojis(status, status_color, objects, config, frame, street_mask):
    """
    Same as draw_objects, but with 'cool' emojis floating away for jaywalkers.
    """
    global EMOJI_PARTICLES
    
    # 1. Standard Drawing Logic (Copied from draw_objects)
    # Config Unpack
    rx1, ry1, rx2, ry2 = config["red_rect"]
    shape_pts = np.array(config["shape"], np.int32)
    shape_pts = shape_pts.reshape((-1, 1, 2))
    
    # Street Polygon
    street_pts = np.array(config.get("street", []), np.int32)
    if len(street_pts) > 0:
        street_pts = street_pts.reshape((-1, 1, 2))
        if STREET_VISIBLE:
            cv2.polylines(frame, [street_pts], isClosed=True, color=(255, 255, 0), thickness=2)

    # Persons
    h, w = frame.shape[:2]
    
    for (objectID, box) in objects.items():
        (xmin, ymin, xmax, ymax) = box
        
        box_color = (0, 255, 0) # Green
        is_jaywalking = False
        
        # Check intersection with street if traffic light is RED
        if status == "RED" and street_mask is not None:
            # Check if touching border
            touches_border = (xmin <= 0 or ymin <= 0 or xmax >= w or ymax >= h)
            
            if touches_border:
                b_y1, b_y2 = max(0, ymin), min(h, ymax)
                b_x1, b_x2 = max(0, xmin), min(w, xmax)
                if b_y2 > b_y1 and b_x2 > b_x1:
                    mask_roi = street_mask[b_y1:b_y2, b_x1:b_x2]
                    if cv2.countNonZero(mask_roi) > 0:
                        is_jaywalking = True
            else:
                b_y1 = max(0, ymax - 5)
                b_y2 = min(h, ymax)
                b_x1, b_x2 = max(0, xmin), min(w, xmax)
                if b_y2 > b_y1 and b_x2 > b_x1:
                    mask_roi = street_mask[b_y1:b_y2, b_x1:b_x2]
                    if cv2.countNonZero(mask_roi) > 0:
                        is_jaywalking = True

            if is_jaywalking:
                box_color = (0, 0, 255) # Red
                
                # --- EMOJI SPAWN LOGIC ---
                # Retrieve tracked object state if possible, or just spawn randomly
                # We spawn continuously if they are red
                EMOJI_PARTICLES.append({
                    'x': float(xmin + xmax) / 2,
                    'y': float(ymin), # Start at head
                    'vx': np.random.uniform(-2, 2),
                    'vy': np.random.uniform(-5, -2), # Float UP
                    'size': np.random.randint(15, 25),
                    'life': 30 # frames
                })

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
        cv2.putText(frame, f"ID {objectID}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # Traffic Light Shape
    cv2.polylines(frame, [shape_pts], isClosed=True, color=status_color, thickness=3)

    # 2. Emoji Particle Update & Draw
    alive_particles = []
    for p in EMOJI_PARTICLES:
        p['x'] += p['vx']
        p['y'] += p['vy']
        p['life'] -= 1
        
        if p['life'] > 0:
            # Draw
            draw_cool_emoji(frame, int(p['x']), int(p['y']), p['size'])
            alive_particles.append(p)
            
    EMOJI_PARTICLES = alive_particles


    return frame


# --- IMAGE EMOJI SUPPORT ---
import os
CACHED_EMOJI_IMG = None

def overlay_image_alpha(img, img_overlay, x, y, size):
    """Overlays rgba image on rgb image at x,y with given size."""
    
    # 1. Resize Overlay
    img_overlay = cv2.resize(img_overlay, (size, size))
    
    h, w, _ = img.shape
    y1, y2 = y - size // 2, y + size // 2
    x1, x2 = x - size // 2, x + size // 2
    
    # Clip coordinates
    y1, y2 = max(0, y1), min(h, y2)
    x1, x2 = max(0, x1), min(w, x2)
    
    # Dimensions of the slice
    h_slice = y2 - y1
    w_slice = x2 - x1
    
    if h_slice <= 0 or w_slice <= 0:
        return

    # Extract slices
    # Note: The overlay might be clipped too, so we need to calculate which part of overlay to use
    # Offset in overlay
    ov_y1 = max(0, -(y - size // 2))
    ov_x1 = max(0, -(x - size // 2))
    ov_y2 = ov_y1 + h_slice
    ov_x2 = ov_x1 + w_slice

    overlay_crop = img_overlay[ov_y1:ov_y2, ov_x1:ov_x2]
    bg_crop = img[y1:y2, x1:x2]
    
    # 1. Prepare RGB and Base Alpha
    if overlay_crop.shape[2] == 4:
        base_alpha = overlay_crop[:, :, 3] / 255.0
        overlay_rgb = overlay_crop[:, :, :3]
    else:
        # Default to opaque (1.0) if no alpha channel
        base_alpha = np.ones((overlay_crop.shape[0], overlay_crop.shape[1]), dtype=np.float32)
        overlay_rgb = overlay_crop

    # 2. Mask out White (or near white)
    # Check if pixels are > 240 in all channels (to catch compression/resizing artifacts around white)
    # OpenCV is BGR, white is [255, 255, 255]
    white_mask = np.all(overlay_rgb >= [240, 240, 240], axis=-1)
    
    # Update alpha: Where white, alpha = 0
    base_alpha[white_mask] = 0.0

    # 3. Blend
    # Broadcast alpha to 3 channels for vectorization (H, W, 1)
    alpha_mask = base_alpha[:, :, np.newaxis]
    
    # Blend: New = Alpha * Overlay + (1 - Alpha) * Background
    # We process as float to avoid overflow/underflow, then cast back to uint8
    blended = (overlay_rgb * alpha_mask + bg_crop * (1.0 - alpha_mask))
    img[y1:y2, x1:x2] = blended.astype(np.uint8)


def draw_image_emoji(frame, x, y, size, img_path="assets/emoji.png"):
    global CACHED_EMOJI_IMG
    
    # Cache Load
    if CACHED_EMOJI_IMG is None:
        if os.path.exists(img_path):
            # Load with Alpha
            loaded = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if loaded is not None:
                print(f"DEBUG: Loaded emoji image. Shape: {loaded.shape}")
                if loaded.shape[2] != 4:
                    print("WARNING: Image does not have an alpha channel (4th channel). Transparency will not work.")
                CACHED_EMOJI_IMG = loaded
        
        if CACHED_EMOJI_IMG is None:
            # Fallback
            draw_cool_emoji(frame, x, y, size)
            return

    overlay_image_alpha(frame, CACHED_EMOJI_IMG, x, y, size)


def draw_objects_image_emojis(status, status_color, objects, config, frame, street_mask):
    """
    Same as draw_objects_emojis but uses assets/emoji.png.
    """
    global EMOJI_PARTICLES
    
    # Reuse the particle update logic but change the draw call
    # 1. Update Particles & Draw Persons (Redundant code but requested as separate function)
    
    # --- COPY OF LOGIC from draw_objects_emojis ---
    # We can effectively just call draw_objects_emojis but we need to inject the different draw function.
    # But since Python checks globals or strict function calls, let's just copy the logic or refactor.
    # User asked for a second function. I will duplicate logic to ensure it's a "second function" as requested.
    
    # Config Unpack
    rx1, ry1, rx2, ry2 = config["red_rect"]
    shape_pts = np.array(config["shape"], np.int32)
    shape_pts = shape_pts.reshape((-1, 1, 2))
    
    street_pts = np.array(config.get("street", []), np.int32)
    if len(street_pts) > 0:
        street_pts = street_pts.reshape((-1, 1, 2))
        if STREET_VISIBLE:
            cv2.polylines(frame, [street_pts], isClosed=True, color=(255, 255, 0), thickness=2)

    h, w = frame.shape[:2]
    
    for (objectID, box) in objects.items():
        (xmin, ymin, xmax, ymax) = box
        box_color = (0, 255, 0)
        is_jaywalking = False
        
        if status == "RED" and street_mask is not None:
            touches_border = (xmin <= 0 or ymin <= 0 or xmax >= w or ymax >= h)
            if touches_border:
                b_y1, b_y2 = max(0, ymin), min(h, ymax)
                b_x1, b_x2 = max(0, xmin), min(w, xmax)
                if b_y2 > b_y1 and b_x2 > b_x1:
                    mask_roi = street_mask[b_y1:b_y2, b_x1:b_x2]
                    if cv2.countNonZero(mask_roi) > 0: is_jaywalking = True
            else:
                b_y1, b_y2 = max(0, ymax - 5), min(h, ymax)
                b_x1, b_x2 = max(0, xmin), min(w, xmax)
                if b_y2 > b_y1 and b_x2 > b_x1:
                    mask_roi = street_mask[b_y1:b_y2, b_x1:b_x2]
                    if cv2.countNonZero(mask_roi) > 0: is_jaywalking = True

            if is_jaywalking:
                box_color = (0, 0, 255)
                EMOJI_PARTICLES.append({
                    'x': float(xmin + xmax) / 2,
                    'y': float(ymin),
                    'vx': np.random.uniform(-2, 2),
                    'vy': np.random.uniform(-5, -2),
                    'size': np.random.randint(20, 40), # Slightly larger for images
                    'life': 30
                })

        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), box_color, 2)
        cv2.putText(frame, f"ID {objectID}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    cv2.polylines(frame, [shape_pts], isClosed=True, color=status_color, thickness=3)

    # 2. Emoji Particle Update & Draw
    alive_particles = []
    for p in EMOJI_PARTICLES:
        p['x'] += p['vx']
        p['y'] += p['vy']
        p['life'] -= 1
        
        if p['life'] > 0:
            # Draw IMAGE
            draw_image_emoji(frame, int(p['x']), int(p['y']), p['size'])
            alive_particles.append(p)
            
    EMOJI_PARTICLES = alive_particles

    return frame
