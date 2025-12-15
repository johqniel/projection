import cv2
import numpy as np
import os
import time
import math

from config import STREET_VISIBLE, BOX_THICKNESS_DEFAULT, BOX_THICKNESS_JAYWALKING, BOX_THICKNESS_TRAFFIC_LIGHT

# --- CACHE ---
CACHED_EMOJI_IMG = None

# --- STATE ---
EMOJI_PARTICLES = []

# --- HELPERS ---
def _load_emoji_cache(img_path="assets/emoji.png"):
    global CACHED_EMOJI_IMG
    if CACHED_EMOJI_IMG is None:
        if os.path.exists(img_path):
            loaded = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if loaded is not None:
                if loaded.shape[2] != 4:
                    print("WARNING: Image does not have an alpha channel.")
                CACHED_EMOJI_IMG = loaded

def _detect_jaywalkers(results_list, objects, street_masks, w, h):
    """
    Returns two lists: safe_persons, jaywalking_persons
    Iterates through ALL configured zones.
    If a person is violating ANY red light zone, they are a jaywalker.
    """
    safe = []
    jaywalkers = []
    
    for (objectID, box) in objects.items():
        (xmin, ymin, xmax, ymax) = box
        
        is_jaywalking = False
        
        # Check against ALL zones
        # results_list: [(status, color), (status, color)...]
        # street_masks: [mask1, mask2...]
        for i, (status, color) in enumerate(results_list):
            if status == "RED":
                mask = street_masks[i]
                if mask is None: continue
                
                touches_border = (xmin <= 0 or ymin <= 0 or xmax >= w or ymax >= h)
                mask_roi = None
                
                if touches_border:
                    b_y1, b_y2 = max(0, ymin), min(h, ymax)
                    b_x1, b_x2 = max(0, xmin), min(w, xmax)
                    if b_y2 > b_y1 and b_x2 > b_x1:
                        mask_roi = mask[b_y1:b_y2, b_x1:b_x2]
                else:
                    b_y1 = max(0, ymax - 5)
                    b_y2 = min(h, ymax)
                    b_x1, b_x2 = max(0, xmin), min(w, xmax)
                    if b_y2 > b_y1 and b_x2 > b_x1:
                         mask_roi = mask[b_y1:b_y2, b_x1:b_x2]
                
                if mask_roi is not None and cv2.countNonZero(mask_roi) > 0:
                    is_jaywalking = True
                    break
        
        if is_jaywalking:
            jaywalkers.append((objectID, box))
        else:
            safe.append((objectID, box))
            
    return safe, jaywalkers


def overlay_image_alpha(img, img_overlay, x, y, size):
    img_overlay = cv2.resize(img_overlay, (size, size))
    h, w, _ = img.shape
    
    y1, y2 = y - size // 2, y + size // 2
    x1, x2 = x - size // 2, x + size // 2
    
    y1, y2 = max(0, y1), min(h, y2)
    x1, x2 = max(0, x1), min(w, x2)
    
    h_slice = y2 - y1
    w_slice = x2 - x1
    
    if h_slice <= 0 or w_slice <= 0: return

    ov_y1 = max(0, -(y - size // 2))
    ov_x1 = max(0, -(x - size // 2))
    ov_y2 = ov_y1 + h_slice
    ov_x2 = ov_x1 + w_slice

    overlay_crop = img_overlay[ov_y1:ov_y2, ov_x1:ov_x2]
    bg_crop = img[y1:y2, x1:x2]
    
    if overlay_crop.shape[2] == 4:
        base_alpha = overlay_crop[:, :, 3] / 255.0
        overlay_rgb = overlay_crop[:, :, :3]
    else:
        base_alpha = np.ones((overlay_crop.shape[0], overlay_crop.shape[1]), dtype=np.float32)
        overlay_rgb = overlay_crop

    white_mask = np.all(overlay_rgb >= [240, 240, 240], axis=-1)
    base_alpha[white_mask] = 0.0

    alpha_mask = base_alpha[:, :, np.newaxis]
    blended = (overlay_rgb * alpha_mask + bg_crop * (1.0 - alpha_mask))
    img[y1:y2, x1:x2] = blended.astype(np.uint8)


# --- REQUESTED FUNCTIONS ---

def draw_street_polygon(frame, config):
    street_pts = np.array(config.get("street", []), np.int32)
    if len(street_pts) > 0:
        street_pts = street_pts.reshape((-1, 1, 2))
        if STREET_VISIBLE:
            cv2.polylines(frame, [street_pts], isClosed=True, color=(255, 255, 0), thickness=2)

def draw_green_persons(frame, green_persons):
    """
    Draws boxes for 'safe' persons.
    green_persons: list of (objectID, box)
    """
    for (objectID, box) in green_persons:
        (xmin, ymin, xmax, ymax) = box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), BOX_THICKNESS_DEFAULT)
        cv2.putText(frame, f"ID {objectID}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def draw_jaywalking_persons_striped(frame, jaywalkers):
    """
    Draws boxes for jaywalkers with a pulsating, marching ants effect.
    Stripes run around the person.
    """
    # Animation parameters
    dash_length = 20
    gap_length = 15
    total_segment = dash_length + gap_length
    
    t = time.time()
    
    # 1. Faster "Running" Effect
    offset = int(t * 100) % total_segment
    
    # 2. Intense Pulse (Faster & Wider Range)
    # Sine wave 0.0 -> 1.0, speed 15
    pulse_norm = (math.sin(t * 15) + 1) / 2 
    
    # 3. Flashing Color
    # Mostly Red, but flashes White at peak intensity
    if pulse_norm > 0.8:
        # FLASH!
        color = (255, 255, 255) 
    else:
        # Deep Red pulsing
        r_val = int(150 + 105 * pulse_norm) # 150 -> 255
        color = (0, 0, r_val)
    
    # 4. Pulsating Thickness (Get wider)
    # Base config thickness + up to 10px extra
    thickness = int(BOX_THICKNESS_JAYWALKING + 10 * pulse_norm)
    
    for (objectID, box) in jaywalkers:
        (xmin, ymin, xmax, ymax) = box
        
        # Define the perimeter as a sequence of points
        # Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left -> Top-Left
        pts = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        
        # Iterate sides
        for i in range(4):
            pt1 = pts[i]
            pt2 = pts[(i + 1) % 4]
            
            # Vector math to move along the line
            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist == 0: continue
            
            ux = dx / dist
            uy = dy / dist
            
            # Start drawing dashes
            # We want to start 'offset' pixels BEFORE the line starts conceptually,
            # so the pattern flows continuously across corners.
            # Simplified: just start from 0 with phase shift
            
            current_dist = -offset
            while current_dist < dist:
                start_d = max(0, current_dist)
                end_d = min(dist, current_dist + dash_length)
                
                if end_d > start_d:
                    p1 = (int(pt1[0] + ux * start_d), int(pt1[1] + uy * start_d))
                    p2 = (int(pt1[0] + ux * end_d), int(pt1[1] + uy * end_d))
                    cv2.line(frame, p1, p2, color, thickness)
                
                current_dist += total_segment

        cv2.putText(frame, f"ID {objectID}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def draw_traffic_light(frame, config, status_color):
    shape_pts = np.array(config["shape"], np.int32)
    shape_pts = shape_pts.reshape((-1, 1, 2))
    cv2.polylines(frame, [shape_pts], isClosed=True, color=status_color, thickness=BOX_THICKNESS_TRAFFIC_LIGHT)

def draw_emojis(frame, use_image=True):
    """
    Updates and draws emoji particles.
    """
    global EMOJI_PARTICLES, CACHED_EMOJI_IMG
    
    if use_image:
        _load_emoji_cache()

    alive_particles = []
    for p in EMOJI_PARTICLES:
        p['x'] += p['vx']
        p['y'] += p['vy']
        p['life'] -= 1
        
        if p['life'] > 0:
            if use_image and CACHED_EMOJI_IMG is not None:
                 overlay_image_alpha(frame, CACHED_EMOJI_IMG, int(p['x']), int(p['y']), p['size'])
            else:
                # Drawn fallback
                 cv2.circle(frame, (int(p['x']), int(p['y'])), p['size'], (0, 255, 255), -1)
            
            alive_particles.append(p)
            
    EMOJI_PARTICLES = alive_particles


# --- ORCHESTRATOR ---

def draw_objects(results_list, objects, config_list, frame, street_masks, use_emojis=True):
    """
    Main Orchestrator for Multiple Zones.
    results_list: List of (status, status_color)
    config_list: List of config dicts
    street_masks: List of mask images
    """
    h, w = frame.shape[:2]
    
    # 1. Draw ALL Streets & Traffic Lights
    for i, config in enumerate(config_list):
        draw_street_polygon(frame, config)
        
        status, status_color = results_list[i]
        draw_traffic_light(frame, config, status_color)
    
    # 2. Classify Persons (Against all zones)
    green_p, red_p = _detect_jaywalkers(results_list, objects, street_masks, w, h)
    
    # 3. Draw Persons
    draw_green_persons(frame, green_p)
    draw_jaywalking_persons_striped(frame, red_p)
    
    # 4. Spawn Particles for Jaywalkers (if enabled)
    if use_emojis:
        for (oid, box) in red_p:
            (xmin, ymin, xmax, ymax) = box
            EMOJI_PARTICLES.append({
                'x': float(xmin + xmax) / 2,
                'y': float(ymin),
                'vx': np.random.uniform(-2, 2),
                'vy': np.random.uniform(-5, -2),
                'size': np.random.randint(20, 40),
                'life': 30
            })
            
    # 6. Update & Draw Particles (if enabled)
    if use_emojis:
        draw_emojis(frame, use_image=True)

    return frame
