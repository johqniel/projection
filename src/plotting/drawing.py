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

