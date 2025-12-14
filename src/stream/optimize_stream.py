import cv2
import time
import numpy as np


from config import FULL_W, FULL_H, GAMMA, CLIP_LIMIT

def crop_generator(input_stream, crop_rect=None):
    """
    Consumes a stream, crops it, and yields cropped frames.
    If crop_rect is None, it yields the full frame (resized to standard).
    """
    
    for frame in input_stream:
        # Crop if defined
        if crop_rect:
            x1, y1, x2, y2 = crop_rect
            # Ensure coordinates are valid
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            frame = frame[y1:y2, x1:x2]
        else:
            # Resize to standard if needed (to keep UI consistent) ONLY if not cropping
            frame = cv2.resize(frame, (FULL_W, FULL_H))
        
        yield frame


def reduce_framerate(stream, fps, target_fps=25, real_time_playback=False):
    """
    Yields frames from the stream at a reduced framerate.
    If real_time_playback is True, it sleeps to match the timing of skipped frames (for files).
    """
    skip_frames = max(1, int(fps / target_fps))
    print(f"Reducing Framerate: Skipping every {skip_frames} frames (Input: {fps:.2f} -> Target: {target_fps})")
    
    count = 0
    for frame in stream:
        count += 1
        if count % skip_frames == 0:
            yield frame
        else: 
            if real_time_playback:
                time.sleep(1.0/fps)
            continue


def flip_stream_generator(stream, flip_code):
    """
    Yields flipped frames from the stream.
    """
    for frame in stream:
        yield cv2.flip(frame, flip_code)


def apply_gamma(image, gamma=0.5):
    # If gamma is 0.5, we want the exponent to be 0.5 (square root), which brightens.
    # We do NOT invert it this time.
    
    table = np.array([((i / 255.0) ** gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)

def apply_clahe(image,clip_limit = 2.0, tile_grid_size = (8, 8)):
    # 1. Convert to LAB color space (we only want to brighten the 'L' - Lightness channel)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # 2. Split the channels
    l, a, b = cv2.split(lab)
    
    # 3. Apply CLAHE to the L-channel
    # clipLimit -> Threshold for contrast limiting (try 2.0 to 4.0)
    # tileGridSize -> Size of the grid for histogram equalization
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    # 4. Merge channels back and convert to BGR
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final

def enhance_light_generator(input_stream):
    """
    Generator that takes a stream of frames and yields brighter frames.
    """
    for frame in input_stream:
        # Apply the correction
        enhanced_frame = apply_clahe(frame, CLIP_LIMIT)
        yield enhanced_frame

import cv2

def invert_colors_generator(input_stream):
    """
    Generator that inverts the colors of every frame in the stream.
    White becomes Black, Blue becomes Orange, etc.
    """
    for frame in input_stream:
        # cv2.bitwise_not calculates (255 - pixel_value) for every pixel
        inverted_frame = cv2.bitwise_not(frame)
        yield inverted_frame

def invert_stream_data_generator(stream_data):
    """
    Generator that inverts colors for a stream of data tuples.
    Expected format: (frame, ...)
    Yields: (inverted_frame, ...)
    """
    for item in stream_data:
        frame = item[0]
        inverted_frame = cv2.bitwise_not(frame)
        # Reconstruct tuple with new frame
        yield (inverted_frame,) + item[1:]