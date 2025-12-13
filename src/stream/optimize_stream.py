import cv2
import time

from config import FULL_W, FULL_H

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

