
import cv2
from vidgear.gears import CamGear

from config import FULL_H, FULL_W, CROP_H, CROP_W


# --- GENERATORS (STREAMS) ---
"""
Module: get_stream.py

This module handles the extraction of video streams from files or URLs.
It provides functions to create generators that yield frames from video sources.

Key Functions:
- `get_raw_stream_generator(...)`: Creates a generator for a video stream.
- `get_camera_stream_generator(...)`: Creates a generator for a camera feed.

Dependencies:
- `config.py`: specific constants (e.g., FULL_H, FULL_W, CROP_H, CROP_W).
- `cv2` (OpenCV): for video capture and frame processing.
- `vidgear`: for threaded, high-performance streaming.
"""

def get_raw_stream_generator(source_path, loop=True):
    """
    Creates a generator that yields frames from a file or URL using OpenCV or VidGear.

    If the source path starts with "http", it is treated as a URL and processed
    via vidgear.gears.CamGear for high-performance threaded streaming.

    Args:
        source_path (str | int): The path to the video file or the URL of the stream.
        loop (bool, optional): Whether to restart the video when it ends. Defaults to True.
                               Note: Livestreams (URLs) will forcefully set this to False.

    Returns:
        tuple: A tuple containing (fps (float), generator (typing.Generator)).
               The generator yields frames (np.ndarray).
    """
    fps = 30.0 # Default value, will be updated if possible

    # 1. Handle URL with VidGear
    if str(source_path).startswith("http"):
        print(f"⏳ Initializing VidGear stream for URL: {source_path}")
        
        # STREAM_RESOLUTION ensures we get best quality (or specified), threaded.
        # logging=True helps debug if there are issues.
        options = {"STREAM_RESOLUTION": "best", "CAP_PROP_FRAME_WIDTH":1920, "CAP_PROP_FRAME_HEIGHT":1080} 
        
        try:
            stream = CamGear(source=source_path, stream_mode=True, logging=True, **options).start()
            fps = stream.framerate if stream.framerate > 0 else 30.0
            loop = False # Livestreams don't loop
            
            print(f"VidGear Stream Started. FPS: {fps}")
            
            def generator():
                 while True:
                    frame = stream.read()
                    if frame is None:
                        print("VidGear stream ended or failed to read frame.")
                        break
                    yield frame
                 stream.stop()
                 
            return fps, generator()

        except Exception as e:
            print(f"VidGear failed: {e}. Falling back to standard method...")
            # Fallthrough to standard CV2 if vidgear fails for some reason (unlikely for youtube)
            pass

    # 2. Standard OpenCV (File or Fallback)
    cap = cv2.VideoCapture(source_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Input FPS: {fps}")

    def generator():
        frames_read = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                if loop:
                    if frames_read == 0:
                         print("Error: Could not read any frames from source, even though opened successfully.")
                         break
                    
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break # End of stream
            
            frames_read += 1
            yield frame
        cap.release()
    
    return fps, generator()


def get_camera_stream_generator(device_index=0):
    """
    Creates a generator that yields frames from a live camera feed.

    Args:
        device_index (int, optional): The camera device index. Defaults to 0.

    Returns:
        tuple: A tuple containing (fps (float), generator (typing.Generator)).
               The generator yields frames (np.ndarray).
    """
    print(f"⏳ Connecting to Camera (Index: {device_index})...")
    cap = cv2.VideoCapture(device_index)
    
    # Optional: Try to set high resolution (e.g. 1920x1080)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        raise ValueError(f"Could not open camera device: {device_index}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    # Some webcams return 0 or incorrect FPS, default to 30 if so
    if fps <= 0:
        fps = 30.0
    
    print(f"Camera FPS: {fps}")

    def generator():
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera stream ended or failed.")
                break 
            yield frame
        cap.release()
    
    return fps, generator()

