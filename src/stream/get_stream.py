
import yt_dlp 
import cv2

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
- `yt_dlp`: for downloading video streams from URLs.
"""

def get_raw_stream_generator(source_path, loop=True):
    """
    Creates a generator that yields frames from a file or URL using OpenCV.

    If the source path starts with "http", it is treated as a URL and processed
    via yt_dlp to extract the best video stream URL.

    Args:
        source_path (str | int): The path to the video file or the URL of the stream.
        loop (bool, optional): Whether to restart the video when it ends. Defaults to True.
                               Note: Livestreams (URLs) will forcefully set this to False.

    Returns:
        tuple: A tuple containing (fps (float), generator (typing.Generator)).
               The generator yields frames (np.ndarray).
    """
    # 1. Handle URL
    if str(source_path).startswith("http"):
        print(f"⏳ Extracting URL: {source_path}")
        ydl_opts = {'format': 'bestvideo/best', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source_path, download=False)
            source_path = info['url']
        loop = False # Livestreams don't loop
        cap = cv2.VideoCapture(source_path)

    else:
        cap = cv2.VideoCapture(source_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Input FPS: {fps}")

    def generator():
        while True:
            ret, frame = cap.read()
            if not ret:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break # End of stream
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

