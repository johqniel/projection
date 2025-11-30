
import yt_dlp 
import cv2


# --- GENERATORS (STREAMS) ---

def get_raw_stream_generator(source_path, loop=True):
    """
    Yields frames from file or URL. 
    If loop=True, restarts video when it ends (good for setup).
    """
    # 1. Handle URL
    if str(source_path).startswith("http"):
        print(f"⏳ Extracting URL: {source_path}")
        ydl_opts = {'format': 'best[ext=mp4]/best', 'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source_path, download=False)
            source_path = info['url']
        loop = False # Livestreams don't loop
        cap = cv2.VideoCapture(source_path)

    else:
        cap = cv2.VideoCapture(source_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source_path}")

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

def crop_generator(input_stream):
    """
    Consumes a stream, crops it, and yields cropped frames.
    """
    # Define Crop Coordinates (Zone 1)
    # Logic: Right half of the image
    A_H, B_H = 0, CROP_H
    A_V, B_V = int(FULL_W - 1.5*CROP_H), FULL_W

    for frame in input_stream:
        # Resize to standard if needed
        frame = cv2.resize(frame, (FULL_W, FULL_H))
        
        # Crop
        cropped = frame[A_H:B_H, A_V:B_V]
        
        yield cropped

