
import yt_dlp 
import cv2

from config import FULL_H, FULL_W, CROP_H, CROP_W


# --- GENERATORS (STREAMS) ---

def get_raw_stream_generator(source_path, loop=True):
    """
    Returns (fps, generator).
    Generator yields frames from file or URL. 
    If loop=True, restarts video when it ends (good for setup).
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

