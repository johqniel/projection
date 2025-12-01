
# --- MODEL CONFIGURATION ---
MODEL_PATH = "detect.tflite" 
#MODEL_PATH = "yolo11n_float32.tflite"

# --- YOLO specific config ---
NMS_THRESHOLD = 0.4 # Threshold for removing overlapping boxes in YOLO


# --- VIDEO INPUT CONFIG ---
VIDEO_PATH = "/Users/nielo/Desktop/projection/samples/G.mp4"
VIDEO_URL = "https://www.youtube.com/watch?v=6dp-bvQ7RWo" # Falls URL gewünscht



# COnfig for people detection
PERSON_CLASS_ID = 0
THRESHOLD = 0.7 


# Visual Config
FULL_W, FULL_H = 1280, 720
FRAME_DELAY_MS = 1
CROP_W, CROP_H = FULL_W // 2, FULL_H // 2
BANNER_TEXT = "ZONE 1 ACTIVE: SURVEILLANCE RUNNING"
SCROLL_SPEED = 6

# Config for border banner
BG_ALPHA = 0.6          # Transparency (0.6 = 60% opaque)
BG_BAR_WIDTH = 80
SHIFT = 40
TEXT_COLOR = (0, 255, 255) 
FONT_SCALE = 1.2
THICKNESS = 2
MARGIN = 40  
letter_spacing = 25



