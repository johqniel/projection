
# --- MODEL CONFIGURATION ---
MODEL_PATH = "detect.tflite" 
#MODEL_PATH = "yolo11n_float32.tflite"

# --- YOLO specific config ---
NMS_THRESHOLD = 0.4 # Threshold for removing overlapping boxes in YOLO


# --- VIDEO INPUT CONFIG ---
VIDEO_PATH = "/Users/nielo/Desktop/projection/samples/O.mov"
VIDEO_URL = "https://www.youtube.com/watch?v=S1A49C6V-dg" # Falls URL gewünscht



# Config for people detection
PERSON_CLASS_ID = 0
THRESHOLD = 0.7 

# Tracker Config
TRACKER_MAX_DISAPPEARED = 20
TRACKER_SMOOTHING_FACTOR = 0.3

# Traffic Light Config
TRAFFIC_LIGHT_RED_DIFF_THRESHOLD = 20 # mean_r > mean_g + THRESHOLD
TRAFFIC_LIGHT_RED_MIN_VALUE = 80      # mean_r > MIN_VALUE

# Street config
STREET_VISIBLE = False

# Visual Config
TARGET_FPS = 10
FULL_W, FULL_H = 1280, 720
FRAME_DELAY_MS = 1
CROP_W, CROP_H = FULL_W // 2, FULL_H // 2
BANNER_TEXT = "ZONE 1 ACTIVE: SURVEILLANCE RUNNING"
SCROLL_SPEED = 6
FLIP_CODE = 1 # 0: Vertical, 1: Horizontal, -1: Both

# Config for border banner
BG_ALPHA = 0.6          # Transparency (0.6 = 60% opaque)
BG_BAR_WIDTH = 80
SHIFT = 40
TEXT_COLOR = (0, 255, 255) 
FONT_SCALE = 1.2
THICKNESS = 2
MARGIN = 40  
letter_spacing = 25



