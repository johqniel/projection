
# --- MODEL CONFIGURATION ---
MODEL_PATH = "detect.tflite" 
#MODEL_PATH = "yolo11n_float32.tflite"
MODEL_PATH = "yolov5s_f16.tflite"

# --- YOLO specific config ---
NMS_THRESHOLD = 0.4 # Threshold for removing overlapping boxes in YOLO


# --- VIDEO INPUT CONFIG ---
VIDEO_PATH = "/Users/nielo/Desktop/projection/samples/M.mov"
VIDEO_URL = "https://www.youtube.com/watch?v=S1A49C6V-dg" # Falls URL gewünscht



# Config for people detection
PERSON_CLASS_ID = 0
THRESHOLD = 0.7 

# Tracker Config
TRACKER_MAX_DISAPPEARED = 20
TRACKER_SMOOTHING_FACTOR = 0.5

# Traffic Light Config
TRAFFIC_LIGHT_RED_DIFF_THRESHOLD = 20 # mean_r > mean_g + THRESHOLD
TRAFFIC_LIGHT_RED_MIN_VALUE = 80      # mean_r > MIN_VALUE

# Street config
STREET_VISIBLE = False

# Visual Config
TARGET_FPS = 15
FULL_W, FULL_H = 1280, 720
FRAME_DELAY_MS = 1
CROP_W, CROP_H = FULL_W // 2, FULL_H // 2
BANNER_TEXT = "Achtung: KI-Erkennung von Rotlichtverstößen aktiv"
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
LETTER_SPACING = int(FONT_SCALE * 20)


# Config for gamma filter
GAMMA = 1.5

CLIP_LIMIT = 4.0




