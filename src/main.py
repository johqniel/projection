# import python scripts
from get_stream import get_raw_stream_generator
from setup import define_traffic_light
from surveillance import run_surveillance, crop_generator, reduce_framerate, flip_stream_generator
from config import VIDEO_PATH, VIDEO_URL, FLIP_CODE, TARGET_FPS
import config
import get_stream
import surveillance
import setup







# --- MAIN ---

def main_camera():
    print("1. Initializing Camera Source...")
    # Use device index 0 (or 1 if 0 is built-in webcam)
    # The user mentioned HDMI capture card, which is often 0 or 1.
    try:
        fps, raw_stream = get_stream.get_camera_stream_generator(0)
    except ValueError:
         print("Camera 0 failed, trying Camera 1...")
         fps, raw_stream = get_stream.get_camera_stream_generator(1)

    print(f"Camera FPS: {fps}")

    stream = reduce_framerate(raw_stream, fps, TARGET_FPS)

    print("2. Define Crop Area...")
    crop_rect = setup.define_crop_area(stream)
    
    stream = crop_generator(stream, crop_rect)

    stream = flip_stream_generator(stream, FLIP_CODE)

    print("3. Starting Setup Phase...")
    tl_config = define_traffic_light(stream)

    print("4. Starting Surveillance Phase...")
    
    run_surveillance(stream, tl_config)

def main_sample():
    print("1. Initializing Video Source...")
    fps, raw_stream = get_raw_stream_generator(VIDEO_PATH, loop=True)

    print("Frame rate of input video: ", fps)

    stream = reduce_framerate(raw_stream, fps, TARGET_FPS)

    print("2. Define Crop Area...")
    # Let the user define the crop area
    crop_rect = setup.define_crop_area(stream)
    
    # Apply cropping (and resizing)
    stream = crop_generator(stream, crop_rect)
    stream = flip_stream_generator(stream, FLIP_CODE)

    print("3. Starting Setup Phase...")
    # Let the user define the refference pixels for the traffic light
    tl_config = define_traffic_light(stream)

    print("4. Starting Surveillance Phase...")
    
    run_surveillance(stream, tl_config)

def main_url():
    print("1. Initializing Video Source...")
    # Erstelle den Roh-Stream (Webcam, Datei oder URL)
    fps, stream = get_raw_stream_generator(VIDEO_URL, loop=True)
    
    print("Frame rate of input video: ", fps)

    stream = reduce_framerate(stream, fps)


    print("2. Define Crop Area ...")

    crop_rect = setup.define_crop_area(stream)

    # Apply cropping (and resizing)
    # Apply cropping (and resizing)
    stream = crop_generator(stream, crop_rect)
    stream = flip_stream_generator(stream, FLIP_CODE)

    print("3. Starting Setup Phase...")
    # Let the user define the refference pixels for the traffic light
    tl_config = define_traffic_light(stream)

    print("4. Starting Surveillance Phase...")

    run_surveillance(stream, tl_config)


if __name__ == "__main__":
    main_camera()