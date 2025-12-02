# import python scripts
from get_stream import get_raw_stream_generator
from setup import define_traffic_light
from surveillance import run_surveillance, crop_generator, reduce_framerate

from config import VIDEO_PATH, VIDEO_URL
import config
import get_stream
import surveillance
import setup






# --- MAIN ---

def main_sample():
    print("1. Initializing Video Source...")
    fps, raw_stream = get_raw_stream_generator(VIDEO_PATH, loop=True)

    print("Frame rate of input video: ", fps)

    print("2. Define Crop Area...")
    # Let the user define the crop area
    crop_rect = setup.define_crop_area(raw_stream)
    
    # Apply cropping (and resizing)
    cropped_stream = crop_generator(raw_stream, crop_rect)

    print("3. Starting Setup Phase...")
    # Let the user define the refference pixels for the traffic light
    tl_config = define_traffic_light(cropped_stream)

    print("4. Starting Surveillance Phase...")
    
    reduced_stream = reduce_framerate(cropped_stream, fps)
    run_surveillance(reduced_stream, tl_config)

def main_url():
    print("1. Initializing Video Source...")
    # Erstelle den Roh-Stream (Webcam, Datei oder URL)
    fps, raw_stream = get_raw_stream_generator(VIDEO_URL, loop=True)
    
    print("Frame rate of input video: ", fps)


    print("2. Define Crop Area ...")

    crop_rect = setup.define_crop_area(raw_stream)

    # Apply cropping (and resizing)
    cropped_stream = crop_generator(raw_stream, crop_rect)

    print("3. Starting Setup Phase...")
    # Let the user define the refference pixels for the traffic light
    tl_config = define_traffic_light(cropped_stream)

    print("4. Starting Surveillance Phase...")

    reduced_stream = reduce_framerate(cropped_stream, fps)

    run_surveillance(reduced_stream, tl_config)

if __name__ == "__main__":
    main_url()