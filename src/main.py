# import python scripts
from get_stream import get_raw_stream_generator
from define_traffic_lights import define_traffic_light
from surveillance import run_surveillance, crop_generator

from config import VIDEO_PATH, VIDEO_URL
import config
import get_stream
import surveillance
import define_traffic_lights






# --- MAIN ---

def main_sample():
    print("1. Initializing Video Source...")
    raw_stream = get_raw_stream_generator(VIDEO_PATH, loop=True)

    print("3. Starting Setup Phase...")
    # Let the user define the refference pixels for the traffic light
    tl_config = define_traffic_light(raw_stream)

    print("4. Starting Surveillance Phase...")

    run_surveillance(raw_stream, tl_config)

def main_url():
    print("1. Initializing Video Source...")
    # Erstelle den Roh-Stream (Webcam, Datei oder URL)
    raw_stream = get_raw_stream_generator(VIDEO_URL, loop=True)
    
    print("2. Applying Cropping Pipeline...")

    raw_stream = crop_generator(raw_stream)

    print("3. Starting Setup Phase...")
    # Let the user define the refference pixels for the traffic light
    tl_config = define_traffic_light(raw_stream)

    print("4. Starting Surveillance Phase...")

    run_surveillance(raw_stream, tl_config)

if __name__ == "__main__":
    main_url()