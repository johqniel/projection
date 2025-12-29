import argparse
import cv2
import sys
import os
import numpy as np

# Ensure src is in path if running from src directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Imports from existing modules
from config import MODEL_PATH, TARGET_FPS, FLIP_CODE
from stream import (
    get_raw_stream_generator, 
    enhance_light_generator, 
    detect_redlight_stream, 
    detect_people_stream, 
    draw_objects_stream,
    flip_stream_generator,
    crop_generator,
    invert_colors_generator
)
from stream.optimize_stream import invert_stream_data_generator
from setup import define_traffic_light, define_crop_area
from analysis import CentroidTracker
from plotting.drawing import draw_green_persons, draw_jaywalking_persons_striped, draw_emojis
import random

# TFLite import block
try:
    import tflite_runtime.interpreter as tflite
    print("Using TFLite Runtime")
except ImportError:
    try:
        import tensorflow.lite as tflite
        print("Using Full TensorFlow")
    except ImportError:
        print("Warning: TensorFlow not found. AI features will be disabled.")
        tflite = None

def main():
    parser = argparse.ArgumentParser(description="Process a video file with the Projection Mapping Pipeline.")
    parser.add_argument("input_path", help="Path to input video file")
    parser.add_argument("--output", help="Path to output video file (optional)", default=None)
    parser.add_argument("--mode", choices=["setup", "people_only", "random_red"], default="setup", help="Processing mode")
    args = parser.parse_args()

    input_path = args.input_path
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        return

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        name, ext = os.path.splitext(input_path)
        output_path = f"{name}_{args.mode}.mp4"

    print(f"Processing: {input_path}")
    print(f"Mode: {args.mode}")
    print(f"Output: {output_path}")

    # Dispatcher
    if args.mode == "people_only":
        process_people_only(input_path, output_path)
    elif args.mode == "random_red":
        process_random_red(input_path, output_path)
    else:
        process_full_setup(input_path, output_path)

def process_people_only(input_path, output_path):
    # 1. Init Stream
    fps, stream = get_raw_stream_generator(input_path, loop=False)
    stream = flip_stream_generator(stream, FLIP_CODE)
    stream = enhance_light_generator(stream)

    # 2. Init AI
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()
    tracker = CentroidTracker()

    # 3. Pipeline
    # Manual loop to control drawing
    writer = None
    count = 0
    
    print("Starting processing (People Only)...")

    try:
        for frame in stream:
            # Detect
            people = detect_people_stream([(frame, [], [])], interpreter, tracker) 
            # stream_data yields (frame, results, people)
            # But detect_people_stream expects iterable
            # Let's just use the direct detect_people function if possible, or wrap
            # Actually detect_people_stream logic is simple:
            # for frame, results_list in stream_data: ...
            
            # Let's call the generator properly
            # We need an adapter:
            # input to detect_people_stream is (frame, results_list)
            # where results_list is empty for this mode
            pass
            # Wait, I can't break the generator loop easily here if I wrap it.
            # Let's rewrite the loop manually.
            
            from analysis import detect_people
            objects = detect_people(frame, interpreter, tracker)
            
            # Draw
            # All green
            # objects is dict {id: box}
            green_persons = [(oid, box) for oid, box in objects.items()]
            draw_green_persons(frame, green_persons)
            
            # Write
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            writer.write(frame)
            cv2.imshow("Processing...", frame)
            if cv2.waitKey(1) == ord('q'): break
            
            count += 1
            if count % 10 == 0: sys.stdout.write(f"\rFrames: {count}"); sys.stdout.flush()
            
    except KeyboardInterrupt: pass
    finally:
        if writer: writer.release()
        cv2.destroyAllWindows()


def process_random_red(input_path, output_path):
    # 1. Init Stream
    fps, stream = get_raw_stream_generator(input_path, loop=False)
    stream = flip_stream_generator(stream, FLIP_CODE)
    stream = enhance_light_generator(stream)

    # 2. Init AI
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()
    tracker = CentroidTracker()
    
    from analysis import detect_people

    writer = None
    count = 0
    
    # State for consistent random ID
    target_id = -1
    
    print("Starting processing (Random Red)...")

    try:
        for frame in stream:
            objects = detect_people(frame, interpreter, tracker)
            
            green_p = []
            red_p = []
            
            if objects:
                start_keys = list(objects.keys())
                
                # Update target if lost
                if target_id not in objects:
                    target_id = random.choice(start_keys)
                
                for oid, box in objects.items():
                    if oid == target_id:
                        red_p.append((oid, box))
                    else:
                        green_p.append((oid, box))
            
            # Draw
            draw_green_persons(frame, green_p)
            draw_jaywalking_persons_striped(frame, red_p)
            
            # Optional: Emojis for the red person?
            # User said "one random id is red". Implies full effect?
            # drawing.py has stateful EMOJI_PARTICLES.
            # We need to manually update them.
            # We need to import the list? 
            # Actually, `draw_emojis` handles the global list update.
            # We just need to append to `EMOJI_PARTICLES` (imported).
            
            from plotting.drawing import EMOJI_PARTICLES
            for (oid, box) in red_p:
                (xmin, ymin, xmax, ymax) = box
                EMOJI_PARTICLES.append({
                    'x': float(xmin + xmax) / 2,
                    'y': float(ymin),
                    'vx': np.random.uniform(-2, 2),
                    'vy': np.random.uniform(-5, -2),
                    'size': np.random.randint(20, 40),
                    'life': 30
                })
            
            draw_emojis(frame, use_image=True)
            
            # Write
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            
            writer.write(frame)
            cv2.imshow("Processing...", frame)
            if cv2.waitKey(1) == ord('q'): break
            
            count += 1
            if count % 10 == 0: sys.stdout.write(f"\rFrames: {count}"); sys.stdout.flush()

    except KeyboardInterrupt: pass
    finally:
        if writer: writer.release()
        cv2.destroyAllWindows()


def process_full_setup(input_path, output_path):
    # This is the original generic main logic moved here
    # --- PHASE 1: SETUP ---
    print("\n--- PHASE 1: SETUP ---")
    print("Initializing setup stream...")
    # Loop=True for setup so user has time
    fps, setup_stream = get_raw_stream_generator(input_path, loop=True)
    
    # 1. Flip
    setup_stream = flip_stream_generator(setup_stream, FLIP_CODE)
    
    print("Step 1: Define Crop Area...")
    crop_rect = define_crop_area(setup_stream)
    
    setup_stream = crop_generator(setup_stream, crop_rect)
    
    print("Step 2: Define Traffic Lights & Streets...")
    tl_configs = define_traffic_light(setup_stream)
    
    cv2.destroyAllWindows()
    print("Setup Complete.")
    print(f"Defined {len(tl_configs)} zones.")

    # --- PHASE 2: PROCESSING ---
    print("\n--- PHASE 2: BATCH PROCESSING ---")
    
    # 1. Re-open stream (loop=False for processing) to start from frame 0
    fps, stream = get_raw_stream_generator(input_path, loop=False)
    
    stream = flip_stream_generator(stream, FLIP_CODE)
    stream = crop_generator(stream, crop_rect)
    stream = enhance_light_generator(stream)
    
    # Prepare Masks
    if crop_rect:
        w_crop = crop_rect[2] - crop_rect[0]
        h_crop = crop_rect[3] - crop_rect[1]
    else:
        cap = cv2.VideoCapture(input_path)
        w_full = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_full = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        w_crop = w_full
        h_crop = h_full

    street_masks = []
    for config in tl_configs:
        street_pts = np.array(config.get("street", []), np.int32)
        mask = np.zeros((h_crop, w_crop), dtype=np.uint8)
        if len(street_pts) > 0:
            cv2.fillPoly(mask, [street_pts], 255)
        street_masks.append(mask)

    # Init AI
    interpreter = tflite.Interpreter(model_path=MODEL_PATH, num_threads=4)
    interpreter.allocate_tensors()
    tracker = CentroidTracker()

    # Build Pipeline
    stream_data = detect_redlight_stream(stream, tl_configs)
    stream_data = detect_people_stream(stream_data, interpreter, tracker)
    
    # Draw
    final_stream = draw_objects_stream(stream_data, tl_configs, street_masks, use_emojis=True)
    
    # Writer Loop
    writer = None
    count = 0
    
    print("Starting processing... (Press 'q' in preview window or Ctrl+C to cancel)")
    
    try:
        for frame in final_stream:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                print(f"Output Video: {w}x{h} @ {fps}fps")
            
            writer.write(frame)
            cv2.imshow("Processing...", frame)
            if cv2.waitKey(1) == ord('q'):
                print("\nCancelled by user.")
                break
            
            count += 1
            if count % 10 == 0:
                sys.stdout.write(f"\rProcessed frames: {count}")
                sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        if writer:
            writer.release()
        cv2.destroyAllWindows()
    
    print(f"\nDone! Saved to {output_path}")

if __name__ == "__main__":
    main()
