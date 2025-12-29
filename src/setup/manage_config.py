
import json
import os
import numpy as np

CONFIG_FILE = "setup_config.json"

def save_config(crop_rect, tl_configs):
    """
    Saves the crop_rect and tl_configs to a JSON file.
    """
    data = {
        "crop_rect": crop_rect,
        "tl_configs": tl_configs
    }
    
    # helper to convert numpy types to python types for json
    def convert(o):
        if isinstance(o, np.int64): return int(o)
        if isinstance(o, np.int32): return int(o)
        raise TypeError

    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f, default=convert, indent=4)
        print(f"Setup configuration saved to {CONFIG_FILE}")
    except Exception as e:
        print(f"Failed to save configuration: {e}")

def load_config():
    """
    Loads config from JSON file.
    Returns (crop_rect, tl_configs) or (None, None) if not found/error.
    """
    if not os.path.exists(CONFIG_FILE):
        return None, None
        
    try:
        with open(CONFIG_FILE, 'r') as f:
            data = json.load(f)
            
        crop_rect = data.get("crop_rect")
        tl_configs = data.get("tl_configs", [])
        
        # Convert lists back to tuples if needed (JSON uses lists)
        if crop_rect: crop_rect = tuple(crop_rect)
        
        # Ensure lists are lists of dicts
        # (JSON handles list of dicts naturally)
        
        print(f"Loaded configuration from {CONFIG_FILE}")
        return crop_rect, tl_configs
        
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return None, None
