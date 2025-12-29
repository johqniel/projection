import sys
import os

print("Python Executable:", sys.executable)
try:
    import mediapipe
    print("\nmediapipe imported successfully.")
    print("Location:", mediapipe.__file__)
    print("Dir:", dir(mediapipe))
    
    if hasattr(mediapipe, 'solutions'):
        print("mediapipe.solutions exists.")
    else:
        print("mediapipe.solutions matches nothing in dir()")
        
    try:
        from mediapipe import solutions
        print("Successfully did: 'from mediapipe import solutions'")
    except ImportError as e:
        print("Failed 'from mediapipe import solutions':", e)

except ImportError as e:
    print("Failed to import mediapipe:", e)
