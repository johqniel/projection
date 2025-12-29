import mediapipe
print("Dir(mediapipe):", dir(mediapipe))
try:
    print("mediapipe.solutions:", mediapipe.solutions)
except AttributeError as e:
    print("Error accessing solutions:", e)

try:
    import mediapipe.python.solutions as solutions
    print("Imported mediapipe.python.solutions successfully")
except ImportError as e:
    print("Error importing mediapipe.python.solutions:", e)
