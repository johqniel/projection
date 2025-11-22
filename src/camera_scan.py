import cv2

def test_cameras():
    print("Scanning for cameras... (This may take a few seconds)")
    
    # Scan the first 5 possible indexes
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Success: Camera found at Index {index}")
                print(f"   Resolution: {frame.shape[1]}x{frame.shape[0]}")
                
                # Show a snapshot
                cv2.imshow(f"Camera Index {index}", frame)
                cv2.waitKey(1000) # Show for 1 second then close
                cv2.destroyAllWindows()
            else:
                print(f"Error: Camera Index {index} opened, but returned no image (Black screen).")
            cap.release()
        else:
            print(f"   No camera at Index {index}")

if __name__ == "__main__":
    test_cameras()