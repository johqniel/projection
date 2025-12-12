import cv2
from config import VIDEO_URL, SCROLL_SPEED
from get_stream import get_raw_stream_generator
from border_text import draw_border_layer
from complex_border import draw_border_banner

def border_test():
    print("1. Initializing Video Source... (Border Test Mode)")
    # Get raw stream from VIDEO_URL (Youtube or file)
    fps, raw_stream = get_raw_stream_generator(VIDEO_URL, loop=True)
    print("Frame rate of input video: ", fps)

    scroll_dist = 0
    
    print("Starting Border Test Loop...")
    for frame in raw_stream:
        if frame is None:
            break
            
        h,w,_ = frame.shape
        # Draw border
        draw_border_banner(frame, w, h, scroll_dist)
        
        # Update scroll
        scroll_dist += SCROLL_SPEED
        
        # Show
        cv2.imshow("Border Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    border_test()
