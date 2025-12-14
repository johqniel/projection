import cv2
from plotting import draw_objects, draw_objects_emojis, draw_objects_image_emojis

def draw_objects_stream(stream_data, config, street_mask, use_emojis=True):
    """
    Generator that draws objects on the frame.
    Input: Stream of (frame, status, status_color, people)
    Yields: frame (modified)
    """
    for frame, status, status_color, people in stream_data:
        if use_emojis:
            frame = draw_objects_image_emojis(status, status_color, people, config, frame, street_mask)
        else:
            frame = draw_objects(status, status_color, people, config, frame, street_mask)
        yield frame

def show_stream(stream, window_name="Modular Stream"):
    """
    Consumes the stream and shows it on screen.
    Breaks on 'q'.
    """
    for frame in stream:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()
