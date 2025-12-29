import cv2
from plotting import draw_objects

def draw_objects_stream(stream_data, config, street_mask, use_emojis=True):
    """
    Generator that draws objects on the frame.
    Input: Stream of (frame, results_list, people)
    Yields: frame (modified)
    """
    for frame, results_list, people in stream_data:
        # results_list = [(status, color), (status, color)...]
        # We now use the unified orchestrator which handles the emoji flag
        frame, is_jaywalking = draw_objects(results_list, people, config, frame, street_mask, use_emojis=use_emojis)
        yield frame, is_jaywalking

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
