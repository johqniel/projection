from analysis import detect_red_light, detect_people

def detect_redlight_stream(stream, config):
    """
    Generator that analyzes traffic lights.
    Yields: (frame, status, status_color)
    """
    for frame in stream:
        status, status_color = detect_red_light(frame, config)
        yield frame, status, status_color

def detect_people_stream(stream_data, interpreter, tracker):
    """
    Generator that analyzes people.
    Input: Stream of (frame, status, status_color)
    Yields: (frame, status, status_color, people)
    """
    for frame, status, status_color in stream_data:
        people = detect_people(frame, interpreter, tracker)
        yield frame, status, status_color, people
