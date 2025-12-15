from analysis import detect_red_light, detect_people

def detect_redlight_stream(stream, config_list):
    """
    Generator that analyzes traffic lights.
    Yields: (frame, results_list)
    results_list: [(status, status_color), ...] one for each config
    """
    for frame in stream:
        results = []
        for config in config_list:
            status, status_color = detect_red_light(frame, config)
            results.append((status, status_color))
        yield frame, results

def detect_people_stream(stream_data, interpreter, tracker):
    """
    Generator that analyzes people.
    Input: Stream of (frame, results_list)
    Yields: (frame, results_list, people)
    """
    for frame, results_list in stream_data:
        people = detect_people(frame, interpreter, tracker)
        yield frame, results_list, people
