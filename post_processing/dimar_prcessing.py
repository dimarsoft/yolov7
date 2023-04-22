from tools.count_results import Result
from tools.labeltools import TrackWorker


def dimar_count_humans(tracks):
    print(f"Dmitrii postprocessing v1.1")

    if len(tracks) == 0:
        return Result(0, 0, 0, [])

    track_labels = TrackWorker.convert_tracks_to_list(tracks)
