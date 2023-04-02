import json
import os
from pathlib import Path

import cv2

from count_results import Result
from post_processing.functions import crossing_bound, calc_inp_outp_people, process_filt, get_centrmass


def load_bound_line(cameras_path):
    with open(cameras_path, 'r') as f:
        bound_line = json.load(f)
    return bound_line


def save_bound_line(cameras_path, bound_line):
    with open(cameras_path, 'w') as f:
        json.dump(bound_line, fp=f, indent=4)


config = {
    "device": "cpu",
    "GOFILE": True,
    "people_id": 0,
    "model_path": os.path.join("ann_mod", "best_b4e54.pt"),
    "track_model_path": os.path.join("ann_mod", "mars-small128.pb"),
    "cameras_path": os.path.join("cfg", "camera_config.json")
}
bound_line_cameras = load_bound_line(config["cameras_path"])

# print(bound_line_cameras)


def _get_centrmass(p1, p2):
    res = (int((p2[0] + p1[0]) / 2), int(p2[1] + 0.35 * (p1[1] - p2[1])))
    return res


def tracks_to_dic(tracks, w, h):
    people_tracks = {}
    # [frame_index, track_id, cls, bbox_left, bbox_top, bbox_w, bbox_h, box.conf]

    for track in tracks:
        cls = track[2]
        if cls != 0:
            continue

        x1, y1, x2, y2 = track[3], track[4], track[3] + track[5], track[4] + track[6]

        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h

        track_id = track[1]
        itt = track[0]

        if track_id in people_tracks.keys():
            people_tracks[track_id]["path"].append(get_centrmass((x1, y1), (x2, y2)))
            people_tracks[track_id]["frid"].append(itt)
            people_tracks[track_id]["bbox"].append([(x1, y1), (x2, y2)])
        else:
            people_tracks.update({track_id: {
                "path": [get_centrmass((x1, y1), (x2, y2))],
                "bbox": [[(x1, y1), (x2, y2)]],
                "frid": [itt]
            }})

    return people_tracks


def get_camera(source):
    input_video = cv2.VideoCapture(source)

    # ширина
    w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # высота
    h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    input_video.release()

    num = Path(source).stem

    return num, w, h


def convert_and_save(folder_path):
    folder_path = Path(folder_path)
    bl = {}
    for i in bound_line_cameras.keys():
        video_source = folder_path / f"{i}.mp4"

        camera_num, w, h = get_camera(str(video_source))

        item = bound_line_cameras[i]
        p1 = item[0]
        p2 = item[1]
        p1[0] = p1[0] / w
        p1[1] = p1[1] / h

        p2[0] = p2[0] / w
        p2[1] = p2[1] / h

        bl[i] = [p1, p2]

    file_to_save = os.path.join("cfg", "camera_config_v2.json")
    save_bound_line(file_to_save, bl)


def timur_count_humans(tracks, source):
    print(f"Timur postprocessing v1.3_02.04.2023")

    camera_num, w, h = get_camera(source)

    print(f"camera_num =  {camera_num}, ({w} {h})")

    people_tracks = tracks_to_dic(tracks, w, h)

    if len(people_tracks) == 0:
        return Result(0, 0, 0, [])

    people_tracks = process_filt(people_tracks)
    bound_line = bound_line_cameras.get(camera_num)

    print(f"bound_line =  {bound_line}")

    tracks_info = []
    for p_id in people_tracks.keys():
        people_path = people_tracks[p_id]
        tr_info = crossing_bound(people_path['path'], bound_line)
        tracks_info.append(tr_info)
        print(f"{p_id}: {tr_info}")

    result = calc_inp_outp_people(tracks_info)

    count_in = result["input"]
    count_out = result["output"]

    deviations = []

    print(f"count_in = {count_in}, count_out = {count_out}")

    return Result(count_in + count_out, count_in, count_out, deviations)
