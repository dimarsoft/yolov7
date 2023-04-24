import json
import os
from pathlib import Path

import cv2

from configs import CAMERAS_PATH, load_bound_line
from tools.count_results import Result, Deviation
from tools.labeltools import get_status
from post_processing.functions import crossing_bound, process_filt, get_centrmass, get_deviations


def save_bound_line(cameras_path, bound_line):
    with open(cameras_path, 'w') as f:
        json.dump(bound_line, fp=f, indent=4)


bound_line_cameras = load_bound_line(CAMERAS_PATH)


# print(bound_line_cameras)


def _get_centrmass(p1, p2):
    res = (int((p2[0] + p1[0]) / 2), int(p2[1] + 0.35 * (p1[1] - p2[1])))
    return res


def update_dict(dict_tracks, track_id, x1, y1, x2, y2, itt):
    if track_id in dict_tracks.keys():
        dict_tracks[track_id]["path"].append(get_centrmass((x1, y1), (x2, y2)))
        dict_tracks[track_id]["frid"].append(itt)
        dict_tracks[track_id]["bbox"].append([(x1, y1), (x2, y2)])
    else:
        dict_tracks.update({track_id: {
            "path": [get_centrmass((x1, y1), (x2, y2))],
            "bbox": [[(x1, y1), (x2, y2)]],
            "frid": [itt]
        }})


def tracks_to_dic(tracks, w, h):
    people_tracks = {}
    helmet_tracks = {}
    vest_tracks = {}

    by_class = {0: people_tracks, 1: helmet_tracks, 2: vest_tracks}

    # [frame_index, track_id, cls, bbox_left, bbox_top, bbox_w, bbox_h, box.conf]

    for track in tracks:
        cls = int(track[2])
        # if cls != 0:             continue

        dict_track = by_class[cls]

        x1, y1, x2, y2 = track[3], track[4], track[3] + track[5], track[4] + track[6]

        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h

        track_id = track[1]
        itt = track[0]

        update_dict(dict_track, track_id, x1, y1, x2, y2, itt)

    return people_tracks, helmet_tracks, vest_tracks


def get_camera(source):
    input_video = cv2.VideoCapture(source)

    # ширина
    w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # высота
    h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = int(input_video.get(cv2.CAP_PROP_FPS))

    input_video.release()

    num = Path(source).stem

    return num, w, h, fps


def convert_and_save(folder_path):
    folder_path = Path(folder_path)
    bl = {}
    for i in bound_line_cameras.keys():
        video_source = folder_path / f"{i}.mp4"

        camera_num, w, h, fps = get_camera(str(video_source))

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


def timur_count_humans(tracks, source, bound_line, log: bool = True) -> Result:
    print(f"Timur postprocessing v1.7_24.04.2023")

    camera_num, w, h, fps = get_camera(source)

    if log:
        print(f"camera_num =  {camera_num}, ({w} {h})")

    people_tracks, helmet_tracks, vest_tracks = tracks_to_dic(tracks, w, h)

    if len(people_tracks) == 0:
        return Result(0, 0, 0, [])

    people_tracks = process_filt(people_tracks)

    if len(people_tracks) == 0:
        return Result(0, 0, 0, [])

    # bound_line = bound_line_cameras.get(camera_num)

    if log:
        print(f"bound_line =  {bound_line}")

    tracks_info = []
    for p_id in people_tracks.keys():
        people_path = people_tracks[p_id]
        tr_info = crossing_bound(people_path['path'], bound_line)
        tracks_info.append(tr_info)
        if log:
            print(f"{p_id}: {tr_info}")

    deviations = []

    deviations_info, result = get_deviations(people_tracks, helmet_tracks, vest_tracks, bound_line, log=log)

    count_in = result["input"]
    count_out = result["output"]

    # print(deviations_info)

    for item in deviations_info:
        frame_id = item["frame_id"]

        start_frame = item["start_frame"]
        end_frame = item["end_frame"]

        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame

        # -+ 1 сек от пересечения, но не забегая за границы человека по треку
        start_frame = max(frame_id - 2*fps, start_frame)
        end_frame = min(frame_id + 2*fps, end_frame)

        if start_frame <= frame_id <= end_frame:
            pass
        else:
            # для проверки
            print(f"bad: {start_frame}, {frame_id}, {end_frame}")

        status = get_status(item["has_helmet"], item["has_uniform"])

        if status > 0:  # 0 нет нарушения
            deviations.append(Deviation(start_frame, end_frame, status))

    if log:
        print(f"{camera_num}: count_in = {count_in}, count_out = {count_out}, deviations = {len(deviations)}")

    return Result(count_in + count_out, count_in, count_out, deviations)
