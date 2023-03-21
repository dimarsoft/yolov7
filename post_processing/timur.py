import json
import os
from pathlib import Path

import cv2

from count_results import Result


def load_bound_line(cameras_path):
    with open(cameras_path, 'r') as f:
        bound_line = json.load(f)
    return bound_line


config = {
    "device": "cpu",
    "GOFILE": True,
    "people_id": 0,
    "model_path": os.path.join("ann_mod", "best_b4e54.pt"),
    "track_model_path": os.path.join("ann_mod", "mars-small128.pb"),
    "cameras_path": os.path.join("cfg", "camera_config.json")
}
bound_line_cameras = load_bound_line(config["cameras_path"])


def get_centrmass(p1, p2):
    res = (int((p2[0] + p1[0]) / 2), int(p2[1] + 0.35 * (p1[1] - p2[1])))
    return res


def tracks_to_dic(tracks, w, h):
    people_tracks = {}
    # [frame_index, track_id, cls, bbox_left, bbox_top, bbox_w, bbox_h, box.conf]

    for track in tracks:
        cls = track[2]
        if cls != 0:
            continue

        bbox = track[1:5]

        x1, y1, x2, y2 = int(bbox[3]), int(bbox[4]), int(bbox[3] + bbox[5]), int(bbox[4] + bbox[6])

        x1 *= w
        x2 *= w
        y1 *= h
        y2 *= h

        track_id = track[1]
        itt = track[0]

        if track_id in people_tracks.keys():
            people_tracks[track_id]["path"].append(get_centrmass((x1, y1), (x2, y2)))
            people_tracks[track_id]["frid"].append(itt)
        else:
            people_tracks.update({track_id: {
                "path": [get_centrmass((x1, y1), (x2, y2))],
                "frid": [itt]
            }})

    return people_tracks


def process_filt(people_tracks):
    max_id = max([int(idv) for idv in people_tracks.keys()])
    max_id += 1
    res = {}
    max_delt = 5  # frame
    for pk in people_tracks.keys():
        path = people_tracks[pk]["path"]
        frid = people_tracks[pk]["frid"]
        new_path = [path[0]]
        for i in range(1, len(frid)):
            if frid[i] - frid[i - 1] > max_delt and len(new_path) > 1:
                if str(pk) in res.keys():
                    new_id = str(max_id)
                    max_id += 1
                else:
                    new_id = str(pk)
                res.update({new_id: new_path})
                new_path = [path[i]]
            else:
                new_path.append(path[i])
        if len(new_path) > 1:
            if str(pk) in res.keys():
                new_id = str(max_id)
                max_id += 1
            else:
                new_id = str(pk)
            res.update({new_id: new_path})
    return res


def get_proj(p1, p2, m):
    k = (p2[0] - p1[0]) / (p2[1] - p1[1])
    f1 = m[1] + k * m[0]
    f2 = p1[1] - (1 / k) * p1[0]
    x_proj = (f1 - f2) / (k + 1 / k)
    y_proj = f1 - k * x_proj
    return [x_proj, y_proj]


def get_norm(p1, p2):
    pc = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
    xmin, xmax = pc[0] - 10, pc[0] + 10
    a = (p2[1] - p1[1]) / (p2[0] - p1[0])
    fnorm = lambda x: pc[1] - (1 / a) * (x - pc[0])
    line_norm = [[xmin, fnorm(xmin)], [xmax, fnorm(xmax)]]
    return line_norm


def crossing_bound(people_path, bound_line):
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    if len(people_path) >= 4:
        p1 = [(people_path[0][0] + people_path[1][0]) / 2, (people_path[0][1] + people_path[1][1]) / 2]
        p2 = [(people_path[-2][0] + people_path[-1][0]) / 2, (people_path[-2][1] + people_path[-1][1]) / 2]
    else:
        p1 = [people_path[0][0], people_path[0][1]]
        p2 = [people_path[-1][0], people_path[-1][1]]

    direction = "up" if p2[1] - p1[1] < 0 else "down"

    line_norm = get_norm(*bound_line)
    p1_proj = get_proj(*line_norm, p1)
    p2_proj = get_proj(*line_norm, p2)

    intersect = intersect(p1_proj, p2_proj, bound_line[0], bound_line[1])
    return {"direction": direction, "intersect": intersect}


def calc_inp_outp_people(tracks_info):
    input_p = 0
    output_p = 0
    for track_i in tracks_info:
        if track_i["intersect"]:
            if track_i["direction"] == 'down':
                input_p += 1
            elif track_i["direction"] == 'up':
                output_p += 1
    return {"input": input_p, "output": output_p}


def get_camera(source):
    input_video = cv2.VideoCapture(source)

    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    # ширина
    w = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    # высота
    h = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    input_video.release()

    num = int(Path(source).stem)

    return num, w, h


def timur_count_humans(tracks, source):

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
        tr_info = crossing_bound(people_path, bound_line)
        tracks_info.append(tr_info)
        print(f"{p_id}: {tr_info}")

    result = calc_inp_outp_people(tracks_info)

    count_in = result["input"]
    count_out = result["output"]

    deviations = []

    return Result(count_in + count_out, count_in, count_out, deviations)
