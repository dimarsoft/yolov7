import torch

from trackers.deep_sort.tracker import Tracker as DeepSortTracker
from trackers.deep_sort.tools import generate_detections as gdet
from trackers.deep_sort.detection import Detection
from trackers.deep_sort import nn_matching

import numpy as np

from utils.general import xyxy2xywh


def _xyxy_to_tlwh(bbox_xyxy):
    x1, y1, x2, y2 = bbox_xyxy

    top = x1
    left = y1
    w = int(x2 - x1)
    h = int(y2 - y1)
    return top, left, w, h


class DeepSort:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self, encoder_model_filename, max_dist=0.4, nn_budget=100, max_age=30, n_init=3):

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
        self.tracker = DeepSortTracker(metric, max_age=max_age, n_init=n_init)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, detections, frame):

        confs = detections[:, 4]
        clss = detections[:, 5]
        xyxys = detections[:, 0:4]

        xyxys_numpy = xyxys.numpy()

        xywhs = xyxy2xywh(xyxys_numpy)

        bbox_tlwh = self._xywh_to_tlwh(xywhs)

        bboxes = np.array([d for d in bbox_tlwh])

        features = self.encoder(frame, bboxes)

        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confs)]

        self.tracker.predict()
        self.tracker.update(detections, clss, confs)
        self.update_tracks()

        # сразу вернем результат

        result = []
        for track in self.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            track_id = track.track_id
            result.append([x1, y1, x2, y2, track_id, track.class_id, track.conf])
        return result

    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            track_id = track.track_id

            tracks.append(Track(track_id, bbox, track.class_id, track.conf))

        self.tracks = tracks


class Track:
    track_id = None
    bbox = None

    def __init__(self, track_id, bbox, class_id, conf):
        self.track_id = track_id
        self.bbox = bbox
        self.class_id = class_id
        self.conf = conf
