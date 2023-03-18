from trackers.deep_sort.tracker import Tracker as DeepSortTracker
from trackers.deep_sort.tools import generate_detections as gdet
from trackers.deep_sort.detection import Detection
from trackers.deep_sort import nn_matching

import numpy as np


class DeepSort:
    tracker = None
    encoder = None
    tracks = None

    def __init__(self, encoder_model_filename, max_dist=0.4, nn_budget=100, max_age=30, n_init=3):

        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_dist, nn_budget)
        self.tracker = DeepSortTracker(metric, max_age=max_age, n_init=n_init)
        self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)

    def update(self, detections, frame):

        if len(detections) == 0:
            return []
        # тут нужна конвертация

        my_detection = []

        confs = detections[:, 4]
        clss = detections[:, 5]

        for obj_i in detections:
            x1, y1, x2, y2 = int(obj_i[0]), int(obj_i[1]), int(obj_i[2]), int(obj_i[3])
            score, class_id = obj_i[4], int(obj_i[5])
            my_detection.append([x1, y1, x2, y2, score])

        detections = my_detection

        bboxes = np.asarray([d[:-1] for d in detections])
        bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
        scores = [d[-1] for d in detections]

        features = self.encoder(frame, bboxes)

        dets = []
        for bbox_id, bbox in enumerate(bboxes):
            dets.append(Detection(bbox, scores[bbox_id], features[bbox_id]))

        self.tracker.predict()
        self.tracker.update(dets, clss, confs)
        self.update_tracks()

        # сразу вернем результат

        result = []
        for track in self.tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            track_id = track.track_id
            result.append([x1, y1, x2, y2, track_id, track.class_id, track.conf])
        return result

    def update_tracks(self):
        tracks = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            id = track.track_id

            tracks.append(Track(id, bbox, track.class_id, track.conf))

        self.tracks = tracks


class Track:
    track_id = None
    bbox = None

    def __init__(self, id, bbox, class_id, conf):
        self.track_id = id
        self.bbox = bbox
        self.class_id = class_id
        self.conf = conf

# from deep_sort.deep_sort.tracker import Tracker as DeepSortTracker
# from deep_sort.tools import generate_detections as gdet
# from deep_sort.deep_sort import nn_matching
# from deep_sort.deep_sort.detection import Detection
# import numpy as np
#
# class Tracker:
#     tracker = None
#     encoder = None
#     tracks = None
#
#     def __init__(self):
#         max_cosine_distance = 0.4
#         nn_budget = None
#
#         encoder_model_filename = ""
#
#         metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
#         self.tracker = DeepSortTracker(metric)
#         self.encoder = gdet.create_box_encoder(encoder_model_filename, batch_size=1)
#
#     def update(self, frame, detections):
#         bboxes = np.asarray([d[:-1] for d in detections])
#         bboxes[:, 2:] = bboxes[:, 2:] - bboxes[:, 0:2]
#         scores = [d[-1] for d in detections]
#
#         features = self.encoder(frame, bboxes)
#
#         dats = []
#         for bbox_id, bbox in enumerate(bboxes):
#             dats.append(Detection(bbox, scores[bbox_id], features[bbox_id]))
#
#         self.tracker.predict()
#         self.tracker.update(dats)
#         self.update_tracks()
#
#     def update_tracks(self):
#         tracks = []
#         for track in self.tracker.tracks:
#             if not track.is_confirmed() or track.time_since_update > 1:
#                 continue
#             bbox = track.to_tlbr()
#             id = track.track_id
#             tracks.append(Track(id, bbox))
#         self.tracks = tracks
#
# class Track:
#     track_id = None
#     bbox = None
#     def __init__(self, id, bbox):
#         self.track_id = id
#         self.bbox = bbox
