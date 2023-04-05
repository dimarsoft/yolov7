from deep_sort_realtime.deepsort_tracker import DeepSort


class FastDeepSortTracker:
    def __init__(
            self,
            max_iou_distance=0.7,
            max_age=30,
            n_init=3,
            nms_max_overlap=1.0,
            max_cosine_distance=0.2,
            nn_budget=None,
            embedder="mobilenet"
    ):
        self.tracker = DeepSort(max_iou_distance=max_iou_distance,
                                max_age=max_age,
                                n_init=n_init,
                                nms_max_overlap=nms_max_overlap,
                                max_cosine_distance=max_cosine_distance,
                                nn_budget=nn_budget,
                                embedder=embedder)

    def update(self, detections, frame):
        dets = []
        for det in detections:
            bbox = det[:4].clone()
            bbox[2] = bbox[2] - bbox[0]
            bbox[3] = bbox[3] - bbox[1]
            dets.append((bbox, det[4], det[5]))

        # bbs expected to be a list of detections, each in tuples of ( [left,top,w,h], confidence, detection_class)
        tracks = self.tracker.update_tracks(dets, frame=frame)

        results = []

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            conf = track.get_det_conf()
            class_id = track.get_det_class()

            results.append([ltrb[0], ltrb[1], ltrb[2], ltrb[3], track_id, class_id, conf])

        return results
