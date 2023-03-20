from deep_sort_realtime.deepsort_tracker import DeepSort


class FastDeepSort:
    def __init__(self, max_age=30):
        self.tracker = DeepSort(max_age=max_age)

    def update(self, detections, frame):
        dets = []
        for det in detections:
            bbox = det[:4]
            bbox[2] = bbox[2] - bbox[0]
            bbox[2] = bbox[3] - bbox[1]
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
