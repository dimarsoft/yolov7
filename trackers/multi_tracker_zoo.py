from pathlib import Path

from configs import WEIGHTS
from trackers.strongsort.utils.parser import get_config


def create_tracker(tracker_type, tracker_config, reid_weights, device, half):
    cfg = get_config()
    cfg.merge_from_file(tracker_config)

    if tracker_type == 'strongsort':
        from trackers.strongsort.strong_sort import StrongSORT
        strongsort = StrongSORT(
            reid_weights,
            device,
            half,
            max_dist=cfg.strongsort.max_dist,
            max_iou_dist=cfg.strongsort.max_iou_dist,
            max_age=cfg.strongsort.max_age,
            max_unmatched_preds=cfg.strongsort.max_unmatched_preds,
            n_init=cfg.strongsort.n_init,
            nn_budget=cfg.strongsort.nn_budget,
            mc_lambda=cfg.strongsort.mc_lambda,
            ema_alpha=cfg.strongsort.ema_alpha,
        )
        return strongsort

    elif tracker_type == 'ocsort':
        from trackers.ocsort.ocsort import OCSort
        ocsort = OCSort(
            det_thresh=cfg.ocsort.det_thresh,
            max_age=cfg.ocsort.max_age,
            min_hits=cfg.ocsort.min_hits,
            iou_threshold=cfg.ocsort.iou_thresh,
            delta_t=cfg.ocsort.delta_t,
            asso_func=cfg.ocsort.asso_func,
            inertia=cfg.ocsort.inertia,
            use_byte=cfg.ocsort.use_byte,
        )
        return ocsort

    elif tracker_type == 'bytetrack':
        from trackers.bytetrack.byte_tracker import BYTETracker
        bytetracker = BYTETracker(
            track_thresh=cfg.bytetrack.track_thresh,
            match_thresh=cfg.bytetrack.match_thresh,
            track_buffer=cfg.bytetrack.track_buffer,
            frame_rate=cfg.bytetrack.frame_rate
        )
        return bytetracker

    elif tracker_type == 'botsort':
        from trackers.botsort.bot_sort import BoTSORT
        botsort = BoTSORT(
            reid_weights,
            device,
            half,
            track_high_thresh=cfg.botsort.track_high_thresh,
            new_track_thresh=cfg.botsort.new_track_thresh,
            track_buffer=cfg.botsort.track_buffer,
            match_thresh=cfg.botsort.match_thresh,
            proximity_thresh=cfg.botsort.proximity_thresh,
            appearance_thresh=cfg.botsort.appearance_thresh,
            cmc_method=cfg.botsort.cmc_method,
            frame_rate=cfg.botsort.frame_rate,
            lambda_=cfg.botsort.lambda_
        )
        return botsort
    elif tracker_type == 'sort':
        from trackers.sort.sort import Sort
        sort = Sort(max_age=cfg.sort.max_age, min_hits=cfg.sort.min_hits)
        return sort
    elif tracker_type == 'deepsort':
        from trackers.deep_sort.deepsort import DeepSort as DeepSort
        reid_weights = Path(WEIGHTS) / "mars-small128.pb"
        print(f"deepsort reid_weights ' {reid_weights}'")
        tracker = DeepSort(encoder_model_filename=reid_weights,
                           max_dist=cfg.deepsort.max_dist,
                           max_age=cfg.deepsort.max_age,
                           n_init=cfg.deepsort.n_init,
                           nn_budget=cfg.deepsort.nn_budget)
        return tracker
    elif tracker_type == 'fastdeepsort':
        from trackers.fast_deep_sort.fast_deepsort_tracker import FastDeepSortTracker
        tracker = FastDeepSortTracker(
            max_age=cfg.fastdeepsort.max_age,
            n_init=cfg.fastdeepsort.n_init,
            nms_max_overlap=cfg.fastdeepsort.nms_max_overlap,
            nn_budget=cfg.fastdeepsort.nn_budget,
            max_iou_distance=cfg.fastdeepsort.max_iou_distance,
            max_cosine_distance=cfg.fastdeepsort.max_cosine_distance,
            embedder=cfg.fastdeepsort.embedder
        )
        return tracker
    elif tracker_type == 'norfair':
        from norfair_tracker import NorFairTracker

        norfair_tracker = NorFairTracker(
            distance_function=cfg.NORFAIR_TRACK.DISTANCE_FUNCTION,
            distance_threshold=cfg.NORFAIR_TRACK.DISTANCE_THRESHOLD,
            hit_counter_max=cfg.NORFAIR_TRACK.HIT_COUNTER_MAX,
            initialization_delay=cfg.NORFAIR_TRACK.INITIALIZATION_DELAY,
            pointwise_hit_counter_max=cfg.NORFAIR_TRACK.POINTWISE_HIT_COUNTER_MAX,
            detection_threshold=cfg.NORFAIR_TRACK.DETECTION_THRESHOLD,
            past_detections_length=cfg.NORFAIR_TRACK.PAST_DETECTIONS_LENGTH,
            reid_distance_threshold=cfg.NORFAIR_TRACK.REID_DISTANCE_THRESHOLD,
            reid_hit_counter_max=cfg.NORFAIR_TRACK.REID_HIT_COUNTER_MAX,
        )

        return norfair_tracker
    else:
        print(f"No such tracker: {tracker_type}!")
        exit()
