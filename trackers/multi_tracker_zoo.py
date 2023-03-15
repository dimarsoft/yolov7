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
        from trackers.deep_sort.tracker import Tracker as DeepSort
        deepsort = DeepSort(metric=cfg.deepsort.metric,
                            max_iou_distance=cfg.deepsort.max_iou_distance,
                            max_age=cfg.deepsort.max_age,
                            n_init=cfg.deepsort.n_unit)
        return deepsort
    elif tracker_type == 'deepsort_enc':
        from trackers.deep_sort.tracker_enc import TrackerWithEncoder as DeepSortWithEncoder
        tracker = DeepSortWithEncoder(encoder_model_filename=cfg.deepsort.encoder_model_filename)
        return tracker
    else:
        print(f"No such tracker: {tracker_type}!")
        exit()
