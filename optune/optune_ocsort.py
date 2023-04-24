
from configs import get_detections_path
from optune.optune_tools import reset_seed, common_run_optuna
from yolo_common.yolo_optune import run_track_yolo


def objective_ocsort(trial):
    """
      asso_func: giou
      conf_thres: 0.5122620708221085
      delta_t: 1
      det_thresh: 0
      inertia: 0.3941737016672115
      iou_thresh: 0.22136877277096445
      max_age: 50
      min_hits: 1
      use_byte: false

      ASSO_FUNCS = {  "iou": iou_batch,
                    "giou": giou_batch,
                    "ciou": ciou_batch,
                    "diou": diou_batch,
                    "ct_dist": ct_dist}

    """

    reset_seed()

    video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\"

    files = None
    # files = ['1', "2", "3"]
    # files = ["3"]
    # files = ['6', "8", "26", "36"]

    # classes = [0]
    classes = None

    change_bb = None  # pavel_change_bbox  # change_bbox

    test_func = "timur"
    # test_func = "popov_alex"
    # test_func = "group3"

    txt_source_folder = str(get_detections_path())

    det_thresh = trial.suggest_float('det_thresh', 0.3, 0.6, step=0.1)

    max_age = trial.suggest_int('max_age', 1, 20, log=True)
    # max_age = int(trial.suggest_categorical('max_age', [3, 10, 50, 100]))

    min_hits = trial.suggest_int('min_hits', 6, 8)
    iou_threshold = trial.suggest_float('iou_threshold', 0.6, 0.8, step=0.1)
    delta_t = trial.suggest_int('delta_t', 5, 15)
    asso_func = trial.suggest_categorical('asso_func', ["iou"])
    inertia = trial.suggest_float('inertia', 0.6, 0.8, step=0.1)
    use_byte = trial.suggest_categorical('use_byte', [True])

    # test_func = trial.suggest_categorical('test_func', ["timur", "group_3"])

    tracker_name = "ocsort"
    tracker_config = \
        {
            tracker_name:
                {
                    "det_thresh": det_thresh,
                    "max_age": max_age,
                    "min_hits": min_hits,
                    "iou_thresh": iou_threshold,
                    "delta_t": delta_t,
                    "asso_func": asso_func,
                    "inertia": inertia,
                    "use_byte": use_byte
                }
        }

    cmp_results = run_track_yolo(txt_source_folder, video_source, tracker_name, tracker_config,
                                 test_func=test_func,
                                 files=files, change_bb=change_bb, classes=classes)

    # accuracy = cmp_results["total_equal_percent"]
    accuracy = cmp_results["total_dev_precision"]

    return accuracy


def run_optuna():
    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\Optune"

    common_run_optuna(tracker_tag="ocsort",
                      output_folder=output_folder,
                      objective=objective_ocsort,
                      trials=100)


if __name__ == '__main__':
    run_optuna()
    # test()
