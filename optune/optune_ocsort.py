import optuna
from optuna.study import StudyDirection

from configs import DETECTIONS_FOLDER
from optune.optune_tools import save_result, reset_seed
from yolo_optune import run_track_yolo


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

    # classes = [0]
    classes = None

    change_bb = None  # pavel_change_bbox  # change_bbox

    test_func = "timur"

    txt_source_folder = DETECTIONS_FOLDER

    conf_thres = trial.suggest_float('conf_thres', 0.3, 0.6, step=0.1)

    # max_age = trial.suggest_int('max_age', 1, 10, log=True)
    max_age = int(trial.suggest_categorical('max_age', [10, 50, 100]))

    min_hits = trial.suggest_int('min_hits', 6, 8)
    iou_threshold = trial.suggest_float('iou_threshold', 0.6, 0.8, step=0.1)
    delta_t = trial.suggest_int('delta_t', 5, 8)
    asso_func = trial.suggest_categorical('asso', ["iou", "giou"])
    inertia = trial.suggest_float('inertia', 0.6, 0.8, step=0.1)
    use_byte = trial.suggest_categorical('use_byte', [True, False])

    # test_func = trial.suggest_categorical('test_func', ["timur", "group_3"])

    tracker_name = "ocsort"
    tracker_config = \
        {
            tracker_name:
                {
                    "det_thresh": conf_thres,
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

    accuracy = cmp_results["total_equal_percent"]

    return accuracy


def run_optuna():
    study = optuna.create_study(direction=StudyDirection.MAXIMIZE)
    study.optimize(objective_ocsort, n_trials=60, show_progress_bar=True)

    trial = study.best_trial

    print(f"Accuracy: {trial.value}")
    print(f"Best hyper parameters: {trial.params}")

    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\"

    save_result(trial, output_folder, "ocsort")


if __name__ == '__main__':
    run_optuna()
