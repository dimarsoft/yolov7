import optuna
from optuna import Study
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial

from configs import get_detections_path
from optune.optune_tools import save_result, reset_seed, add_date_prefix
from yolo_optune import run_track_yolo


def objective_bytetrack(trial):
    """
bytetrack:
  track_thresh: 0.6  # tracking confidence threshold
  track_buffer: 30   # the frames for keep lost tracks
  match_thresh: 0.8  # matching threshold for tracking
  frame_rate: 30     # FPS
  conf_thres: 0.5122620708221085
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

    txt_source_folder = str(get_detections_path())

    track_thresh = trial.suggest_float('track_thresh', 0.3, 0.9, step=0.1)
    track_buffer = trial.suggest_int('track_buffer', 5, 50, step=5)
    match_thresh = trial.suggest_float('match_thresh', 0.3, 1.0, step=0.1)
    frame_rate = int(trial.suggest_categorical('frame_rate', [3, 12, 20, 30]))
    conf_thres = trial.suggest_float('conf_thres', 0.3, 0.9, step=0.1)

    tracker_name = "bytetrack"
    tracker_config = \
        {
            tracker_name:
                {
                    "track_thresh": track_thresh,
                    "frame_rate": frame_rate,
                    "track_buffer": track_buffer,
                    "match_thresh": match_thresh,
                    "conf_thres": conf_thres,
                }
        }

    cmp_results = run_track_yolo(txt_source_folder, video_source, tracker_name, tracker_config,
                                 test_func=test_func,
                                 files=files, change_bb=change_bb, classes=classes)

    accuracy = cmp_results["total_equal_percent"]

    return accuracy


def my_callback(study: Study, trial: FrozenTrial) -> None:
    save_folder = study.user_attrs["save_folder"]
    tag = study.user_attrs["tag"]

    save_result(trial, save_folder, tag, use_date=False, study=study)


def run_optuna() -> None:
    study = optuna.create_study(direction=StudyDirection.MAXIMIZE)

    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\"

    tag = add_date_prefix("bytetrack")

    study.set_user_attr("save_folder", output_folder)
    study.set_user_attr("tag", tag)

    study.optimize(objective_bytetrack, n_trials=4, show_progress_bar=True, callbacks=[my_callback])

    trial = study.best_trial

    print(f"Accuracy: {trial.value}")
    print(f"Best hyper parameters: {trial.params}")

    save_result(trial, output_folder, tag, use_date=False, study=study)


if __name__ == '__main__':
    run_optuna()
    # test()
