import optuna
from optuna.study import StudyDirection

from configs import get_detections_path
from optune.optune_tools import save_result, reset_seed, add_date_prefix, save_callback
from yolo_common.yolo_optune import run_track_yolo


def objective_sort(trial):
    """
sort:
  max_age: 1
  min_hits: 3
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

    max_age = trial.suggest_int('max_age', 1, 50, step=1)
    min_hits = trial.suggest_int('min_hits', 1, 50, step=1)

    tracker_name = "sort"
    tracker_config = \
        {
            tracker_name:
                {
                    "max_age": max_age,
                    "min_hits": min_hits,
                }
        }

    cmp_results = run_track_yolo(txt_source_folder, video_source, tracker_name, tracker_config,
                                 test_func=test_func,
                                 files=files, change_bb=change_bb, classes=classes)

    accuracy = cmp_results["total_equal_percent"]

    return accuracy


def run_optuna():
    study = optuna.create_study(direction=StudyDirection.MAXIMIZE)

    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\Optune"

    tag = add_date_prefix("sort")

    study.set_user_attr("save_folder", output_folder)
    study.set_user_attr("tag", tag)

    study.optimize(objective_sort, n_trials=100, show_progress_bar=True, callbacks=[save_callback])

    trial = study.best_trial

    print(f"Accuracy: {trial.value}")
    print(f"Best hyper parameters: {trial.params}")

    save_result(trial, output_folder, tag, use_date=False, study=study)


if __name__ == '__main__':
    run_optuna()
