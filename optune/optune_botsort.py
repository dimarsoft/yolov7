from configs import get_detections_path
from optune.optune_tools import reset_seed, common_run_optuna
from yolo_optune import run_track_yolo


def objective_botsort(trial):
    """
botsort:
  appearance_thresh: 0.4818211117541298
  cmc_method: sparseOptFlow, orb, sift, ecc
  conf_thres: 0.3501265956918775
  frame_rate: 30
  lambda_: 0.9896143462366406
  match_thresh: 0.22734550911325851
  new_track_thresh: 0.21144301345190655
  proximity_thresh: 0.5945380911899254
  track_buffer: 60
  track_high_thresh: 0.33824964456239337
  with_reid: True # Вкл/выкл использования модели для предикта фичей
    """

    reset_seed()

    video_source = "d:\\AI\\2023\\corridors\\dataset-v1.1\\test\\"

    files = None
    # files = ['1', "2", "3"]
    # files = ["6"]

    # classes = [0]
    classes = None

    change_bb = None  # pavel_change_bbox  # change_bbox

    test_func = "timur"

    txt_source_folder = str(get_detections_path())

    appearance_thresh = trial.suggest_float('appearance_thresh', 0.2, 0.8, step=0.1)
    conf_thres = trial.suggest_float('conf_thres', 0.2, 0.8, step=0.1)
    match_thresh = trial.suggest_float('match_thresh', 0.1, 0.8, step=0.1)
    new_track_thresh = trial.suggest_float('new_track_thresh', 0.1, 0.8, step=0.1)
    proximity_thresh = trial.suggest_float('proximity_thresh', 0.3, 0.8, step=0.1)
    track_high_thresh = trial.suggest_float('track_high_thresh', 0.1, 0.8, step=0.1)

    lambda_ = trial.suggest_float('lambda_', 0.8, 1.0, step=0.1)

    track_buffer = trial.suggest_int('track_buffer', 20, 100, step=10)

    cmc_method = trial.suggest_categorical('cmc_method',
                                           ["sparseOptFlow", "orb", "sift", "ecc"])

    tracker_name = "botsort"

    tracker_config = \
        {
            tracker_name:
                {
                    "frame_rate": 12,  # но надо как в видео
                    "appearance_thresh": appearance_thresh,
                    "conf_thres": conf_thres,
                    "match_thresh": match_thresh,
                    "new_track_thresh": new_track_thresh,
                    "proximity_thresh": proximity_thresh,
                    "track_high_thresh": track_high_thresh,
                    "track_buffer": track_buffer,
                    "cmc_method": cmc_method,
                    "lambda_": lambda_,
                    "with_reid": False
                }
        }

    cmp_results = run_track_yolo(txt_source_folder, video_source, tracker_name, tracker_config,
                                 test_func=test_func,
                                 files=files, change_bb=change_bb, classes=classes)

    accuracy = cmp_results["total_equal_percent"]

    return accuracy


def run_optuna():
    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\Optune"

    common_run_optuna(tracker_tag="botsort",
                      output_folder=output_folder,
                      objective=objective_botsort,
                      trials=100)


if __name__ == '__main__':
    run_optuna()
