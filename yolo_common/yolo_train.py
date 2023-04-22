import argparse
import json
from pathlib import Path

from configs import parse_yolo_version, YoloVersion
from tools.path_tools import create_session_folder
from utils.general import set_logging
from yolov7 import YOLO7
from yolov8_ultralitics import YOLO8UL


def create_yolo_train_model(yolo_version, model):
    if yolo_version == YoloVersion.yolo_v7:
        return YOLO7(model)

    if yolo_version == YoloVersion.yolo_v8 or yolo_version == YoloVersion.yolo_v8ul:
        return YOLO8UL(model)


def run_train_yolo_cli(opt_info):
    set_logging()

    yolo_info, weights, output_folder = \
        opt_info.yolo, opt_info.weights, opt_info.output_folder

    model = weights

    print(f"yolo version = {yolo_info}")
    yolo_version = parse_yolo_version(yolo_info)

    if yolo_version is None:
        raise Exception(f"unsupported yolo version {yolo_info}")

    # в выходной папке создаем папку с сессией: дата_трекер туда уже сохраняем все файлы

    session_folder = create_session_folder(yolo_version, output_folder, "train")

    session_info = dict()

    session_info['model'] = str(Path(model).name)
    session_info['yolo_version'] = str(yolo_version)

    session_info['batch'] = str(opt_info.batch)
    session_info['epochs'] = str(opt_info.epochs)
    session_info['weights'] = str(opt_info.weights)
    session_info['data'] = str(opt_info.data)

    session_info_path = str(Path(session_folder) / 'session_info.json')

    train_runs_path = str(Path(session_folder) / 'train_runs')

    with open(session_info_path, "w") as session_info_file:
        json.dump(session_info, fp=session_info_file, indent=4)

    yolo_model = create_yolo_train_model(yolo_version, model)

    # args = dict(model=model, data=data, batch=batch, epochs=epochs, project=train_runs_path, verbose=True)

    opt_info.project = train_runs_path
    opt_info.verbose = True

    yolo_model.train(**opt_info)


def run_train_yolo(**opt_info):
    set_logging()

    yolo_info, weights, output_folder = \
        opt_info.get("yolo"), opt_info.get("../weights"), opt_info.get("project")

    # del opt_info["yolo"]

    opt_info.pop("yolo")

    model = weights

    print(f"yolo version = {yolo_info}")
    yolo_version = parse_yolo_version(yolo_info)

    if yolo_version is None:
        raise Exception(f"unsupported yolo version {yolo_info}")

    # в выходной папке создаем папку с сессией: дата_трекер туда уже сохраняем все файлы

    session_folder = create_session_folder(yolo_version, output_folder, "train")

    session_info = dict()

    session_info['model'] = str(Path(model).name)
    session_info['yolo_version'] = str(yolo_version)

    session_info['batch'] = str(opt_info.get("batch"))
    session_info['epochs'] = str(opt_info.get("epochs"))
    session_info['weights'] = str(opt_info.get("../weights"))
    session_info['data'] = str(opt_info.get("data"))

    session_info_path = str(Path(session_folder) / 'session_info.json')

    train_runs_path = str(Path(session_folder) / 'train_runs')

    with open(session_info_path, "w") as session_info_file:
        json.dump(session_info, fp=session_info_file, indent=4)

    yolo_model = create_yolo_train_model(yolo_version, model)

    # args = dict(model=model, data=data, batch=batch, epochs=epochs, project=train_runs_path, verbose=True)

    opt_info["project"] = train_runs_path
    opt_info["verbose"] = True

    if yolo_version != YoloVersion.yolo_v7:
        opt_info.pop("../weights")
        opt_info["model"] = weights

    yolo_model.train(**opt_info)


def run_example():
    model = "yolov8n.pt"

    output_folder = "d:\\AI\\2023\\corridors\\dataset-v1.1\\my_train"

    data = "D:\\AI\\2023\\dataset-v1.1\\data_custom.yaml"

    args = dict(yolo="8ul", weights=model, data=data, batch=1, epochs=1, project=output_folder)

    run_train_yolo(**args)


if __name__ == '__main__':
    run_example()

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolo7.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default='data/coco.yaml', help='data.yaml path')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.p5.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')
    parser.add_argument('--workers', type=int, default=8, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--entity', default=None, help='W&B entity')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--linear-lr', action='store_true', help='linear LR')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--artifact_alias', type=str, default="latest", help='version of dataset artifact to be used')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0],
                        help='Freeze layers: backbone of yolov7=50, first3=0 1 2')
    parser.add_argument('--v5-metric', action='store_true', help='assume maximum recall as 1.0 in AP calculation')
    opt = parser.parse_args()

    run_train_yolo_cli(opt)
