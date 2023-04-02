import os
from datetime import datetime
from pathlib import Path


def get_video_files(source, files: list) -> list:
    # список файлов с видео для обработки
    list_of_videos = []

    source_path = Path(source)

    if source_path.is_dir():
        for entry in source_path.iterdir():
            # check if it is a file
            if entry.is_file() and entry.suffix == ".mp4":
                if files is None:
                    list_of_videos.append(str(entry))
                else:
                    if entry.stem in files:
                        list_of_videos.append(str(entry))

    else:
        list_of_videos.append(str(source))

    return list_of_videos


def create_session_folder(yolo_version, output_folder, task: str) -> str:
    now = datetime.now()

    session_folder_name = f"{now.year:04d}_{now.month:02d}_{now.day:02d}_{now.hour:02d}_{now.minute:02d}_" \
                          f"{now.second:02d}_{yolo_version}_{task}"

    session_folder = str(Path(output_folder) / session_folder_name)

    try:
        os.makedirs(session_folder, exist_ok=True)
        print(f"Directory '{session_folder}' created successfully")
    except OSError as error:
        print(f"Directory '{session_folder}' can not be created. {error}")

    return str(session_folder)
