__version__ = 1.1

import torch

from utils.torch_utils import date_modified, git_describe


def print_version():
    git_info = git_describe()

    if git_info is None:
        git_info = f"{date_modified()}"
    else:
        git_info = f"git: {git_info}, {date_modified()}"

    version = f'Version: {__version__}, {git_info}, torch {torch.__version__}'  # string
    print(f"Humans, helmets and uniforms. {version}")


if __name__ == '__main__':
    print_version()
