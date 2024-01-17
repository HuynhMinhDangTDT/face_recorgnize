import logging
import os
import shutil
import subprocess
import tkinter as tk
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from gui import App

from colorama import Fore, Style
from invoke import task

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@task(help={"path": "File or directory to format in place. Required."})
def apply(c, path):
    """
    Apply formatting to the code using black and isort.

    Modify the code in place and make it compliant with PEP 8.

    """
    c.run(f"black {path} --line-length 79")
    c.run(f"isort {path} --profile black --atomic")


@task
def upgrade_env(c):
    """
    Upgrade the existing urobot environment to the newest version.

    """
    c.run(f"conda env update --file environment.yml --prune")
    logging.info("Done")


@task
def upgrade_req(c):
    """
    Upgrade environment.yml and requirements.txt with new installed package.

    """
    c.run(f"conda env export > environment.yml --no-builds")
    c.run(f"pip list --format=freeze > requirements.txt")
    logging.info("Done")


@task
def checkface(c):
    """
    Run test with current machine camera.

    """
    c.run(f"python src/face_rec.py")


@task
def remove_processed(c):
    """
    Remove processed images folder.

    """
    try:
        shutil.rmtree(f"Dataset\FaceData\processed")
    except:
        pass



def print_result(process_name, result):
    """Print the result of a test with color."""
    process_result = "Done" if result.returncode == 0 else "Failed"
    if process_result == "Failed":
        logging.error(
            f"{process_name} "
            + (50 - len(process_name)) * "."
            + Fore.RED
            + f" {process_result}"
            + Style.RESET_ALL
        )
    else:
        logging.info(
            f"{process_name} "
            + (50 - len(process_name)) * "."
            + Fore.GREEN
            + f" {process_result}"
            + Style.RESET_ALL
        )


def execute_process(commands):
    run_command = partial(subprocess.run, capture_output=True)

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(run_command, list(commands.values())))

    process_names = list(commands.keys())
    for process_name, result in zip(process_names, results):
        print_result(process_name, result)


@task
def gui(c):
    app = App(cmd=c)
    app.mainloop()


@task
def train(c):
    """
    Process image and retrain model.

    """
    c.run(f"inv remove-processed")

    commands_remove_empty = {
        "remove_empty_train": [
            "inv",
            "remove-empty",
            "Dataset/FaceData/raw/train",
        ],
        "remove_empty_test": [
            "inv",
            "remove-empty",
            "Dataset/FaceData/raw/test",
        ],
    }

    commands_image_process = {
        "process_train_image": [
            "python",
            "src/align_dataset_mtcnn.py",
            "Dataset/FaceData/raw/train",
            "Dataset/FaceData/processed/train",
            "--image_size",
            "160",
            "--margin",
            "32",
            "--random_order",
            "--gpu_memory_fraction",
            "0.25",
        ],
        "process_test_image": [
            "python",
            "src/align_dataset_mtcnn.py",
            "Dataset/FaceData/raw/test",
            "Dataset/FaceData/processed/test",
            "--image_size",
            "160",
            "--margin",
            "32",
            "--random_order",
            "--gpu_memory_fraction",
            "0.25",
        ],
    }
    
    commands_train = {
        "train_model": [
            "python",
            "src/classifier.py",
            "TRAIN",
            "Dataset/FaceData/processed/train",
            "Models/20180402-114759.pb",
            "facemodel.pkl",
            "--batch_size",
            "256",
        ],
    }

    commands_validate = {
        "update_new_model": [
            "python",
            ".\src\score_metric.py",
            "--path",
            ".\Dataset\FaceData\processed\\test\\",
        ],
    }
    execute_process(commands_remove_empty)
    execute_process(commands_image_process)
    execute_process(commands_train)
    execute_process(commands_validate)

    
@task
def remove_empty(c, root):
    folders = list(os.walk(root))[1:]

    for folder in folders:
        if not folder[2]:
            os.rmdir(folder[0])