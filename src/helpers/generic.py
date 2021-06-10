from pathlib import Path
import datetime


def ensureFolderExists(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def nfttToFrequencies(nftt: int):
    return nftt // 2 + 1