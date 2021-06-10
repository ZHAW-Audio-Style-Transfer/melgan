from __future__ import annotations
from abc import ABC, abstractmethod
from logger.logger import Logger
from config.config import Config
from pathlib import Path
from tensorflow import keras as K


class BaseModel(ABC):
    def __init__(self):
        Logger.log(f"Initializing {self.__class__.__name__}...")
        self._load() or self._build()

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def predict(self, file: str):
        raise NotImplementedError

    @abstractmethod
    def summary(self):
        raise NotImplementedError

    @abstractmethod
    def _build(self):
        raise NotImplementedError

    @abstractmethod
    def _load(self) -> bool:
        raise NotImplementedError

    def _summary(self, models: dict[str, K.Model]):
        for name, model in models.items():
            Logger.log(f"\nSummary of {name}:")
            model.summary()

            K.utils.plot_model(
                model,
                show_shapes=True,
                to_file=createFolder(self._getModelSavePath()).joinpath(f"{name}.png"),
            )

    def _getModelSavePath(self):
        return Path(Config().environment.output_path, "model")

    def _tensorboardCallback(self):
        path = Path(Config().environment.output_path, "tensorboard")
        return K.callbacks.TensorBoard(log_dir=path, histogram_freq=1)


def createFolder(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path
