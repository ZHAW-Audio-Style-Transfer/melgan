from helpers.generic import get_timestamp
import json
import os
from pathlib import Path
from typing import Literal, NamedTuple, Union
import typedload


class ConfigEnvironment(NamedTuple):
    output_path: str
    log_file: str
    rickSamplesDirectory: str


class ConfigCollection(NamedTuple):
    baseFolder: str
    rickPath: str
    mortyPath: str
    sampleRate: int
    sliceWidth: int
    nftt: int
    window_length: int
    strides: int
    batchTruncRatio: float
    min_level_db: int
    reference_db_level: int
    mel_frequencies: Union[int, None]


class ConfigAdamOptimizer(NamedTuple):
    learning_rate: float
    beta_1: float


class ConfigModelGenerator(NamedTuple):
    optimizer: ConfigAdamOptimizer
    filters: int
    unetBlocks: int
    lambda_adversarial: float
    lambda_travel: float
    lambda_margin: float


class ConfigModelDiscriminator(NamedTuple):
    optimizer: ConfigAdamOptimizer
    filters: int
    labelSmoothing: float


class ConfigModelSiamese(NamedTuple):
    optimizer: ConfigAdamOptimizer
    filters: int
    latentDim: int
    margin: int


class ConfigModel(NamedTuple):
    name: Union[
        Literal["MelGAN"],
    ]
    generator: ConfigModelGenerator
    discriminator: ConfigModelDiscriminator
    siamese: ConfigModelSiamese


class ConfigTraining(NamedTuple):
    model: ConfigModel
    epochs: int
    batch_size: int
    sliceWidth: int
    generatorUpdateFrequency: int
    dataRepetition: int


class ConfigType(NamedTuple):
    name: str
    environment: ConfigEnvironment
    collection: ConfigCollection
    training: ConfigTraining


config: ConfigType = None


def loadConfig(file: str):
    global config
    with open(file) as f:
        loadedConfig = json.load(f)

    if "saved_run" in loadedConfig["environment"]:
        loadedConfig["environment"]["output_path"] = loadedConfig["environment"][
            "saved_run"
        ]
        loadedConfig["environment"].pop("saved_run")
    else:
        loadedConfig["environment"][
            "output_path"
        ] += f"/{get_timestamp()}_{loadedConfig['name']}"

    config = typedload.load(loadedConfig, ConfigType)


def saveConfig():
    global config

    file = Path(config.environment.output_path, "config.json")
    os.makedirs(config.environment.output_path, exist_ok=True)

    configCopy = typedload.dump(config)
    configCopy["environment"]["saved_run"] = config.environment.output_path

    with open(file, "w") as f:
        json.dump(configCopy, f)


def Config():
    return config
