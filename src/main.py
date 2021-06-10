import os

# Disable TensorFlow infos
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf

import argparse
from models.base import BaseModel
from models.init import getModel
from pathlib import Path

from logger.logger import Logger
from config.config import Config, loadConfig, saveConfig


parser = argparse.ArgumentParser(
    description="Executes a training session or makes predictions."
)
parser.add_argument(
    "--config",
    type=str,
    default="config/files/example.json",
    help="Environment and training configuration.",
)
parser.add_argument(
    "--run",
    type=str,
    default=None,
    help="Run folder to load config from",
)
parser.add_argument(
    "--predictfile",
    type=str,
    default=None,
    help="Audio file to predict",
)
parser.add_argument(
    "--predictonly",
    default=False,
    action="store_true",
    help="Only run prediction",
)
parser.add_argument(
    "--predictricksamples",
    default=False,
    action="store_true",
    help="Predict some Rick Samples",
)

args = parser.parse_args()

# Load the config
if args.run:
    loadConfig(Path(args.run, "config.json"))
else:
    loadConfig(args.config)

Logger.initialize()

Logger.log("Arguments: ", str(args))
Logger.log(f"Environment: {Config().environment}")
Logger.log(f"Collection Base: {Config().collection.baseFolder}")
Logger.log(f"Collection Rick: {Config().collection.rickPath}")
Logger.log(f"Collection Morty: {Config().collection.mortyPath}")

model: BaseModel = getModel(Config().training.model.name)

saveConfig()


if not args.predictonly:
    model.summary()
    model.train(args.predictfile, args.predictricksamples)

if args.predictfile:
    model.predict([args.predictfile])

if args.predictricksamples:
    model.predictRickSamples()