from logger.logger import Logger
import os
import numpy as np
import librosa
from numpy.lib.npyio import save
import soundfile as sf
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

# Disable GPU for tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

###############
epoch = "1"
run = "GTZAN-mel"
MEL = True
SLICE_WIDTH = 72
SLICES_PER_SONG = 40
###############

SAMPLE_RATE = 16000
# SAMPLE_RATE = 22050
SLICE_WIDTH_GENERATION = 240
NFTT = 6 * 192
WINDOW = 6 * 192
STRIDE = 192
MEL_FREQUENCIES = 192

MIN_DB = -100
DB_REFERENCE_LEVEL = 20

Logger.initialize(False)


def saveSpectrogram(predictedDbScaleSpectrogram, predictedFile, mel=False):
    spectrogram = librosa.db_to_power(predictedDbScaleSpectrogram)

    if mel:
        # print(["librosa", str(predictedFile)])
        wav = librosa.feature.inverse.mel_to_audio(
            spectrogram,
            sr=SAMPLE_RATE,
            n_fft=NFTT,
            hop_length=STRIDE,
            win_length=WINDOW,
            power=2,
        )
    else:
        wav = librosa.griffinlim(spectrogram, hop_length=STRIDE, win_length=WINDOW)

    predictFilePath = f"{predictedFile}.wav"
    sf.write(predictFilePath, wav, SAMPLE_RATE)


files = list(Path(f"out/").glob(f"*{run}/predictions/*_{epoch}_*"))
# files.reverse()

for item in tqdm(
    list(
        filter(
            lambda item: item.is_dir() and item.name != "temp",
            files,
        )
    )
):
    datasetPath = str(item)

    try:
        if "jazz.00069" in datasetPath:
            SLICES_PER_SONG = 35
        dataset = tf.data.experimental.load(
            datasetPath,
            tf.TensorSpec((SLICES_PER_SONG * SLICE_WIDTH)),
            compression=None,
            reader_func=None,
        )
    except Exception as e:
        Logger.error(f"Error occurred :(")
        Logger.warn(
            f"Does SLICES_PER_SONG ({SLICES_PER_SONG}) match the dataset item length?"
        )
        raise e

    spectrogram = np.asarray(list(iter(dataset)))
    saveSpectrogram(spectrogram, datasetPath, mel=MEL)
