# %% imports
from helpers.generic import ensureFolderExists, nfttToFrequencies
import tensorflow as tf
import tensorflow_io as tfio
from pathlib import Path
import shutil
import subprocess
from tqdm import tqdm
import librosa

# %% Definitions
SAMPLE_RATE = 16000
# SAMPLE_RATE = 22050
SLICE_WIDTH = 72
SLICE_WIDTH_GENERATION = 240
NFTT = 6 * 192
WINDOW = 6 * 192
STRIDE = 192
MEL_FREQUENCIES = 192

MIN_DB = -100
DB_REFERENCE_LEVEL = 20

frequencies = nfttToFrequencies(NFTT)

SAVE_PATH_NAME = f"sw{SLICE_WIDTH_GENERATION}-{SAMPLE_RATE}hz-nftt{NFTT}-window{WINDOW}-stride{STRIDE}-power2"
SPECTROGRAM_FOLDER = "data/spectrograms/"

sourcePath = Path(f"data/source")
preparedPath = Path(f"data/prepared")
tempPathSpectrograms = Path(f"data/spectrograms_temp")


def convertToMono(inFile, outFile, cwd, sampleRate):
    command = [
        "ffmpeg",
        "-i",
        inFile,
        "-ac",
        "1",
        "-ar",
        str(sampleRate),
        outFile,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    res = subprocess.run(command, cwd=cwd)


# stereo to mono & reduce sampling rate to SAMPLE_RATE Hz
def convert(extension: str = "mp3", overwrite: bool = False):
    suffix = f"_mono_{SAMPLE_RATE}"

    sourceFiles = list(
        filter(
            lambda path: f"{suffix}.{extension}" not in path.name,
            list(sourcePath.glob(f"**/*.{extension}")),
        )
    )

    for file in tqdm(sourceFiles):
        inFile = file.name
        outFile = f"{file.stem}{suffix}{file.suffix}"

        preparedFile = Path(preparedPath, file.relative_to(sourcePath))

        if overwrite or not preparedFile.exists():
            convertToMono(inFile, outFile, file.parent, SAMPLE_RATE)

            ensureFolderExists(preparedFile.parent)
            Path(file.parent, outFile).replace(preparedFile)


# %% Spectrogram
def generateSpectrogram(
    file, sliceWidth=SLICE_WIDTH, melFrequencies=MEL_FREQUENCIES, mel=False
):
    audioTensor = tfio.audio.AudioIOTensor(str(file)).to_tensor()
    if audioTensor.dtype == tf.int16:
        audioTensor = tf.cast(audioTensor, tf.float32) / 32768.0

    if mel:
        spectrogram = tf.transpose(
            librosa.feature.melspectrogram(
                tf.squeeze(audioTensor).numpy(),
                sr=SAMPLE_RATE,
                n_fft=NFTT,
                hop_length=STRIDE,
                win_length=WINDOW,
                power=2,
                n_mels=melFrequencies,
            )
        )
        dbscale_spectrogram = librosa.core.power_to_db(spectrogram, top_db=80.0)
    else:
        spectrogram = tfio.experimental.audio.spectrogram(
            tf.squeeze(audioTensor, axis=[-1]),
            nfft=NFTT,
            window=WINDOW,
            stride=STRIDE,
        )
        dbscale_spectrogram = tfio.experimental.audio.dbscale(spectrogram, top_db=80)

    dbscale_spectrogram_ref = tf.subtract(
        dbscale_spectrogram,
        DB_REFERENCE_LEVEL,
    )

    normalized_dbscale_spectrogram = tf.clip_by_value(
        (((dbscale_spectrogram_ref) - MIN_DB) / (-MIN_DB)) * 2.0 - 1.0, -1.0, 1.0
    )

    sliceCount = normalized_dbscale_spectrogram.shape[0]
    frequencies = normalized_dbscale_spectrogram.shape[1]

    remainder = sliceCount % sliceWidth
    if remainder > 0:
        end_of_song = normalized_dbscale_spectrogram[
            sliceCount - sliceWidth : sliceCount
        ]
        normalized_dbscale_spectrogram = normalized_dbscale_spectrogram[:-remainder]
        normalized_dbscale_spectrogram = tf.concat(
            [normalized_dbscale_spectrogram, end_of_song], 0
        )
    sliceCount = normalized_dbscale_spectrogram.shape[0]

    reshaped = tf.reshape(
        normalized_dbscale_spectrogram,
        [int(sliceCount / sliceWidth), sliceWidth, frequencies],
    )

    final = tf.transpose(reshaped, [0, 2, 1])
    return final


# %% Creating Dataset
def createDataset(
    folder: str, limit: int = None, extension: str = "mp3", mel: bool = False
):
    audioTensors = tf.constant(
        [], shape=(0, MEL_FREQUENCIES if mel else frequencies, SLICE_WIDTH_GENERATION)
    )
    fileList = list(preparedPath.glob(f"{folder}/**/*.{extension}"))[:limit]

    elementSpec = None

    ensureFolderExists(tempPathSpectrograms)
    shutil.rmtree(tempPathSpectrograms)

    savePaths = []
    for i, file in enumerate(tqdm(fileList)):
        spectrogram = generateSpectrogram(file, SLICE_WIDTH_GENERATION, mel=mel)
        audioTensors = tf.concat([audioTensors, spectrogram], 0)

        if i > 0 and i % 50 == 0 or i == len(fileList) - 1:
            datasetNew = tf.data.Dataset.from_tensor_slices(audioTensors)

            savePath = Path(tempPathSpectrograms, str(i))

            ensureFolderExists(savePath)
            tf.data.experimental.save(datasetNew, str(savePath))
            elementSpec = datasetNew.element_spec
            savePaths.append(savePath)

            audioTensors = tf.constant(
                [],
                shape=(
                    0,
                    MEL_FREQUENCIES if mel else frequencies,
                    SLICE_WIDTH_GENERATION,
                ),
            )

    datasets = []
    for path in savePaths:
        datasets.append(
            tf.data.experimental.load(
                str(path),
                elementSpec,
                compression=None,
                reader_func=None,
            )
        )

    datasetFinal = datasets.pop()
    for dataset in datasets:
        datasetFinal = datasetFinal.concatenate(dataset)

    print("Spectrogram generation done.")
    print(len(datasetFinal))

    savePathSpectrograms = Path(
        SPECTROGRAM_FOLDER, f"{SAVE_PATH_NAME}-mel{'Yes' if mel else 'No'}"
    )

    datasetPath = Path(savePathSpectrograms, folder)

    ensureFolderExists(datasetPath)
    shutil.rmtree(datasetPath)
    tf.data.experimental.save(datasetFinal, str(datasetPath))
    with open(Path(savePathSpectrograms, "element_spec.txt"), "w") as file:
        file.write(f"{elementSpec}")
    print("Dataset saved")

    print("Cleaning up")
    shutil.rmtree(tempPathSpectrograms)
    print("Cleaned up")


if __name__ == "__main__":
    convert("wav", overwrite=False)
    createDataset("GTZAN/jazz", extension="wav", mel=True)
    createDataset("GTZAN/classical", extension="wav", mel=True)
    # createDataset("AcousticGuitar", extension="mp3", mel=True)
    # createDataset("maestroPiano2015", extension="mp3", mel=True)
