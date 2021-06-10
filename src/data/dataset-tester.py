from helpers.generic import nfttToFrequencies
import tensorflow as tf
import librosa
import soundfile as sf
from pathlib import Path

# %% Definitions
SAMPLE_RATE = 16000
SLICE_WIDTH = 72
SLICE_WIDTH_GENERATION = 240
NFTT = 6 * 192
WINDOW = 6 * 192
STRIDE = 192
MEL_FREQUENCIES = 192

MIN_DB = -100
DB_REFERENCE_LEVEL = 20

FREQUENCIES = nfttToFrequencies(NFTT)

dataset = tf.data.experimental.load(
    "data/spectrograms/sw240-16000hz-nftt1152-window1152-stride192-power2-melYes/GTZAN/classical",
    tf.TensorSpec((MEL_FREQUENCIES, SLICE_WIDTH_GENERATION)),
    compression=None,
    reader_func=None,
)

# spectrogram = next(iter(dataset.batch(4).skip(27)))
spectrogram = next(iter(dataset.batch(4)))
shape = spectrogram.shape

spectrogram = tf.transpose(
    tf.reshape(
        tf.transpose(tf.squeeze(spectrogram), [0, 2, 1]),
        [shape[0] * shape[2], shape[1]],
    )
)

denormalized_dbscale = (
    ((tf.clip_by_value(spectrogram, -1.0, 1.0) + 1.0) / 2.0) * -MIN_DB
) + MIN_DB
denormalized_dbscale = tf.add(denormalized_dbscale, DB_REFERENCE_LEVEL)


spectrogram = librosa.core.db_to_power(denormalized_dbscale.numpy())

wav = librosa.feature.inverse.mel_to_audio(
    spectrogram,
    sr=SAMPLE_RATE,
    n_fft=NFTT,
    hop_length=STRIDE,
    win_length=WINDOW,
    power=2,
)

filePath = f"{Path('dataset-tester')}.wav"
sf.write(filePath, wav, SAMPLE_RATE)
