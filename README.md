# Audio Style Transfer - MelGAN

This is an implementation of the [MelGAN network by Marco Pasini](https://github.com/marcoppasini/MelGAN-VC) developed by [Michael Schaufelberger (michaelschufi)](https://github.com/michaelschufi), [Raphael Mailänder (mailaenderli)](https://github.com/mailaenderli), and [Gabriel Koch (elessar-ch)](https://github.com/elessar-ch) during our Bachelor thesis at the Institute of Embedded Systems at with ZHAW School of Engineering.

![ZHAW InES Logo](images/ines_logo.png)

The implementation uses Python with TensorFlow and Librosa.

The project framework itself is inspired from [unmix-io](https://github.com/unmix-io/unmix-net).

## Prerequisites & Setup
We use conda and [Anaconda Project](https://github.com/Anaconda-Platform/anaconda-project) to manage dependencies.
This allows specifying conda and pip dependencies at the same time.
We only use pip packages and use only conda for providing Python and NVIDIA CUDA drivers.

### Anaconda Project Installation

1. Install Conda for your operating system  
   (Miniconda should probably also work, but was not tested)

   Windows:
   ```powershell
   choco install -y anaconda3
   ```
2. Initialize Conda
   ```
   conda init {bash, cmd.exe, fish, powershell, tcsh, xonsh, zsh}
   ```
3. Install Anaconda Project
   Make sure you run this inside the base conda environment on Windows (permission problems otherwise).
   ```
   conda install anaconda-project
   ```
4. Specify that anaconda-project should use the global env directory to create its environments.
   
   Windows (Anaconda 3):
   ```powershell
   [Environment]::SetEnvironmentVariable("ANACONDA_PROJECT_ENVS_PATH", "YOUR_ANACONDA_PATH\envs", [System.EnvironmentVariableTarget]::Machine)
   ```

   Otherwise, the conda environment doesn't have a name and it makes the integration with tools like VS Code brake in certain cases.

### CUDA Drivers
We strongly recommend using GPUs or TPUs to train the network.

To install the drivers necessary, consult the [TensorFlow documentation](https://www.tensorflow.org/install/gpu#software_requirements).

The implementation was tested with
- cudatoolkit version 11.0.221
- cudnn version 8.0.4

installed via conda
```
conda install -c nvidia cudatoolkit==11.0.221 cudnn==8.0.4
```

### Package Installation
1. Install packages
   To install all packages, run the following:
   
   ```
   anaconda-project prepare
   ```
2. Activate the conda environment  
   ```
   conda activate melgan-UwpwzX7tcrj7eAR1RJVec
   ```

   Note: You might need to do this multiple times, since there's a bug that Pip dependencies are not linked when installed right after the initial conda environment creation.

### FFmpeg & Pydot
An installation of FFmpeg is required for preprocessing.
For printing the model summary, pydot (and GraphViz) is required.

### Datasets
Out of 3 datasets we used during our thesis, 1 of them is publically available and 1 partially.

- GTZAN - Includes 100 songs per 10 different genres, 30 seconds each (1GB)
  - [Download from Kaggle](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)
  - We mainly tested the Jazz and Classical genres.
- Guitar and Piano Datasets (Instrumental-only)
  - The Maestro Dataset v2 from [Kaggle](https://www.kaggle.com/jackvial/themaestrodatasetv2) consists of 200 hours of piano recordings with midi transcriptions.
    - We used only songs from 2015 as they do not include other instruments.
  - The guitar dataset we used, was from our personal collection.

## Usage

### Repository Structure
The repository is a simple framework that allows for training different model implementations and predicting them, as well as pre- and post-processing.

- `main.py` - The main script used to start/resume a training session and/or predicting songs.
- `save-predicted-spectrogram.py` - Script to convert predicted spectrograms to audio files.
- `models` - Contains all models/networks
  - The models implement the base class `BaseModel` in `models/base.py`
- `logger` - Just a static logging class
- `helpers` - Some helper functions
- `data` - All datasets reside in here. See [below](#preprocessing).
- `config` - Configuration files that specify hyperparameters, the dataset to be used, model architecture (number of filter, layers, etc.) and more.

### Preprocessing
Before starting the training, the audio files from the dataset have to be turned into spectrograms.

You can do that the following way:

1. Unpack the dataset into the `data/source` folder (arbitrarily nested).  
   e.g. `data/source/GTZAN/jazz` or `data/source/GTZAN/classical`
2. Open the [`spectrogram_generator.py`](src/data/spectrogram_generator.py) file and specify the constants at the top of the file.
3. At the very bottom of the file, specify the following:
   - Use 
     ```python
     convert(extension: str = "mp3", overwrite=False)
     ``` 
     to resample the audio files to the specified sample rate (`SAMPLE_RATE`) and convert from stereo to mono.
   - Use  
     ```python
     createDataset(folder: str, limit: int = None, extension: str = "mp3", mel: bool = False)
     ```
     to create a dataset after resampling the audio files.
4. Run the script  
   ```bash
   python data/spectrogram_generator.py
   ```

You can test the generated dataset before training with `data/dataset-tester.py`.

### Creating a Run Configuration File
Configuration files specify hyperparameters, the dataset to be used, model architecture (number of filter, layers, etc.) and more.

They are stored in [`config/files`](src/config/files).

We have prepared two example configuration files for the GTZAN dataset.
One uses mel spectrograms and the other does not.

The configuration is loaded at the start of the `main.py` script. 
It has basic type-validation, to, for example, prevent accidental usage of integers instead of floats.

The configuration variables are used throughout the code by calling the `Config()` function:

```python
from config.config import Config

print(Config().training.model.name)
```

### Training
To start a training you have to specify a configuration file (see above).

Run the following command to start a training run
```
python main.py --config config/files/example_GTZAN-mel.json
```

#### Resuming a Run
To resume a run, specify the path to the run folder created:
```
python main.py --run out/2021-06-01-15-32-30_GTZAN-mel
```
#### Periodical Predictions
To periodically run predictions during training you have two options:
- Specifying a single file using the commandline argument `--predictfile` like:
- Specifying a folder, which holds files to predict.
  1. Create a folder anywhere in the project and place the songs in there.
  2. Specify the path top the folder in the configuration file under `environment.rickSamplesDirectory`.
  3. Run the training with `--predictricksamples`.

#### Predicting after the Run Completed
To predict a file, run
```
python main.py --run out/2021-06-01-15-32-30_GTZAN-mel --predictfile path/to/file.wav --predictonly
```

To predict the prepared folder, run
```
python main.py --run out/2021-06-01-15-32-30_GTZAN-mel --predictricksamples --predictonly
```

#### Converting Predictions to Audio Files
During training, predictions are automatically stored inside the run's output folder.  
To not slow down the training process, they are not automatically converted into audio files during training.

Use the `convert-predicted-spectrogram.py` script to do that.
Simply specify the output folder and the epoch in which the predictions were created and the dataset parameters.
The predictions itself can be found in the output folder under `out/`.
