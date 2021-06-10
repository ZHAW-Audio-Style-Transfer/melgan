from __future__ import annotations
import multiprocessing as mp
from multiprocessing.context import Process
from models.MelGAN.discriminator import Discriminator
from models.MelGAN.generator import Generator
from models.MelGAN.siamese import Siamese
import os

from tensorflow.python.ops.numpy_ops.np_math_ops import isnan
from data.spectrogram_generator import convertToMono, generateSpectrogram
from itertools import combinations
from typing import Any, TypedDict
from helpers.generic import ensureFolderExists, nfttToFrequencies
from config.config import Config
import tensorflow as tf
from tensorflow import keras as K
from models.base import BaseModel
from pathlib import Path
from logger.logger import Logger
import numpy as np
from tqdm import tqdm
import librosa
import soundfile as sf
import csv

import psutil


class TrainLosses(TypedDict):
    generator_adversarial_loss: Any
    travel_loss: Any
    margin_loss: Any
    total_generator_loss: Any
    total_discriminator_loss: Any
    siamese_loss: Any
    generator_adversarial_loss_melgan: Any
    travel_loss_melgan: Any
    margin_loss_melgan: Any
    total_generator_loss_melgan: Any
    total_discriminator_loss_melgan: Any
    siamese_loss_melgan: Any


class MelGAN(BaseModel):
    lossInstance = K.losses.BinaryCrossentropy(from_logits=True)
    nextEpoch = 0

    def __init__(self):
        # Config values
        self.sliceWidthDataset = Config().collection.sliceWidth
        self.sliceWidth = Config().training.sliceWidth
        self.nfft = Config().collection.nftt
        self.window_length = Config().collection.window_length
        self.strides = Config().collection.strides
        self.frequencies = Config().collection.mel_frequencies or nfttToFrequencies(
            Config().collection.nftt
        )

        self.min_level_db = Config().collection.min_level_db
        self.reference_db_level = Config().collection.reference_db_level
        self.sampleRate = Config().collection.sampleRate
        self.generatorUpdateFrequency = Config().training.generatorUpdateFrequency
        self.dataRepetition = Config().training.dataRepetition

        self.discriminatorLabelSmoothing = (
            Config().training.model.discriminator.labelSmoothing
        )
        self.lambda_margin = Config().training.model.generator.lambda_margin
        self.lambda_travel = Config().training.model.generator.lambda_travel
        self.siameseMargin = Config().training.model.siamese.margin

        # Optimizers
        self.optimizerGeneratorSiamese = tf.keras.optimizers.Adam(
            Config().training.model.generator.optimizer.learning_rate,
            Config().training.model.generator.optimizer.beta_1,
        )
        self.optimizerDiscriminatorMorty = tf.keras.optimizers.Adam(
            Config().training.model.discriminator.optimizer.learning_rate,
            Config().training.model.discriminator.optimizer.beta_1,
        )

        self.checkpoint = tf.train.Checkpoint(
            optimizerGeneratorSiamese=self.optimizerGeneratorSiamese,
            optimizerDiscriminatorMorty=self.optimizerDiscriminatorMorty,
        )
        self.checkpointManager = tf.train.CheckpointManager(
            self.checkpoint, self._getModelSavePath().joinpath("optimizers"), 1
        )

        self.predictInputCache = {}
        # self.optimizerSiamese = tf.keras.optimizers.Adam(
        #     Config().training.model.siamese.optimizer.learning_rate,
        #     Config().training.model.siamese.optimizer.beta_1,
        # )

        self.summary_writer = tf.summary.create_file_writer(
            Config().environment.output_path + "/tensorboard"
        )

        super().__init__()

    def train(self, predictFile: str = None, predictRickSamples: bool = False):
        datasetRickOriginal = self.__loadDataset(Config().collection.rickPath)
        datasetMortyOriginal = self.__loadDataset(Config().collection.mortyPath)

        maxDatasetLength = max(len(datasetRickOriginal), len(datasetMortyOriginal))
        truncationLength = int(maxDatasetLength * Config().collection.batchTruncRatio)
        truncationLengthRick = min(truncationLength, len(datasetRickOriginal))
        truncationLengthMorty = min(truncationLength, len(datasetMortyOriginal))

        datasetRickOriginal = datasetRickOriginal.take(truncationLengthRick)
        datasetMortyOriginal = datasetMortyOriginal.take(truncationLengthMorty)

        datasetRickRepeated = datasetRickOriginal.repeat(self.dataRepetition).map(
            self.tfRandomCrop, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        datasetMortyRepeated = datasetMortyOriginal.repeat(self.dataRepetition).map(
            self.tfRandomCrop, num_parallel_calls=tf.data.experimental.AUTOTUNE
        )

        batchSize = Config().training.batch_size
        epochs = Config().training.epochs

        for epoch in range(self.nextEpoch, epochs):
            Logger.log(f"Epoch {epoch+1}/{epochs}")
            Logger.time("epoch")

            dsRickRepeatedLength = len(datasetRickRepeated)
            dsMortyRepeatedLength = len(datasetMortyRepeated)

            datasetRick = datasetRickRepeated.shuffle(dsRickRepeatedLength).batch(
                batchSize
            )
            datasetMorty = datasetMortyRepeated.shuffle(dsMortyRepeatedLength).batch(
                batchSize
            )

            batchesPerEpoch = min(len(datasetRick), len(datasetMorty))
            # batchesPerEpoch = 20
            datasetRickIter = iter(datasetRick)
            datasetMortyIter = iter(datasetMorty)

            batchLosses = []
            skippedBatchesCount = 0

            process = psutil.Process(os.getpid())
            memoryUsedStart = process.memory_info().rss // 1024 // 1024

            Logger.log(f"Start of epoch: {memoryUsedStart} MB")

            for i in tqdm(range(batchesPerEpoch)):
                rick = next(datasetRickIter)
                morty = next(datasetMortyIter)

                if tf.experimental.numpy.any(
                    tf.math.less(
                        tf.math.count_nonzero(rick, axis=[1, 2]),
                        int(rick.shape[1] * rick.shape[2] * 0.98),
                    )
                ) or tf.experimental.numpy.any(
                    tf.math.less(
                        tf.math.count_nonzero(morty, axis=[1, 2]),
                        int(morty.shape[1] * morty.shape[2] * 0.98),
                    )
                ):
                    skippedBatchesCount += 1
                    continue

                losses = self.tfTrainStep(
                    rick, morty, (i % self.generatorUpdateFrequency == 0)
                )
                for key in losses.keys():
                    if losses[key] is None:
                        if len(batchLosses) < 1:
                            losses[key] = 0
                        else:
                            losses[key] = batchLosses[-1][key]

                batchLosses.append(losses)

            Logger.log("Epoch finished")

            memoryUsedEnd = process.memory_info().rss // 1024 // 1024
            Logger.log(
                f"End of epoch: {memoryUsedEnd} MB (Diff: {memoryUsedEnd - memoryUsedStart} MB)"
            )

            Logger.warn(f"Skipped batches: {skippedBatchesCount}")

            epochLosses = self.__dict_mean(batchLosses)
            self.__checkNan(epochLosses, batchLosses)
            self.__writeTensorboardLogs("epoch", epoch + 1, epochLosses)

            self.nextEpoch = epoch + 1
            if epoch == 0 or (epoch + 1) % 10 == 0:
                Logger.time("saveModels")
                self.__saveModels(epoch + 1)
                Logger.time_end("saveModels")

            if epoch == 0 or (epoch + 1) % 20 == 0:
                Logger.time("predictions")

                if predictFile:
                    self.predict([predictFile])
                if predictRickSamples:
                    self.predictRickSamples()

                if not predictFile and not predictRickSamples:
                    Logger.log("No files for predictions given...")

                Logger.time_end("predictions")
            Logger.time_end("epoch")

    @tf.function
    def tfRandomCrop(self, image):
        return tf.image.random_crop(image, size=[self.frequencies, self.sliceWidth])

    @tf.function
    def tfTrainStep(self, rick, morty, trainAll) -> TrainLosses:
        travelLoss = None
        generatorLoss = None
        marginLossRick = None
        siameseLoss = None
        travelLossMelGan = None
        generatorLossMelGan = None
        marginLossRickMelGan = None
        siameseLossMelGan = None

        with tf.GradientTape(persistent=True) as tape:
            ricks = tf.split(rick, 3, axis=2)

            rickAsMorty0 = self.modelGeneratorRickMorty(ricks[0], training=True)
            rickAsMorty1 = self.modelGeneratorRickMorty(ricks[1], training=True)
            rickAsMorty2 = self.modelGeneratorRickMorty(ricks[2], training=True)

            rickAsMorty = tf.concat([rickAsMorty0, rickAsMorty1, rickAsMorty2], axis=2)

            discMorty = self.modelDiscriminatorMorty(morty, training=True)
            discRickAsMorty = self.modelDiscriminatorMorty(rickAsMorty, training=True)

            if trainAll:
                siamRick0 = self.modelSiamese(ricks[0], training=True)
                siamRick2 = self.modelSiamese(ricks[2], training=True)
                siamRickAsMorty0 = self.modelSiamese(rickAsMorty0, training=True)
                siamRickAsMorty2 = self.modelSiamese(rickAsMorty2, training=True)

            # Adversial Loss
            (
                discriminatorAdversarialLoss,
                generatorAdversarialLoss,
            ) = self.tfCalcAdversarialLoss(
                discMorty,
                discRickAsMorty,
            )

            discriminatorAdversarialLossMelGan = (
                self.d_loss_f(discRickAsMorty) + self.d_loss_r(discMorty)
            ) / 2.0
            generatorAdversarialLossMelGan = self.g_loss_f(discRickAsMorty)

            if trainAll:
                # Margin Loss
                marginLossRick = self.tfCalcMarginLoss(siamRick0, siamRick2)
                marginLossRickMelGan = self.loss_siamese(siamRick0, siamRick2)
                # marginLossRickAsMorty = self.tfCalcMarginLoss(siamRickAsMorty)

                # Travel Loss
                travelLoss = self.tfCalcTravelLoss(
                    siamRick0, siamRick2, siamRickAsMorty0, siamRickAsMorty2
                )
                travelLossMelGan = self.loss_travel(
                    siamRick0, siamRickAsMorty0, siamRick2, siamRickAsMorty2
                )

                # Final loss
                generatorLoss = (
                    generatorAdversarialLoss
                    + self.lambda_margin * marginLossRick
                    + self.lambda_travel * travelLoss
                )
                siameseLoss = (
                    generatorAdversarialLoss + self.lambda_margin * marginLossRick
                )

                generatorLossMelGan = (
                    generatorAdversarialLossMelGan
                    + self.lambda_margin * marginLossRickMelGan
                    + self.lambda_travel * travelLossMelGan
                )
                siameseLossMelGan = (
                    generatorAdversarialLossMelGan
                    + self.lambda_margin * marginLossRickMelGan
                )

            discriminatorLoss = discriminatorAdversarialLoss
            discriminatorLossMelGan = discriminatorAdversarialLossMelGan

        if trainAll:
            self.tfDoGradientStep(
                generatorLossMelGan,
                self.modelGeneratorRickMorty.trainable_variables
                + self.modelSiamese.trainable_variables,
                tape,
                self.optimizerGeneratorSiamese,
            )

        self.tfDoGradientStep(
            discriminatorLossMelGan,
            self.modelDiscriminatorMorty.trainable_variables,
            tape,
            self.optimizerDiscriminatorMorty,
        )

        return {
            "generator_adversarial_loss": generatorAdversarialLoss,
            "travel_loss": travelLoss,
            "margin_loss": marginLossRick,
            "total_generator_loss": generatorLoss,
            "total_discriminator_loss": discriminatorLoss,
            "siamese_loss": siameseLoss,
            "generator_adversarial_loss_melgan": generatorAdversarialLossMelGan,
            "travel_loss_melgan": travelLossMelGan,
            "margin_loss_melgan": marginLossRickMelGan,
            "total_generator_loss_melgan": generatorLossMelGan,
            "total_discriminator_loss_melgan": discriminatorLossMelGan,
            "siamese_loss_melgan": siameseLossMelGan,
        }

    def tfDoGradientStep(
        self, loss, trainable_variables, tape, optimizer: K.optimizers.Optimizer
    ):
        temp = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(
            zip(
                temp,
                trainable_variables,
            )
        )

    def tfCalcAdversarialLoss(self, discMorty, discRickAsMorty):
        discMortyLoss = self.lossInstance(
            tf.multiply(tf.ones_like(discMorty), self.discriminatorLabelSmoothing),
            discMorty,
        )

        discRickAsMortyLoss = self.lossInstance(
            tf.zeros_like(discRickAsMorty),
            discRickAsMorty,
        )

        genRickAsMortyLoss = self.lossInstance(
            tf.ones_like(discRickAsMorty),
            discRickAsMorty,
        )

        discriminatorLoss = tf.math.reduce_sum(
            [
                discMortyLoss,
                discRickAsMortyLoss,
            ]
        )

        generatorLoss = tf.math.reduce_sum(
            [
                genRickAsMortyLoss,
            ]
        )

        return discriminatorLoss, generatorLoss

    def tfCalcTravelLoss(
        self, siamRick0, siamRick2, siamRickAsMorty0, siamRickAsMorty2
    ):

        # tijS = siamRick0 - siamRick1
        # tijT = siamRickAsMorty0 - siamRickAsMorty1

        # cosine(tijS, tijT)

        diffRick = tf.math.subtract(siamRick0, siamRick2)
        diffRickAsMorty = tf.math.subtract(siamRickAsMorty0, siamRickAsMorty2)

        cosineLossFunction = K.losses.CosineSimilarity()
        mseLossRickToMorty = K.losses.mean_squared_error(diffRick, diffRickAsMorty)
        cosineLossRickToMorty = cosineLossFunction(diffRick, diffRickAsMorty)

        travelLossRickToMorty = tf.math.reduce_mean(
            mseLossRickToMorty + cosineLossRickToMorty
        )

        return travelLossRickToMorty

    def tfCalcMarginLoss(self, siam0, siam2):
        loss = tf.math.subtract(siam0, siam2)

        loss = tf.math.reduce_mean(
            tf.nn.relu(self.siameseMargin - tf.norm(loss, axis=1, ord=2))
        )

        return loss

    def loss_travel(self, sa, sab, sa1, sab1):
        l1 = tf.reduce_mean(((sa - sa1) - (sab - sab1)) ** 2)
        l2 = tf.reduce_mean(
            tf.reduce_sum(
                -(
                    tf.nn.l2_normalize(sa - sa1, axis=[-1])
                    * tf.nn.l2_normalize(sab - sab1, axis=[-1])
                ),
                axis=-1,
            )
        )
        return l1 + l2

    def loss_siamese(self, sa, sa1):
        logits = tf.sqrt(tf.reduce_sum((sa - sa1) ** 2, axis=-1, keepdims=True))
        return tf.reduce_mean(tf.square(tf.maximum((self.siameseMargin - logits), 0)))

    def d_loss_f(self, fake):
        return tf.reduce_mean(tf.maximum(1 + fake, 0))

    def d_loss_r(self, real):
        return tf.reduce_mean(tf.maximum(1 - real, 0))

    def g_loss_f(self, fake):
        return tf.reduce_mean(-fake)

    def tfGetPairs(self, size):
        return np.asarray(list(combinations(range(size), 2)))

    def __saveModels(self, nextEpoch):
        K.models.save_model(
            self.modelGeneratorRickMorty,
            self._getModelSavePath().joinpath("generatorRickMorty"),
        )
        K.models.save_model(
            self.modelDiscriminatorMorty,
            self._getModelSavePath().joinpath("discriminatorMorty"),
        )
        K.models.save_model(
            self.modelSiamese,
            self._getModelSavePath().joinpath("siamese"),
        )

        self.checkpointManager.save()

        nextEpochFile = self._getModelSavePath().joinpath("next_epoch.txt")
        self.nextEpoch = nextEpoch

        with open(nextEpochFile, "w") as file:
            file.write(str(nextEpoch))

    def __loadModels(self):
        self.modelGeneratorRickMorty = K.models.load_model(
            self._getModelSavePath().joinpath("generatorRickMorty"),
            compile=False,
        )
        self.modelDiscriminatorMorty = K.models.load_model(
            self._getModelSavePath().joinpath("discriminatorMorty"),
            compile=False,
        )
        self.modelSiamese = K.models.load_model(
            self._getModelSavePath().joinpath("siamese"),
            compile=False,
        )

        status = self.checkpoint.restore(self.checkpointManager.latest_checkpoint)
        # status.assert_consumed()
        status.assert_existing_objects_matched()

        nextEpochFile = self._getModelSavePath().joinpath("next_epoch.txt")

        if nextEpochFile.exists():
            with open(nextEpochFile, "r") as file:
                nextEpoch = int(file.readline())

        return nextEpoch if nextEpoch else 0

    def __checkNan(self, epochLosses, batchLosses):
        if isnan(epochLosses["total_generator_loss"]) or isnan(
            epochLosses["total_discriminator_loss"]
            or isnan(epochLosses["siamese_loss"])
        ):
            Logger.error("Caught epoch_losses with NAN!")
            Logger.log(epochLosses)
            Logger.log(batchLosses[-10:])
            csv_file = "batchlosses.csv"
            try:
                with open(csv_file, "w") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=batchLosses[0].keys())
                    writer.writeheader()
                    for data in batchLosses:
                        writer.writerow(data)
            except IOError:
                Logger.error("I/O error")

    def __dict_mean(self, dict_list):
        mean_dict = {}
        for key in dict_list[0].keys():
            mean_dict[key] = sum(d[key] for d in dict_list) / len(dict_list)
        return mean_dict

    def __writeTensorboardLogs(self, scope: str, step: int, losses: dict):
        with self.summary_writer.as_default():
            with tf.name_scope(scope):
                for name, value in losses.items():
                    tf.summary.scalar(name, value, step=step)

    def predictRickSamples(self):
        sourcePath = Path(Config().environment.rickSamplesDirectory)
        predictFiles = list(sourcePath.glob("**/*.mp3"))
        self.predict(list(map(str, predictFiles)))

    def predict(self, files: list[str]):

        for file in files:
            if file in self.predictInputCache:
                spectrogram = self.predictInputCache[file]
            else:
                sourceFile = Path(file)
                convertedFile = Path(
                    Config().environment.output_path,
                    "predictions/temp",
                    sourceFile.name,
                )
                predictedFile = Path(
                    Config().environment.output_path, "predictions", sourceFile.name
                )
                cwd = Path(os.getcwd())

                ensureFolderExists(convertedFile.parent)
                convertToMono(sourceFile, convertedFile, str(cwd), self.sampleRate)
                spectrogram = generateSpectrogram(
                    str(convertedFile),
                    self.sliceWidth,
                    melFrequencies=Config().collection.mel_frequencies,
                    mel=Config().collection.mel_frequencies is not None,
                )

                # take only the first minute of the song
                spectrogram = spectrogram[:40]
                self.predictInputCache[sourceFile] = spectrogram

            spectrogramThirds = tf.split(spectrogram, 3, axis=2)

            predictedSpectrogram = tf.concat(
                list(
                    map(
                        lambda spectrogram: self.modelGeneratorRickMorty.predict(
                            spectrogram, batch_size=Config().training.batch_size
                        ),
                        spectrogramThirds,
                    )
                ),
                axis=2,
            )

            shape = predictedSpectrogram.shape

            predictedSpectrogram = tf.transpose(
                tf.reshape(
                    tf.transpose(tf.squeeze(predictedSpectrogram), [0, 2, 1]),
                    [shape[0] * shape[2], shape[1]],
                )
            )

            denormalized_dbscale = (
                ((tf.clip_by_value(predictedSpectrogram, -1.0, 1.0) + 1.0) / 2.0)
                * -self.min_level_db
            ) + self.min_level_db
            denormalized_dbscale = tf.add(denormalized_dbscale, self.reference_db_level)

            path = f"{Path(predictedFile.parent, Config().name)}_{self.nextEpoch}_{predictedFile.stem}"
            dataset = tf.data.Dataset.from_tensor_slices(denormalized_dbscale)
            tf.data.experimental.save(
                dataset,
                path,
            )
            Logger.log(f"Saved predicted spectrogram to {path}")

            return path

    def summary(self):
        super()._summary(
            {
                "Generator": self.modelGeneratorRickMorty,
                "Discriminator": self.modelDiscriminatorMorty,
                "Siamese": self.modelSiamese,
            }
        )

    def _build(self):
        self.modelGeneratorRickMorty = Generator(
            self.frequencies,
            self.sliceWidth,
            Config().training.model.generator.filters,
            Config().training.model.generator.unetBlocks,
        ).build()

        self.modelDiscriminatorMorty = Discriminator(
            self.frequencies,
            self.sliceWidth,
            Config().training.model.discriminator.filters,
        ).build()

        self.modelSiamese = Siamese(
            self.frequencies,
            self.sliceWidth,
            Config().training.model.siamese.filters,
            Config().training.model.siamese.latentDim,
        ).build()

    def _load(self) -> bool:
        try:
            self.nextEpoch = self.__loadModels()
            Logger.log(f"Loaded pre-existing model.")
            return True
        except:
            return False

    def __loadDataset(self, path: str):
        return tf.data.experimental.load(
            str(Path(Config().collection.baseFolder, path)),
            tf.TensorSpec((self.frequencies, self.sliceWidthDataset)),
            compression=None,
            reader_func=None,
        )
