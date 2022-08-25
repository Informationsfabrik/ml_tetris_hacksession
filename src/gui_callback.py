import threading

import keras
import PySimpleGUI as sg

from config import BATCH_SIZE, LEARNING_RATE, TF_CLASSIFICATION_TRAIN_EPOCHS


class GUICallback(keras.callbacks.Callback):
    def __init__(self):
        self.accuracy = 0
        self.loss = 0
        self.batch = 1
        self.epoch = 1
        self.num_of_epochs = TF_CLASSIFICATION_TRAIN_EPOCHS
        self.window = None
        self.display_train_info()

    def display_train_info(self):
        text_epochs = sg.Text(
            f"Epoch: {self.epoch}/{self.num_of_epochs}",
            font=("Any 15"),
            justification="center",
            key="epoch",
        )
        text_accuracy = sg.Text(
            f"Accuracy: {self.accuracy}",
            font=("Any 15"),
            justification="center",
            key="accuracy",
        )
        text_loss = sg.Text(
            f"loss: {self.loss}", font=("Any 15"), justification="center", key="loss"
        )
        text_batch = sg.Text(
            f"batch: {self.batch}", font=("Any 15"), justification="center", key="batch"
        )
        layout = [
            [text_epochs],
            [text_accuracy, text_loss],
            [text_batch],
            [
                sg.ProgressBar(
                    max_value=80, orientation="h", size=(20, 20), key="progress"
                )
            ],
        ]

        self.window = sg.Window(
            f"Training of model", layout, location=(800, 400), element_justification="c"
        )

    def update_window(self):
        event, values = self.window.read(timeout=1)

        if event == sg.WIN_CLOSED or event == "Exit":
            exit()

        self.window["epoch"].update(f"Epoch: {self.epoch}/{self.num_of_epochs}")
        self.window["accuracy"].update(f'Accuracy: {"%.4f"%self.accuracy}')
        self.window["loss"].update(f'loss: {"%.4f"%self.loss}')
        self.window["batch"].update(f"batch: {self.batch}")
        self.window["progress"].update(self.batch)

    def on_train_begin(self, logs=None):
        if logs:
            keys = list(logs.keys())
            # print("\n ____________ \n Starting training; got log keys: {}".format(keys))

    def on_train_end(self, logs=None):
        if logs:
            if self.window:
                self.window.close()
            keys = list(logs.keys())
            # print("\n ____________ \n Stop training; got log keys: {}".format(keys))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        if logs:
            self.accuracy = logs.get("accuracy", self.accuracy)
            self.loss = logs.get("loss", self.loss)

            if self.window:
                self.update_window()
            # keys = list(logs.keys())
            # print("\n ____________ \n Start epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        if logs:
            self.accuracy = logs.get("accuracy", self.accuracy)
            self.loss = logs.get("loss", self.loss)

            if self.window:
                self.update_window()
            # keys = list(logs.keys())
            # print("\n ____________ \n End epoch {} of training; got log keys: {}".format(epoch, keys))

    def on_train_batch_begin(self, batch, logs=None):
        self.batch = batch
        if logs:
            self.accuracy = logs.get("accuracy", self.accuracy)
            self.loss = logs.get("loss", self.loss)

            if self.window:
                self.update_window()

    def on_train_batch_end(self, batch, logs=None):
        self.batch = batch
        if logs:
            self.accuracy = logs.get("accuracy", self.accuracy)
            self.loss = logs.get("loss", self.loss)

            if self.window:
                self.update_window()
