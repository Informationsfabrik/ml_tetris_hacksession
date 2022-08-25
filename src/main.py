import threading
from pathlib import Path

import PySimpleGUI as sg
import questionary

import music
from ColorDetectionTetris import ColorDetectionTetris
from data_generator import data_generator_cli
from gui_common import move_center
from KerasModelTetris import KerasModelTetris
from KeyTetris import KeyTetris
from tf_classification import custom_model_cli


def run_cli():
    layout_column = [
        [
            sg.Text(
                "What do you want to play?",
                size=(40, 1),
                font=("Any 15"),
                justification="center",
            )
        ],
        # [sg.Button("Keyboard Tetris", size=(40, 1))],
        [sg.Button("Color detection Tetris", size=(40, 1))],
        [sg.Button("Teachable machine Tetris", size=(40, 1))],
        [sg.Button("Custom ML Tetris", size=(40, 1))],
    ]

    layout = [[sg.Column(layout_column, element_justification="center")]]

    window = sg.Window("Start your journey", layout, location=(800, 400), finalize=True)
    move_center(window)

    while True:
        event, _ = window.read(timeout=20)

        if event in ["Exit", sg.WIN_CLOSED]:
            exit(0)

        if event == "Keyboard Tetris":
            game = KeyTetris()
            break

        if event == "Color detection Tetris":
            game = ColorDetectionTetris()
            break

        if event == "Teachable machine Tetris":
            game = KerasModelTetris(
                path_to_model=Path("data/models/teachablemachine/keras_model.h5")
            )
            break

        if event == "Custom ML Tetris":
            data_generator_cli()
            custom_model_cli()
            game = KerasModelTetris(
                path_to_model=Path("data/models/simple_tf/tf_classification.h5")
            )
            break

    window.Close()
    game.start()


def play_bg_music():
    while True:
        music.play()


def start_bg_music():
    music_thread = threading.Thread(target=play_bg_music)
    music_thread.daemon = True
    music_thread.start()


if __name__ == "__main__":
    # start_bg_music()
    run_cli()
