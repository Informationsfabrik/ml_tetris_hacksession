from os.path import exists
from typing import Dict

import cv2
import numpy as np
import PySimpleGUI as sg
import yaml
from screeninfo import get_monitors

from config import CAMERA_DEVICE_ID, COLOR_CONFIG_PATH
from gui_common import move_center


def read_color_config() -> Dict[str, np.array]:

    if not exists(COLOR_CONFIG_PATH):
        init_color_config()

    with open(COLOR_CONFIG_PATH, "r") as file:
        color_config = yaml.safe_load(file)

    # convert to np arrays
    color_config_as_np_arrays = {
        key: np.array(hsv) for key, hsv in color_config.items()
    }

    return color_config_as_np_arrays


def write_color_config(hsv_colors: Dict[str, np.array]):
    with open(COLOR_CONFIG_PATH, "w") as file:
        yaml.dump(hsv_colors, file)


def init_color_config():
    default_config = {
        "blue_lower": [0, 0, 0],
        "blue_upper": [179, 255, 255],
        "pink_lower": [0, 0, 0],
        "pink_upper": [179, 255, 255],
        "green_lower": [0, 0, 0],
        "green_upper": [179, 255, 255],
    }
    write_color_config(default_config)


def nothing(x):
    pass


def hsv_calc(title: str, lower: np.array, upper: np.array):
    slider = sg.Column(
        [
            [
                sg.Frame(
                    "LH",
                    [
                        [
                            sg.Slider(
                                range=(0, 179),
                                orientation="h",
                                default_value=lower[0],
                                key="lh",
                            )
                        ]
                    ],
                ),
            ],
            [
                sg.Frame(
                    "LS",
                    [
                        [
                            sg.Slider(
                                range=(0, 255),
                                orientation="h",
                                default_value=lower[1],
                                key="ls",
                            )
                        ]
                    ],
                ),
            ],
            [
                sg.Frame(
                    "LV",
                    [
                        [
                            sg.Slider(
                                range=(0, 255),
                                orientation="h",
                                default_value=lower[2],
                                key="lv",
                            )
                        ]
                    ],
                ),
            ],
            [
                sg.Frame(
                    "UH",
                    [
                        [
                            sg.Slider(
                                range=(0, 179),
                                orientation="h",
                                default_value=upper[0],
                                key="uh",
                            )
                        ]
                    ],
                ),
            ],
            [
                sg.Frame(
                    "US",
                    [
                        [
                            sg.Slider(
                                range=(0, 255),
                                orientation="h",
                                default_value=upper[1],
                                key="us",
                            )
                        ]
                    ],
                ),
            ],
            [
                sg.Frame(
                    "UV",
                    [
                        [
                            sg.Slider(
                                range=(0, 255),
                                orientation="h",
                                default_value=upper[2],
                                key="uv",
                            )
                        ]
                    ],
                ),
            ],
        ],
        element_justification="c",
    )

    bla = sg.Frame("Customize HSV Values", [[sg.Push(), slider, sg.Push()]])

    input_video_elem = sg.Frame("Input Video", [[sg.Image(filename="", key="video")]])

    mask_elem = sg.Frame("Mask", [[sg.Image(filename="", key="mask")]])

    result_elem = sg.Frame("Result", [[sg.Image(filename="", key="result")]])

    layout = [
        [sg.Text(f"{title}", size=(40, 1), font=("Any 15"), justification="center")],
        [bla, input_video_elem],
        [[mask_elem, result_elem]],
        [[sg.Push(), sg.Button("next step")]],
    ]

    window = sg.Window(f"{title}", layout, location=(800, 400), finalize=True)
    move_center(window)

    cap = cv2.VideoCapture(CAMERA_DEVICE_ID)

    print(title)

    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height

    first_run = 0

    while True:
        event, values = window.read(timeout=20)

        if event == "Exit" or event == sg.WIN_CLOSED:
            break

        ret, frame = cap.read()
        height, width = frame.shape[:2]
        scale_factor = screen_height / height
        height = (height * scale_factor) / 3
        width = (width * scale_factor) / 3

        frame = cv2.resize(frame, (int(width), int(height)))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lh = values["lh"]
        ls = values["ls"]
        lv = values["lv"]
        uh = values["uh"]
        us = values["us"]
        uv = values["uv"]

        l_blue = np.array([lh, ls, lv])
        u_blue = np.array([uh, us, uv])
        mask = cv2.inRange(hsv, l_blue, u_blue)
        result = cv2.bitwise_or(frame, frame, mask=mask)

        # frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_BGRA2BGR)
        # result_bgr = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)

        videobytes = cv2.imencode(".ppm", frame[:, :, :3])[1].tobytes()
        maskbytes = cv2.imencode(".ppm", mask_bgr)[1].tobytes()
        resultbytes = cv2.imencode(".ppm", result[:, :, :3])[1].tobytes()

        window["video"].update(data=videobytes)
        window["mask"].update(data=maskbytes)
        window["result"].update(data=resultbytes)

        # key = cv2.waitKey(1)

        if first_run < 2:
            move_center(window)
            first_run += 1

        if event == "next step":
            break

    cap.release()
    window.Close()

    lower = [lh, ls, lv]
    upper = [uh, us, uv]

    return lower, upper


def display_disclaimer():
    text = "You will now be asked to configure three colors one by one. Each time a window will open up. Configure the sliders until ONLY the color you are trying to configure is recognized in the camera windows."

    layout_column = [
        [
            sg.Text(
                text,
                font=("Any 15"),
                size=(40, None),
                justification="center",
                key="text",
            )
        ],
        [sg.Button("Okay", size=(40, 1))],
    ]
    layout = [[sg.Column(layout_column, element_justification="center")]]

    window = sg.Window("Disclaimer", layout, location=(800, 400), finalize=True)
    move_center(window)

    while True:
        event, values = window.read(timeout=20)

        if event == "Exit" or event == sg.WIN_CLOSED:
            window.Close()
            return

        if event == "Okay":
            window.Close()
            return


def configure_colors():
    color_config = read_color_config()

    display_disclaimer()

    blue_lower, blue_upper = hsv_calc(
        "Configure blue", color_config["blue_lower"], color_config["blue_upper"]
    )

    pink_lower, pink_upper = hsv_calc(
        "Configure pink", color_config["pink_lower"], color_config["pink_upper"]
    )
    green_lower, green_upper = hsv_calc(
        "Configure green", color_config["green_lower"], color_config["green_upper"]
    )

    hsv_colors_new = {
        "blue_lower": blue_lower,
        "blue_upper": blue_upper,
        "pink_lower": pink_lower,
        "pink_upper": pink_upper,
        "green_lower": green_lower,
        "green_upper": green_upper,
    }

    write_color_config(hsv_colors_new)
