import logging
import random
import shutil

import cv2
import numpy as np
import PySimpleGUI as sg
import questionary
from sklearn.model_selection import train_test_split

from config import (
    CAMERA_DEVICE_ID,
    HEIGHT,
    LEFT_KEY,
    MOVEMENTS,
    OTHER_KEY,
    RIGHT_KEY,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    TURN_KEY,
    VALIDATION_DATA_PATH,
    WIDTH,
)
from gui_common import boolean_question

logger = logging.getLogger(__name__)


def run_data_generator():
    logger.info("START DATA GENERATOR")
    # Create folder structure
    for path in [TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH]:
        for move in MOVEMENTS:
            target_path = path / move
            if target_path.exists():
                shutil.rmtree(target_path)
            target_path.mkdir(parents=True)
    left_images = []
    right_images = []
    turn_images = []
    other_images = []
    frame_counter = 0
    cap = cv2.VideoCapture(CAMERA_DEVICE_ID, cv2.CAP_DSHOW)
    ret, _ = cap.read()
    y0 = 10
    while ret:
        ret, frame = cap.read()
        y = frame.shape[0]
        x = frame.shape[1]
        min_width_height = min(y, x)
        startx = x // 2 - min_width_height // 2
        starty = y // 2 - min_width_height // 2
        frame = frame[
            starty : starty + min_width_height, startx : startx + min_width_height
        ]

        frame = cv2.resize(frame, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        white_canvas = np.zeros((int(HEIGHT / 2.5), WIDTH, 3), np.uint8)
        white_canvas = 255 - white_canvas
        INSTRUCTIONS = f'Instructions: \n  Press and hold the "l" key to save images \n   -> Left movement class {len(left_images)}/400\n Press and hold the "r" key to save images \n   -> Right movement class {len(right_images)}/400\n Press and hold the "t" key to save images \n   -> Turn movement class {len(turn_images)}/400\n Press and hold the "a" key to save images \n   -> No movement class {len(other_images)}/400\n Press "esc" to persist the images.'

        for i, line in enumerate(INSTRUCTIONS.split("\n")):
            dy = (
                cv2.getTextSize(
                    line, fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=0.8, thickness=1
                )[0][1]
                + 10
            )

            y = y0 + i * dy
            white_canvas = cv2.putText(
                white_canvas,
                text=line,
                org=(0, y),
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=0.8,
                color=(0, 0, 0),
                thickness=1,
                lineType=1,
            )

        stacked_frame = np.vstack((white_canvas, frame))
        cv2.imshow("WHITE", stacked_frame)
        key = cv2.waitKey(75)
        if key == 27:
            break
        elif key == LEFT_KEY:
            if frame_counter % 2 == 0 and len(left_images) < 400:
                left_images.append(frame)
            frame_counter += 1
        elif key == RIGHT_KEY:
            if frame_counter % 2 == 0 and len(right_images) < 400:
                right_images.append(frame)
            frame_counter += 1
        elif key == TURN_KEY:
            if frame_counter % 2 == 0 and len(turn_images) < 400:
                turn_images.append(frame)
            frame_counter += 1
        elif key == OTHER_KEY:
            if frame_counter % 2 == 0 and len(other_images) < 400:
                other_images.append(frame)
            frame_counter += 1
        else:
            frame_counter = 0
    for name, images in zip(
        MOVEMENTS, [left_images, turn_images, right_images, other_images]
    ):
        X_train, X_test = train_test_split(images, test_size=0.2, random_state=42)
        X_train, X_val = train_test_split(images, test_size=0.2, random_state=42)
        list_of_datasets = [
            (TRAIN_DATA_PATH, X_train),
            (VALIDATION_DATA_PATH, X_val),
            (TEST_DATA_PATH, X_test),
        ]

        for path, image_list in list_of_datasets:
            for idx, image in enumerate(image_list):
                cv2.imwrite(f"{str(path / name)}/{idx:05d}.png", image)
    cap.release()
    cv2.destroyAllWindows()


def data_generator_cli():
    if not TRAIN_DATA_PATH.is_dir():
        run_data_generator()
    elif boolean_question("Do you want to create train and test data?"):
        run_data_generator()


if __name__ == "__main__":
    data_generator_cli()
