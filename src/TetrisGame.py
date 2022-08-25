import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple

import cv2
import numpy as np
import PySimpleGUI as sg

import music
from config import CAMERA_DEVICE_ID, MUSIC_ON, SPEED
from gui_common import move_center
from tetris import Tetris, TetrisAction


class TetrisGame(ABC):
    """The abstract base class for all tetris games.
    The various input methods should subclass this one.
    The subclasses should overwrite detect_command_from_image().
    """

    def start(self):
        cam = cv2.VideoCapture(CAMERA_DEVICE_ID)
        self.play(cam)
        cam.release()
        cv2.destroyAllWindows()

    @classmethod
    def end_step(cls) -> bool:
        """Check if the escape key was pressed

        Returns:
            bool: True if the Escape-key has been pressed
        """
        escape_key = 27
        return cv2.waitKey(1) == escape_key

    def get_command_and_overlay(
        self,
        frame: Any,
        base_img: Optional[np.ndarray] = None,
        keyboard_event: str | None = None,
    ) -> Tuple[Optional[str], np.ndarray]:
        """Using the current webcam frame, get the command for that frame using the specific implementation
        (e.g. color-based or teachable machine).
        Draw the detection upon the base image and return it together with the detected command.

        Args:
            frame (Any): the webcam image
            base_img (Optional[np.ndarray], optional): the tetris game image or None. Defaults to None.
            keyboard_event (str | None): The keyboard event, if there is any. Defaults to None

        Returns:
            Tuple[Optional[str], np.ndarray]: the detected command and the resulting image
        """
        bbox_array = np.zeros([480, 640, 4], dtype=np.uint8)
        command, detected, bbox_array = self.detect_command_from_key_or_image(
            frame.copy(), bbox_array, keyboard_event
        )
        # command = None
        # detected = "other"
        # bbox_array = bbox_array

        if base_img is not None:
            tetris_img = np.dstack((base_img, np.zeros((440, 340))))
            bbox_array[:, :320] = cv2.resize(tetris_img, (320, 480))
        cv2.putText(
            img=bbox_array,
            text="Nothing detected"
            if detected is None
            else f"{detected}: {str(command)}",
            org=(470, 35),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=0.5,
            color=(255, 255, 255),
            thickness=1,
        )

        bbox_array[:, :, 3] = (bbox_array.max(axis=2) > 0).astype(int) * 255
        return command, bbox_array

    @classmethod
    def combine_images(cls, bbox_array: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Blend the bbox_array (which shoul contain the tetris overlay) and the frame (which should contain the webcam image)
        and blend them together.

        Args:
            bbox_array (np.ndarray): the tetris overlay image
            frame (np.ndarray): the webcam image

        Returns:
            np.ndarray: frame overlayed with bbox_array
        """
        # extract alpha channel from foreground image as mask and make 3 channels
        alpha = bbox_array[:, :, 3]
        alpha = cv2.merge([alpha, alpha, alpha])

        # extract bgr channels from foreground image
        front = bbox_array[:, :, 0:3]

        # blend the two images using the alpha channel as controlling mask
        return np.where(alpha == (0, 0, 0), frame, front)

    def play(self, cam: cv2.VideoCapture) -> None:
        """Create a new game of Tetris and create the window that will display the game along with the webcam image.
        Runs in a while-loop until the end step is reached (the game is finished).

        Args:
            cam (cv2.VideoCapture): the video capture that is used to get the images.

        Raises:
            IOError: when the video frame could not be fetched.
        """
        # setup the game
        tetris = Tetris()
        interval_update = SPEED
        last_update = time.time()
        input_video_elem = sg.Frame(
            "Tetris Game", [[sg.Image(filename="", key="video")]]
        )

        layout = [[input_video_elem]]
        window = sg.Window(
            "Enjoy your game",
            layout,
            location=(800, 400),
            finalize=True,
            return_keyboard_events=True,
        )

        num_frames = 0

        if MUSIC_ON:
            self.start_bg_music()

        # start the game loop
        print("Game started!")
        keyboard_event = None
        while not self.end_step():
            # handle UI events
            event, values = window.read(timeout=30)
            if event in ["Exit", sg.WIN_CLOSED]:
                exit(0)
            if event in ["Left:37", "Right:39", "Up:38", "Down:40"]:  # keyboard events
                keyboard_event = event

            # get a webcam frame
            _, frame = cam.read()
            if frame is None:
                raise IOError(
                    "Could not get image from webcam. Make sure that all programs that use the webcam are closed."
                )

            # determine command and draw it on top of the game canvas
            command, bbox_array = self.get_command_and_overlay(
                frame, tetris.canvas, keyboard_event
            )

            # convert the command to a key and update the game
            key = self.get_key_from_command(command)
            if time.time() - last_update > interval_update:
                tetris.advance_one_step(action=key)
                last_update = time.time()
                keyboard_event = None

            # draw the result image to the window
            result = self.combine_images(bbox_array, frame)

            # image = np.ascontiguousarray(result, dtype=np.uint8)
            image_bgr = result[:, :, :3]  # cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
            resultbytes = cv2.imencode(".ppm", image_bgr)[
                1
            ].tobytes()  # ppm more efficient than jpg
            window["video"].update(data=resultbytes)

            if num_frames < 2:
                num_frames += 1
                move_center(window)

    @classmethod
    def get_key_from_command(cls, command: TetrisAction) -> int:
        """Encode the command into a key.

        Args:
            command (TetrisAction): the command to encode

        Returns:
            int: the encoded command
        """
        key = TetrisAction.Nothing
        if command:
            if command == "right":
                key = TetrisAction.Right
            elif command == "left":
                key = TetrisAction.Left
            elif command == "turn":
                key = TetrisAction.RotateCounterclockwise
        return key

    @abstractmethod
    def detect_command_from_key_or_image(
        self, frame: np.ndarray, bbox_array: np.ndarray, keyboard_event: str | None
    ) -> Tuple[str, str, np.ndarray]:
        """Detects a user command from a given (camera) image.

        Args:
            frame (np.ndarray): image to inspect
            bbox_array (np.ndarray): transparent overlay that may be altered to
            show e.g. where an object was detected
            keyboard_event (str | None): the detected key event, if any
        Returns:
            Tuple[str, np.ndarray]: command, name of what was detected and (possibly altered) overlay
        """
        pass

    @classmethod
    def start_bg_music(cls):
        music_thread = threading.Thread(target=cls.play_bg_music)
        music_thread.daemon = True
        music_thread.start()

    @classmethod
    def play_bg_music(cls):
        while True:
            music.play()
