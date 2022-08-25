from typing import Tuple

import cv2
import numpy as np
import PySimpleGUI as sg

from config import CAMERA_DEVICE_ID, COLOR_DETECTION_MIN_AREA
from gui_common import boolean_question, move_center
from hsv_picker import configure_colors, read_color_config
from TetrisGame import TetrisGame


class ColorDetectionTetris(TetrisGame):
    def start(self):
        need_to_reconfigure = boolean_question(
            "Do you want to reconfigure the color settings?"
        )
        cam = cv2.VideoCapture(0)
        while need_to_reconfigure:
            cam.release()
            configure_colors()
            cam = cv2.VideoCapture(0)
            need_to_reconfigure = self.show_color_confirmation(cam)
        cam.release()
        super().start()

    def show_color_confirmation(self, cam: cv2.VideoCapture) -> None:
        input_video_elem = sg.Frame(
            "Color Detection", [[sg.Image(filename="", key="video")]]
        )

        layout = [
            [
                sg.Text(
                    "Check if the colors are detected properly",
                    size=(40, 1),
                    font=("Any 15"),
                    justification="center",
                )
            ],
            [input_video_elem],
            [
                sg.Text(
                    "Do you need to change your configuration?",
                    size=(40, 1),
                    font=("Any 13"),
                    justification="center",
                )
            ],
            [[sg.Push(), sg.Button("No"), sg.Button("Yes")]],
        ]

        window = sg.Window(
            "Configuration check", layout, location=(800, 400), finalize=True
        )

        first_run = 0

        ret, _ = cam.read()
        while not self.end_step() and ret:
            event, values = window.read(timeout=20)
            ret, frame = cam.read()

            bbox_array = self.get_command_and_overlay(frame)[1]
            result = self.combine_images(bbox_array, frame)

            resultbytes = cv2.imencode(".ppm", result[:, :, :3])[1].tobytes()
            window["video"].update(data=resultbytes)
            # cv2.imshow("Check if the colors are detected properly, press esc to exit", result)

            if first_run < 2:
                first_run += 1
                move_center(window)

            if event == "Exit" or event == sg.WIN_CLOSED:
                exit(0)

            if event == "Yes":
                window.Close()
                return True

            if event == "No":
                window.Close()
                return False

        # cv2.destroyAllWindows()
        window.Close()
        return False

    def detect_command_from_key_or_image(
        self, frame: np.ndarray, bbox_array: np.ndarray, keyboard_event: str | None
    ) -> Tuple[str, str, np.ndarray]:
        command = None
        height, width = frame.shape[:2]

        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv_img = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # create pink or blue color mask and remove noise by erosion and dialation
        pink_mask = ColorDetectionTetris._create_mask(color="pink", hsv_img=hsv_img)
        blue_mask = ColorDetectionTetris._create_mask(color="blue", hsv_img=hsv_img)
        green_mask = ColorDetectionTetris._create_mask(color="green", hsv_img=hsv_img)

        (
            cntrs,
            color,
            detected_area,
        ) = ColorDetectionTetris._grab_contours_and_detected_color(
            pink_mask, blue_mask, green_mask, width
        )

        if detected_area < COLOR_DETECTION_MIN_AREA:
            color = None
            command = None
        else:
            for cntr in cntrs:
                # get bounding boxes
                pad = 10
                x, y, w, h = cv2.boundingRect(cntr)
                # print(area_detected_object)

                if color == "blue":

                    cv2.rectangle(
                        bbox_array,
                        (x - pad, y - pad),
                        (x + w + pad, y + h + pad),
                        (245, 90, 65),
                        4,
                    )
                    command = "right"
                elif color == "pink":
                    cv2.rectangle(
                        bbox_array,
                        (x - pad, y - pad),
                        (x + w + pad, y + h + pad),
                        (212, 65, 245),
                        4,
                    )
                    command = "left"
                elif color == "green":
                    cv2.rectangle(
                        bbox_array,
                        (x - pad, y - pad),
                        (x + w + pad, y + h + pad),
                        (80, 150, 245),
                        4,
                    )
                    command = "turn"

        return command, color, bbox_array

    @staticmethod
    def _create_mask(color, hsv_img):

        color_config = read_color_config()

        if color == "pink":
            mask = cv2.inRange(
                hsv_img, color_config["pink_lower"], color_config["pink_upper"]
            )
        elif color == "blue":
            mask = cv2.inRange(
                hsv_img, color_config["blue_lower"], color_config["blue_upper"]
            )
        elif color == "green":
            mask = cv2.inRange(
                hsv_img, color_config["green_lower"], color_config["green_upper"]
            )
        else:
            raise Exception("Unknown color: " + color)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        return mask

    @staticmethod
    def _grab_contours_and_detected_color(pink_mask, blue_mask, green_mask, width):
        pink_object_mask = pink_mask[:, 0:width]
        blue_object_mask = blue_mask[:, 0:width]
        green_object_mask = green_mask[:, 0:width]
        pink_cnts = cv2.findContours(
            pink_object_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        blue_cnts = cv2.findContours(
            blue_object_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        green_cnts = cv2.findContours(
            green_object_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        pink_cnts = pink_cnts[0] if len(pink_cnts) == 2 else pink_cnts[1]
        blue_cnts = blue_cnts[0] if len(blue_cnts) == 2 else blue_cnts[1]
        green_cnts = green_cnts[0] if len(green_cnts) == 2 else green_cnts[1]

        sum_pink = sum(
            [ColorDetectionTetris.get_area(cv2.boundingRect(cnt)) for cnt in pink_cnts]
        )
        sum_blue = sum(
            [ColorDetectionTetris.get_area(cv2.boundingRect(cnt)) for cnt in blue_cnts]
        )
        sum_green = sum(
            [ColorDetectionTetris.get_area(cv2.boundingRect(cnt)) for cnt in green_cnts]
        )

        max_area = max(sum_pink, sum_blue, sum_green)

        if sum_pink == max_area:
            return pink_cnts, "pink", max_area
        elif sum_blue == max_area:
            return blue_cnts, "blue", max_area
        else:
            return green_cnts, "green", max_area

    @staticmethod
    def get_area(rect):
        return rect[2] * rect[3]
