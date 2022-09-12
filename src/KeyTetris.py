from typing import Optional, Tuple

import numpy as np

from config import MOVEMENTS
from TetrisGame import TetrisGame


class KeyTetris(TetrisGame):
    def detect_command_from_key_or_image(
        self, frame: np.ndarray, bbox_array: np.ndarray, keyboard_event: Optional[str]
    ) -> Tuple[str, str, np.ndarray]:

        if keyboard_event is not None:
            if keyboard_event.startswith("Left"):
                return "left", "Left", bbox_array
            elif keyboard_event.startswith("Right"):
                return "right", "Right", bbox_array
            elif keyboard_event.startswith("Up"):
                return "turn", "Turn", bbox_array

        return None, "None", bbox_array
