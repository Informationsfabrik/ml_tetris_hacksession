from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

from config import MODEL_MINIMUM_CONFIDENCE
from TetrisGame import TetrisGame

COMMANDS = ["left", "turn", "right", None]


class KerasModelTetris(TetrisGame):
    def __init__(self, path_to_model: Path) -> None:
        self.path_to_model = path_to_model

        self.model = load_model(path_to_model)
        self.input_shape = tuple(self.model.input.shape[1:3])

        with open(path_to_model.parent / "labels.txt", "r") as fp:
            self.labels = [line.split(" ")[-1].strip() for line in fp.readlines()]

    def detect_command_from_key_or_image(
        self, frame: np.ndarray, bbox_array: np.ndarray, keyboard_event: str | None
    ) -> Tuple[str | None, str, np.ndarray]:

        data = np.ndarray(
            shape=(1, self.input_shape[0], self.input_shape[1], 3), dtype=np.float32
        )

        # center crop and resize
        image = ImageOps.fit(
            Image.fromarray(frame, mode="RGB"),
            self.input_shape,
            Image.ANTIALIAS,
        )
        image_array = np.asarray(image)
        image_array = image_array[:, :, ::-1]  # BGR -> RGB

        normalized_image_array = image_array.astype(np.float32) / 127.0 - 1
        data[0] = normalized_image_array
        model_prediction = self.model.predict(data)[0]
        # print(model_prediction)
        if any(model_prediction > MODEL_MINIMUM_CONFIDENCE):
            command_index = np.argmax(model_prediction)
            command = COMMANDS[command_index]
            class_name = self.labels[command_index]
        else:
            command = None
            class_name = "No detection"

        return command, class_name, bbox_array


# from PIL import Image
# import numpy as np

# rotate_img = Image.open("data/train_data/rotate/00008.png")
# rotate_img_arr = np.asarray(rotate_img)

# machine = TeachableMachineTetris(path_to_model=Path("data/models/simple_tf/tf_classification.h5"))

# result = machine.detect_command_from_image(rotate_img_arr, None)
