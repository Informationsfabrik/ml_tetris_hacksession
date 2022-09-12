"""Load and execute model from the teachablemachine"""

from os import PathLike
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras import Model
from tensorflow.keras.models import load_model


def predict_single_image(
    img_or_path: Union[str, PathLike, Image.Image] = Path("data/images/tasse.jpg"),
    model_or_path: Union[str, PathLike, Model] = Path(
        "data/models/teachablemachine/keras_model.h5"
    ),
    input_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Load a keras image classification model and apply it to an image.

    If given an image path, will load it using pillow.
    You can also provide a pillow image directly.
    You need to provde an image with 3 color channels in this order: RGB
    Use PIL.Image.open(..) to load an image yourself.

    If given a model path, will load it from there.
    You can also provide a keras model directly.
    You load_model(..) from tensorflow.keras.models to load a model yourself.

    Will return an array of probabilities.

    Args:
        img_or_path (str | PathLike | Image.Image, optional): The path to an image or a pillow image. Defaults to Path("data/images/tasse.jpg").
        model_or_path (str | PathLike | Model, optional): The path to a keras model or the model itself. Defaults to Path("data/models/teachablemachine/keras_model.h5").
        input_size (Tuple[int, int] | None): the image input size of the model. Will try to infer it from the model if given None. Defaults to None.

    Returns:
        np.ndarray: array of probabilities.
    """

    if isinstance(img_or_path, (str, PathLike)):
        image = Image.open(img_or_path)
    else:
        image = img_or_path

    # Load the model
    model = load_model(model_or_path)

    if input_size is None:
        input_size = tuple(model.input.shape[1:3])

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, input_size[0], input_size[1], 3), dtype=np.float32)

    # resize the image to the input size, usually 224x224, with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    image = ImageOps.fit(image, input_size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    return model.predict(data)[0]


if __name__ == "__main__":
    predict_single_image()
