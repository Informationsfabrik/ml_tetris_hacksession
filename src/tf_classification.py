from os import PathLike
from pickle import FALSE
from typing import List, Tuple

import numpy as np
import questionary
import tensorflow as tf
from keras_preprocessing.image import DirectoryIterator, ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from config import (
    BATCH_SIZE,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    LEARNING_RATE,
    MOVEMENTS,
    TEST_DATA_PATH,
    TF_CLASSIFICATION_TRAIN_EPOCHS,
    TF_SIMPLE_CLASSIFICATION_MODEL_PATH,
    TRAIN_DATA_PATH,
    USE_PRETRAINED_MOBILE_NET,
    VALIDATION_DATA_PATH,
)
from gui_callback import GUICallback
from gui_common import boolean_question


def prep_fn(img):
    # center crop and normalize
    y = img.shape[0]
    x = img.shape[1]
    min_width_height = min(y, x)
    startx = x // 2 - (min_width_height // 2)
    starty = y // 2 - (min_width_height // 2)
    img = img[starty : starty + min_width_height, startx : startx + min_width_height]
    img = (img.astype(np.float32) / 127.0) - 1
    return img


def create_data_generators() -> Tuple[
    DirectoryIterator, DirectoryIterator, DirectoryIterator
]:
    """Create the datasets generators for the training from the folders that were created using data_generator.py

    Returns:
        Tuple[DirectoryIterator, DirectoryIterator, DirectoryIterator]: an tuple object that contains
        the train, validation and test dataset generators.

    """
    train_generator = ImageDataGenerator(
        preprocessing_function=prep_fn,
        rotation_range=5,
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # shear_range=0.05,
        zoom_range=0.05,
        # horizontal_flip=False,
        fill_mode="nearest",
    ).flow_from_directory(
        directory=TRAIN_DATA_PATH,
        target_size=(FRAME_HEIGHT, FRAME_WIDTH),
        class_mode="categorical",
        seed=42,
        batch_size=BATCH_SIZE,
        classes=MOVEMENTS,
    )

    validation_generator = ImageDataGenerator(
        preprocessing_function=prep_fn
    ).flow_from_directory(
        directory=VALIDATION_DATA_PATH,
        target_size=(FRAME_HEIGHT, FRAME_WIDTH),
        class_mode="categorical",
        seed=42,
        batch_size=BATCH_SIZE,
        classes=MOVEMENTS,
    )

    test_generator = ImageDataGenerator(
        preprocessing_function=prep_fn
    ).flow_from_directory(
        directory=TEST_DATA_PATH,
        target_size=(FRAME_HEIGHT, FRAME_WIDTH),
        class_mode="categorical",
        shuffle=False,
        batch_size=BATCH_SIZE,
        classes=MOVEMENTS,
    )

    return train_generator, validation_generator, test_generator


def create_model(class_names: List[str]) -> tf.keras.Model:
    """Create a new Keras model.

    Args:
        class_names (List[str]): the list of class names

    Returns:
        tf.keras.Model: a keras model
    """
    # determine the number of classes. This is the number of output neurons that our model needs to provide.
    num_classes = len(class_names)

    if USE_PRETRAINED_MOBILE_NET is False:

        # we create a Keras Sequential model, which is a sequence of layers.
        # the output of each layer is the input for the next one.
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    filters=16,
                    kernel_size=3,
                    padding="same",
                    activation="relu",
                    input_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3),
                ),  # a convolutional layer that applies multiple "filters" to our image, creating new features
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(
                    filters=32,
                    kernel_size=3,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(
                    filters=64,
                    kernel_size=3,
                    padding="same",
                    activation="relu",
                ),
                tf.keras.layers.MaxPooling2D(),  # a pooling layer that "summarizes" the features created by the convolutional layer
                tf.keras.layers.Dropout(
                    0.1
                ),  # the dropout layer randomly hides neuron outputs during training for the next layer, making it more robust
                tf.keras.layers.Flatten(),  # flattens the 3d output of MaxPooling2D into a 1d vector
                tf.keras.layers.Dense(
                    128, activation="relu"
                ),  # a fully connected layer that learns the connection between our image features and our classes
                tf.keras.layers.Dense(
                    num_classes, activation="softmax"
                ),  # a softmax layer that outputs relative confidences for each class
            ]
        )

    else:
        model = tf.keras.Sequential(
            [
                tf.keras.applications.MobileNetV2(
                    input_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3), include_top=False
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        model.layers[0].trainable = False

    model.summary()

    return model


def train_model(
    train_generator: DirectoryIterator,
    validation_generator: DirectoryIterator,
    model: tf.keras.Model,
) -> tf.keras.callbacks.History:
    """Train a model on the train and validation dataset.

    Args:
        train_generator (DirectoryIterator):  the train dataset from create_data_generators()
        validation_generator (DirectoryIterator): the valid dataset from create_data_generators()
        model (tf.keras.Model): the model to train from create_model()

    Returns:
        tf.keras.callbacks.History: the history containing accuracy and loss values per epoch
    """
    gui_c = GUICallback()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            min_delta=0.0001,
            patience=10,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_accuracy",
            factor=0.1,
            patience=5,
            verbose=0,
            mode="auto",
            min_delta=0.0001,
            cooldown=0,
            min_lr=0,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            TF_SIMPLE_CLASSIFICATION_MODEL_PATH,
            save_best_only=True,
            monitor="val_accuracy",
        ),
        gui_c,
    ]

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model.fit(
        train_generator,
        epochs=TF_CLASSIFICATION_TRAIN_EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
    )


def visualize_training_results(history: tf.keras.callbacks.History) -> None:
    """Open a window with two plots showing the accuracy and loss for the train and validation set per epoch.

    Args:
        history (tf.keras.callbacks.History): the history object that is returned by the fit()-function.
    """
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(len(history.epoch))

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Training Accuracy")
    plt.plot(epochs_range, val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Training Loss")
    plt.plot(epochs_range, val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    plt.show()


def evaluate_on_test_set(
    model: tf.keras.Model, test_generator: DirectoryIterator
) -> float:
    """Use the test_generator generator on the model, print out the confusion matrix and balanced accuracy score.
    Return the score.

    Args:
        model (tf.keras.Model): the trained model
         test_generator: DirectoryIterator: the test data generator

    Returns:
        float: balanced accuracy score for the test set
    """

    prediction = model.predict(test_generator)

    print(f"Class order: {MOVEMENTS}")

    print("\nConfusion Matrix:")
    y_pred = prediction.argmax(axis=1)
    y_true = test_generator.labels

    print(
        "i-th row and j-th column entry indicates the number of samples with true label being i-th class and "
        "predicted label being j-th class. "
    )
    print(confusion_matrix(y_pred=y_pred, y_true=y_true))

    print("\nBalanced Accuracy:")
    bal_acc = balanced_accuracy_score(y_pred=y_pred, y_true=y_true)
    print(bal_acc)

    return bal_acc


def generate_model(train_result_viz: bool = True) -> float:
    train_generator, validation_generator, test_generator = create_data_generators()

    class_names: List[str] = list(train_generator.class_indices.keys())
    print(f"Class names: {class_names}")

    model = create_model(class_names=class_names)

    history = train_model(
        train_generator=train_generator,
        validation_generator=validation_generator,
        model=model,
    )

    if train_result_viz:
        visualize_training_results(history=history)

    return evaluate_on_test_set(model, test_generator)


def custom_model_cli():
    if not TF_SIMPLE_CLASSIFICATION_MODEL_PATH.is_file():
        generate_model()
    elif boolean_question("Do you want to retrain your custom ML model?"):
        generate_model()


if __name__ == "__main__":
    custom_model_cli()
