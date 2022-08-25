from pathlib import Path

MOVEMENTS = ["left", "rotate", "right", "other"]

# game
SPEED = 1.0
MODEL_MINIMUM_CONFIDENCE = 0.8

# data generator
WIDTH = 480
HEIGHT = 480
LEFT_KEY = 108
RIGHT_KEY = 114
TURN_KEY = 116
OTHER_KEY = 97
SPLIT_SIZE = 0.1

# model input size
FRAME_WIDTH = 224
FRAME_HEIGHT = 224

# training parameters
TF_CLASSIFICATION_TRAIN_EPOCHS = 25
LEARNING_RATE = 0.001
BATCH_SIZE = 16
USE_PRETRAINED_MOBILE_NET = False

# data placement
COLOR_CONFIG_PATH = "data/color_config.yaml"
COLOR_DETECTION_MIN_AREA = 5000
MUSIC_BASE_PATH = "data/music/"
MUSIC_ON = False
DATA_BASE_PATH = Path("data")
TRAIN_DATA_PATH = DATA_BASE_PATH / "train_data"
VALIDATION_DATA_PATH = DATA_BASE_PATH / "validation_data"
TEST_DATA_PATH = DATA_BASE_PATH / "test_data"
MODELS_BASE_PATH = DATA_BASE_PATH / "models"
TF_SIMPLE_CLASSIFICATION_MODEL_PATH = (
    MODELS_BASE_PATH / "simple_tf" / "tf_classification.h5"
)

# camera settings
CAMERA_DEVICE_ID = 0
