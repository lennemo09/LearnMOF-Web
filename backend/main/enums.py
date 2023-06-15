from enum import Enum, auto, unique


@unique
class BaseEnum(str, Enum):
    @staticmethod
    def _generate_next_value_(name: str, *_):
        """
        Automatically generate values for enum.
        Enum values are lower-cased enum member names.
        """
        return name.lower()

    @classmethod
    def get_values(cls) -> list[str]:
        # noinspection PyUnresolvedReferences
        return [m.value for m in cls]


class Label(BaseEnum):
    CRYSTAL = auto()
    CHALLENGING_CRYSTAL = auto()
    NON_CRYSTAL = auto()


class_names = {0: "challenging-crystal", 1: "crystal", 2: "non-crystal"}

MODEL_NAME = "../LearnMOF_MODEL_BATCH5_tested_acc95"
ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png", "gif")

MODEL_DIR = "."
RESULTS_DIR = "results"
MODEL_PATH = MODEL_DIR + "/" + MODEL_NAME

UPLOAD_FOLDER = "static/images"


SEED = 1337
IMG_SIZE = 512

BATCH_SIZE = 4  # Number of images per batch to load into memory, since we are testing only a couple of images, 1 is enough
NUM_WORKERS = 1  # Number of CPU cores to load images

WELLS_PER_ROW = 4
