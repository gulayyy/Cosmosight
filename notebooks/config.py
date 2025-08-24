IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20

DATASET_DIR = "../data/split_dataset"
TRAIN_DIR   = f"{DATASET_DIR}/train"
VAL_DIR     = f"{DATASET_DIR}/val"
TEST_DIR    = f"{DATASET_DIR}/test"

MODEL_SAVE_PATH   = "models/astro_model.h5"
CLASS_NAMES_PATH  = "models/class_names.json"
HISTORY_PATH      = "models/training_history.npy"
