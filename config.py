import os

# Configuration file for Astronomical Image Classification Project
# Absolute paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model hyperparameters
IMAGE_SIZE = (224, 224)  # Input image dimensions for MobileNetV2
BATCH_SIZE = 32          # Training batch size
EPOCHS = 20              # Maximum training epochs

# Dataset directory structure
DATASET_DIR = os.path.join(BASE_DIR, "data", "split_dataset")
TRAIN_DIR   = os.path.join(DATASET_DIR, "train")
VAL_DIR     = os.path.join(DATASET_DIR, "val") 
TEST_DIR    = os.path.join(DATASET_DIR, "test")

# Model and output paths
MODEL_SAVE_PATH   = os.path.join(BASE_DIR, "models", "astro_model_improved.keras")
CLASS_NAMES_PATH  = os.path.join(BASE_DIR, "models", "class_names.json")
HISTORY_PATH      = os.path.join(BASE_DIR, "models", "improved_training_history.npy")
