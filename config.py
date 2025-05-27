# config.py
import torch

# --- Device Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model & VLM Configuration ---
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
VLM_MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct" # Or your specific VLM model
VLM_MIN_PIXELS = 128 * 28 * 28
VLM_MAX_PIXELS = 640 * 28 * 28

# --- Training Hyperparameters ---
BATCH_SIZE = 4
BOTTLENECK_DIM = 192  # For adapters
LEARNING_RATE = 3e-4
NUM_EPOCHS = 5
ALPHA = 0.2  # Weight for visual adapter
BETA = 0.2   # Weight for text adapter
GAMMA = 0.3  # Weight for context adapter

# --- Dataset Configuration ---
# Update these paths to your actual data directories
BASE_DATA_DIR = "/kaggle/working/RAFDB" # Example, adjust as needed
TRAIN_DIR = f"{BASE_DATA_DIR}/train"
TEST_DIR = f"{BASE_DATA_DIR}/test"

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# --- Evaluation ---
NUM_SAMPLES_TO_DISPLAY_PER_CLASS = 3