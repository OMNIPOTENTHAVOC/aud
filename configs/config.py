from pathlib import Path
import torch

# -----------------------
# Dataset Paths
# -----------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_ROOT = PROJECT_ROOT / "datasets" / "archive" / "LA" / "LA"

TRAIN_DIR = DATA_ROOT / "ASVspoof2019_LA_train"

DEV_DIR = DATA_ROOT / "ASVspoof2019_LA_dev"

EVAL_DIR = DATA_ROOT / "ASVspoof2019_LA_eval"

PROTOCOL_DIR = DATA_ROOT / "ASVspoof2019_LA_cm_protocols"


# -----------------------
# Audio Parameters
# -----------------------

SAMPLE_RATE = 16000

N_MELS = 128

N_FFT = 512

WIN_LENGTH = 400

HOP_LENGTH = 160

AUDIO_DURATION = 4

MAX_AUDIO_LENGTH = SAMPLE_RATE * AUDIO_DURATION


# -----------------------
# Training
# -----------------------

BATCH_SIZE = 32

LEARNING_RATE = 1e-4

NUM_EPOCHS = 30


# -----------------------
# Device
# -----------------------

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
