"""
Configuration file for the NER API
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model directory
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Model files
MODEL_FILE = MODEL_DIR / "ner_address_model_final.onnx"
TOKENIZER_DIR = MODEL_DIR

# Google Drive file IDs
GOOGLE_DRIVE_FILE_ID = "19wXYDJytJor4i5C_E4Q19aR4bDz5xd87"
GOOGLE_DRIVE_FOLDER_ID = "1U3Kb-Nmv_8dXfLu_GF6w7KZXHQ7KC43P"

# API Configuration
API_TITLE = "NER Address Vietnam API"
API_VERSION = "1.0.0"
API_DESCRIPTION = "Named Entity Recognition API for Vietnamese Addresses using ONNX Runtime"

# Model configuration
MAX_LENGTH = 128
