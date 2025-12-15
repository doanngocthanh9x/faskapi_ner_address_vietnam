"""
Configuration file for the application
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "model.onnx"))
TOKENIZER_NAME = os.getenv("TOKENIZER_NAME", "vinai/phobert-base")

# API settings
API_TITLE = "Vietnamese Address NER API"
API_DESCRIPTION = "Named Entity Recognition for Vietnamese addresses using ONNX Runtime"
API_VERSION = "1.0.0"
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ONNX Runtime settings
ONNX_PROVIDERS = os.getenv("ONNX_PROVIDERS", "CPUExecutionProvider").split(",")

# Model settings
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", "256"))

# Label mapping
LABEL_MAP = {
    0: "O",
    1: "B-CITY",
    2: "I-CITY",
    3: "B-DISTRICT",
    4: "I-DISTRICT",
    5: "B-WARD",
    6: "I-WARD",
    7: "B-STREET",
    8: "I-STREET",
}
