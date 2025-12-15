"""
Script to download model files from Google Drive
"""
import gdown
import os
from pathlib import Path
import zipfile
try:
    from . import config
except ImportError:
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import config


def download_from_google_drive(file_id: str, output_path: str):
    """
    Download a file from Google Drive
    
    Args:
        file_id: Google Drive file ID
        output_path: Local path to save the file
    """
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading from Google Drive: {file_id}")
    gdown.download(url, output_path, quiet=False)
    print(f"Downloaded to: {output_path}")


def download_folder_from_google_drive(folder_id: str, output_dir: str):
    """
    Download a folder from Google Drive
    
    Args:
        folder_id: Google Drive folder ID
        output_dir: Local directory to save the files
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"Downloading folder from Google Drive: {folder_id}")
    gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)
    print(f"Downloaded folder to: {output_dir}")


def extract_zip(zip_path: str, extract_to: str):
    """
    Extract a zip file
    
    Args:
        zip_path: Path to the zip file
        extract_to: Directory to extract to
    """
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to: {extract_to}")


def download_models():
    """
    Download all necessary model files
    """
    # Create model directory if it doesn't exist
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download main model file
    if not config.MODEL_FILE.exists():
        print("Downloading main model file...")
        try:
            download_from_google_drive(
                config.GOOGLE_DRIVE_FILE_ID,
                str(config.MODEL_FILE)
            )
        except Exception as e:
            print(f"Error downloading model file: {e}")
            print("Trying alternative download method...")
            # If direct download fails, try downloading as zip
            zip_path = config.MODEL_DIR / "model.zip"
            download_from_google_drive(
                config.GOOGLE_DRIVE_FILE_ID,
                str(zip_path)
            )
            if zip_path.exists():
                extract_zip(str(zip_path), str(config.MODEL_DIR))
                os.remove(zip_path)
    else:
        print(f"Model file already exists: {config.MODEL_FILE}")
    
    # Download tokenizer and other files from folder
    if not config.TOKENIZER_DIR.exists():
        print("Downloading tokenizer files...")
        try:
            download_folder_from_google_drive(
                config.GOOGLE_DRIVE_FOLDER_ID,
                str(config.MODEL_DIR)
            )
        except Exception as e:
            print(f"Error downloading folder: {e}")
            print("Please download manually from:")
            print(f"https://drive.google.com/drive/folders/{config.GOOGLE_DRIVE_FOLDER_ID}")
    else:
        print(f"Tokenizer directory already exists: {config.TOKENIZER_DIR}")
    
    print("\nModel download completed!")
    print(f"Model files location: {config.MODEL_DIR}")


if __name__ == "__main__":
    download_models()
