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
    
    # Check if critical files exist
    model_onnx_exists = config.MODEL_FILE.exists()
    model_data_exists = (config.MODEL_DIR / "ner_address_model_final.onnx.data").exists()
    config_exists = (config.MODEL_DIR / "config.json").exists()
    
    if model_onnx_exists and model_data_exists and config_exists:
        print("All model files already exist!")
        print(f"  - {config.MODEL_FILE}")
        print(f"  - {config.MODEL_DIR / 'ner_address_model_final.onnx.data'}")
        print(f"  - {config.MODEL_DIR / 'config.json'}")
        return
    
    print("Downloading model files from Google Drive folder...")
    print("This may take 2-3 minutes (downloading ~1.5 GB)...")
    
    try:
        # Download entire folder which contains all necessary files
        download_folder_from_google_drive(
            config.GOOGLE_DRIVE_FOLDER_ID,
            str(config.MODEL_DIR)
        )
        
        # Verify critical files
        if not config.MODEL_FILE.exists():
            print(f"⚠️  Warning: {config.MODEL_FILE} not found after download")
        if not (config.MODEL_DIR / "ner_address_model_final.onnx.data").exists():
            print(f"⚠️  Warning: ner_address_model_final.onnx.data not found after download")
        if not (config.MODEL_DIR / "config.json").exists():
            print(f"⚠️  Warning: config.json not found after download")
            
        print("\n✅ Model download completed!")
        print(f"Model files location: {config.MODEL_DIR}")
        
        # List downloaded files
        print("\nDownloaded files:")
        for file in sorted(config.MODEL_DIR.glob("*")):
            if file.is_file():
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")
                
    except Exception as e:
        print(f"\n❌ Error downloading from Google Drive: {e}")
        print("\n📥 Please download manually:")
        print(f"1. Go to: https://drive.google.com/drive/folders/{config.GOOGLE_DRIVE_FOLDER_ID}")
        print(f"2. Download all files to: {config.MODEL_DIR}")
        print("\nRequired files:")
        print("  - ner_address_model_final.onnx")
        print("  - ner_address_model_final.onnx.data")
        print("  - config.json")
        print("  - tokenizer.json")
        print("  - vocab.txt")
        print("  - tokenizer_config.json")
        print("  - special_tokens_map.json")


if __name__ == "__main__":
    download_models()
