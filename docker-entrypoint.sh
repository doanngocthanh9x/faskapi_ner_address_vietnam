#!/bin/bash
set -e

echo "=========================================="
echo "Starting NER Address Vietnam API"
echo "=========================================="

# Check if all required model files exist
ONNX_FILE="/app/models/ner_address_model_final.onnx"
DATA_FILE="/app/models/ner_address_model_final.onnx.data"
CONFIG_FILE="/app/models/config.json"

if [ ! -f "$ONNX_FILE" ] || [ ! -f "$DATA_FILE" ] || [ ! -f "$CONFIG_FILE" ]; then
    echo "Model files not found. Downloading from Google Drive..."
    echo "This will take 2-3 minutes (downloading ~1.5 GB)..."
    echo ""
    
    python faskapi/download_model.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Model downloaded successfully!"
        
        # Verify files
        if [ -f "$ONNX_FILE" ] && [ -f "$DATA_FILE" ]; then
            echo "✅ All required files present"
        else
            echo "⚠️  Some files may be missing:"
            [ ! -f "$ONNX_FILE" ] && echo "  - Missing: ner_address_model_final.onnx"
            [ ! -f "$DATA_FILE" ] && echo "  - Missing: ner_address_model_final.onnx.data"
            [ ! -f "$CONFIG_FILE" ] && echo "  - Missing: config.json"
        fi
    else
        echo "⚠️  Model download failed. API will start but predictions won't work."
        echo "Please download manually from:"
        echo "https://drive.google.com/drive/folders/1U3Kb-Nmv_8dXfLu_GF6w7KZXHQ7KC43P"
    fi
else
    echo "✅ All model files already exist:"
    echo "  - ner_address_model_final.onnx"
    echo "  - ner_address_model_final.onnx.data"
    echo "  - config.json"
fi

echo ""
echo "Starting FastAPI server..."
echo "=========================================="

# Execute the main command
exec "$@"
