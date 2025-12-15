#!/bin/bash
set -e

echo "=========================================="
echo "Starting NER Address Vietnam API"
echo "=========================================="

# Check if model exists
if [ ! -f "/app/models/ner_address_model_final.onnx" ]; then
    echo "Model not found. Downloading from Google Drive..."
    python faskapi/download_model.py
    
    if [ $? -eq 0 ]; then
        echo "✅ Model downloaded successfully!"
    else
        echo "⚠️  Model download failed. API will start but predictions won't work."
        echo "Please mount models volume or download manually."
    fi
else
    echo "✅ Model already exists at /app/models/ner_address_model_final.onnx"
fi

echo ""
echo "Starting FastAPI server..."
echo "=========================================="

# Execute the main command
exec "$@"
