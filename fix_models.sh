#!/bin/bash
# Script to fix and re-download models

echo "Cleaning up incomplete model files..."
cd /workspaces/faskapi_ner_address_vietnam

# Remove incomplete downloads
rm -f models/*.tmp
rm -f models/*.crdownload

echo ""
echo "Re-downloading all model files..."
python faskapi/download_model.py

echo ""
echo "Checking downloaded files..."
ls -lh models/

echo ""
echo "Required files check:"
echo "✓ ner_address_model_final.onnx: $([ -f models/ner_address_model_final.onnx ] && echo 'EXISTS' || echo 'MISSING')"
echo "✓ ner_address_model_final.onnx.data: $([ -f models/ner_address_model_final.onnx.data ] && echo 'EXISTS' || echo 'MISSING')"
echo "✓ config.json: $([ -f models/config.json ] && echo 'EXISTS' || echo 'MISSING')"
echo "✓ tokenizer.json: $([ -f models/tokenizer.json ] && echo 'EXISTS' || echo 'MISSING')"
