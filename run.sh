#!/bin/bash

# Script to setup and run the NER API

echo "==================================="
echo "NER Address Vietnam API Setup"
echo "==================================="

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download models
echo "Downloading models..."
python faskapi/download_model.py

# Run the API
echo "Starting API server..."
uvicorn faskapi.main:app --host 0.0.0.0 --port 8000 --reload
