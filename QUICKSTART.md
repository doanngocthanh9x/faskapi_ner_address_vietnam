# Quick Start Guide

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Running the API (without model)

The API can run without a trained model for testing:

```bash
python run.py
```

Visit: http://localhost:8000/docs

## Training and Using a Model

### Step 1: Download the dataset

```bash
python scripts/download_dataset.py
```

### Step 2: Train and convert to ONNX

```bash
python scripts/train_and_convert.py
```

This will create `models/model.onnx`

### Step 3: Restart the API

```bash
python run.py
```

Now the prediction endpoints will work!

## Testing the API

```bash
# In another terminal
python test_api.py
```

Or use curl:

```bash
# Health check
curl http://localhost:8000/health

# Predict
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Số 123 Đường Nguyễn Văn Linh, Quận 7, TP HCM"}'
```

## Docker Deployment

```bash
# Build and run
docker-compose up --build
```

## Troubleshooting

**Problem**: "Model not loaded" error
**Solution**: Train the model first using `python scripts/train_and_convert.py`

**Problem**: Import errors
**Solution**: Install dependencies: `pip install -r requirements.txt`

**Problem**: Port 8000 already in use
**Solution**: Change port in run.py or use: `uvicorn app.main:app --port 8001`
