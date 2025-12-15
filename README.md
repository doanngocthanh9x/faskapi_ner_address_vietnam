# FastAPI Vietnamese Address NER

Named Entity Recognition (NER) API for Vietnamese addresses using FastAPI and ONNX Runtime.

This project provides a high-performance REST API for extracting named entities from Vietnamese address text, powered by ONNX Runtime for efficient inference.

## Features

- 🚀 **FastAPI** - Modern, fast web framework for building APIs
- ⚡ **ONNX Runtime** - Optimized inference engine for production
- 🇻🇳 **Vietnamese NER** - Specialized for Vietnamese address extraction
- 📊 Dataset from 🤗 [dathuynh1108/ner-address-standard-dataset](https://huggingface.co/datasets/dathuynh1108/ner-address-standard-dataset)
- 🔄 Batch processing support
- 📝 Automatic API documentation (Swagger UI)

## Entity Types

The model recognizes the following address components:
- **CITY** - City/Province (Thành phố/Tỉnh)
- **DISTRICT** - District (Quận/Huyện)
- **WARD** - Ward/Commune (Phường/Xã)
- **STREET** - Street name (Đường/Phố)

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/doanngocthanh9x/faskapi_ner_address_vietnam.git
cd faskapi_ner_address_vietnam
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Train and convert model to ONNX (optional - see Training section):
```bash
python scripts/train_and_convert.py
```

## Usage

### Start the API Server

```bash
python run.py
```

Or using uvicorn directly:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Text Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Số 123 Đường Nguyễn Văn Linh, Quận 7, Thành phố Hồ Chí Minh"}'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '[
    {"text": "456 Lê Lai, Phường 1, Quận 5, TP HCM"},
    {"text": "789 Trần Hưng Đạo, Quận 1, Hà Nội"}
  ]'
```

### Python Client Example

```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Số 123 Đường Nguyễn Văn Linh, Quận 7, TP HCM"}
)
result = response.json()
print(result)
```

### Test Script

Run the included test script:
```bash
python test_api.py
```

## Training Your Own Model

### Option 1: Using the Training Script

The repository includes a complete training pipeline:

```bash
python scripts/train_and_convert.py
```

This script will:
1. Load the Vietnamese address NER dataset from Hugging Face
2. Train a transformer model (default: PhoBERT)
3. Convert the trained model to ONNX format
4. Save the ONNX model to `models/model.onnx`

### Option 2: Manual Training and Conversion

1. Train your PyTorch model:
```python
from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained(
    "vinai/phobert-base",
    num_labels=9
)
# ... train your model ...
```

2. Convert to ONNX:
```python
import torch

dummy_input = tokenizer("Sample text", return_tensors="pt", padding='max_length', max_length=128)

torch.onnx.export(
    model,
    (dummy_input['input_ids'], dummy_input['attention_mask']),
    "models/model.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence'},
        'attention_mask': {0: 'batch_size', 1: 'sequence'},
        'logits': {0: 'batch_size', 1: 'sequence'}
    },
    opset_version=14
)
```

## Project Structure

```
faskapi_ner_address_vietnam/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   └── onnx_predictor.py    # ONNX inference logic
├── scripts/
│   ├── __init__.py
│   └── train_and_convert.py # Model training and conversion
├── models/                   # ONNX models directory
│   └── model.onnx           # (generated after training)
├── requirements.txt          # Python dependencies
├── run.py                   # Server startup script
├── test_api.py              # API testing script
└── README.md                # This file
```

## Requirements

Key dependencies:
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `onnxruntime` - ONNX inference engine
- `transformers` - Model training and tokenization
- `torch` - PyTorch for training
- `pydantic` - Data validation

See `requirements.txt` for complete list.

## Performance

ONNX Runtime provides:
- ⚡ Faster inference compared to PyTorch
- 💾 Smaller model size
- 🔧 Better CPU utilization
- 🌐 Cross-platform compatibility

## Development

### Running in Development Mode

```bash
# With auto-reload
uvicorn app.main:app --reload

# With custom host and port
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

### Adding New Features

1. Model changes: Edit `app/onnx_predictor.py`
2. API endpoints: Edit `app/main.py`
3. Training logic: Edit `scripts/train_and_convert.py`

## Troubleshooting

### Model Not Found Error

If you see "Model file not found" error:
1. Train the model using `python scripts/train_and_convert.py`
2. Or place a pre-trained ONNX model at `models/model.onnx`

### Import Errors

Make sure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### Port Already in Use

Change the port in `run.py` or use:
```bash
uvicorn app.main:app --port 8001
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset: [dathuynh1108/ner-address-standard-dataset](https://huggingface.co/datasets/dathuynh1108/ner-address-standard-dataset)
- PhoBERT: Vietnamese language model by VinAI Research
- FastAPI: Modern web framework by Sebastián Ramírez
- ONNX Runtime: Microsoft's ML inference engine

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
