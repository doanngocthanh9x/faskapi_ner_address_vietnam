# Examples

This directory contains example scripts demonstrating how to use the Vietnamese Address NER API.

## Available Examples

### example_usage.py

A comprehensive Python script showing:
- How to check API health status
- Single address prediction
- Batch prediction of multiple addresses

**Usage:**

```bash
# Make sure the API server is running first
python run.py

# In another terminal, run the example
python examples/example_usage.py
```

## Creating Your Own Examples

Here's a minimal example to get you started:

```python
import requests

# Make a prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Số 123 Đường ABC, Quận 1, TP HCM"}
)

result = response.json()
print(result)
```

## API Endpoints

- `GET /` - Root endpoint with basic info
- `GET /health` - Health check
- `POST /predict` - Single text prediction
- `POST /batch_predict` - Batch prediction
- `GET /docs` - Interactive API documentation (Swagger UI)
- `GET /redoc` - Alternative API documentation (ReDoc)
