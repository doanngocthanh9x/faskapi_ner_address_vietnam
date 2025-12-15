"""
Simple test script for the API
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print("Health Check:", response.json())


def test_root():
    """Test root endpoint"""
    response = requests.get(f"{BASE_URL}/")
    print("Root:", response.json())


def test_predict():
    """Test prediction endpoint"""
    test_text = "Số 123 Đường Nguyễn Văn Linh, Quận 7, Thành phố Hồ Chí Minh"
    
    payload = {
        "text": test_text
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print("\nPrediction Result:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


def test_batch_predict():
    """Test batch prediction endpoint"""
    test_texts = [
        {"text": "456 Lê Lai, Phường 1, Quận 5, TP HCM"},
        {"text": "789 Trần Hưng Đạo, Quận 1, Hà Nội"}
    ]
    
    response = requests.post(f"{BASE_URL}/batch_predict", json=test_texts)
    print("\nBatch Prediction Results:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        test_health()
        test_root()
        
        # These will only work if model is loaded
        try:
            test_predict()
            test_batch_predict()
        except Exception as e:
            print(f"\nNote: Prediction tests require trained ONNX model: {e}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running!")
        print("Start the server with: python run.py")
