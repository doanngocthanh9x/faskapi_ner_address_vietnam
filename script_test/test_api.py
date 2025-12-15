"""
Test script for NER API
"""
import requests
import json


# API endpoint
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")


def test_predict(text: str):
    """Test predict endpoint"""
    print(f"\n=== Testing Predict Endpoint ===")
    print(f"Input text: {text}")
    
    response = requests.post(
        f"{BASE_URL}/predict",
        json={"text": text}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")


def test_extract(text: str):
    """Test extract endpoint"""
    print(f"\n=== Testing Extract Endpoint ===")
    print(f"Input text: {text}")
    
    response = requests.post(
        f"{BASE_URL}/extract",
        json={"text": text}
    )
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    # Test examples
    test_texts = [
        "123 Đường Nguyễn Huệ, Phường Bến Nghé, Quận 1, Thành phố Hồ Chí Minh",
        "456 Lê Lợi, Phường 4, Quận 3, TP.HCM",
        "Số 10 Trần Hưng Đạo, Phường Cầu Ông Lãnh, Quận 1, Hồ Chí Minh",
    ]
    
    # Test health
    test_health()
    
    # Test predict
    for text in test_texts:
        test_predict(text)
    
    # Test extract
    for text in test_texts:
        test_extract(text)
