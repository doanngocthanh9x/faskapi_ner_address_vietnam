"""
Example usage of the Vietnamese Address NER API

This script demonstrates how to use the API from Python code.
Make sure the API server is running before executing this script.
"""

import requests
import json


# Configuration
API_URL = "http://localhost:8000"


def check_api_status():
    """Check if API is running and healthy"""
    try:
        response = requests.get(f"{API_URL}/health")
        health = response.json()
        print("✅ API Status:")
        print(f"   Status: {health.get('status')}")
        print(f"   Model Loaded: {health.get('model_loaded')}")
        return health.get('model_loaded', False)
    except Exception as e:
        print(f"❌ Failed to connect to API: {e}")
        print(f"   Make sure the server is running at {API_URL}")
        return False


def predict_address(text: str):
    """Predict entities in a Vietnamese address"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"text": text}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n📍 Text: {result['text']}")
            print("   Entities found:")
            for entity in result['entities']:
                print(f"   - {entity['entity']}: {entity['text']}")
            return result
        else:
            print(f"❌ Error: {response.json()}")
            return None
            
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return None


def batch_predict(addresses: list):
    """Predict entities for multiple addresses"""
    try:
        payload = [{"text": addr} for addr in addresses]
        response = requests.post(
            f"{API_URL}/batch_predict",
            json=payload
        )
        
        if response.status_code == 200:
            results = response.json()
            print("\n📦 Batch Prediction Results:")
            for i, result in enumerate(results, 1):
                print(f"\n   Address {i}: {result['text']}")
                print(f"   Entities: {len(result['entities'])}")
                for entity in result['entities']:
                    print(f"      - {entity['entity']}: {entity['text']}")
            return results
        else:
            print(f"❌ Error: {response.json()}")
            return None
            
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
        return None


def main():
    """Main function demonstrating API usage"""
    print("=" * 60)
    print("Vietnamese Address NER API - Example Usage")
    print("=" * 60)
    
    # Check API status
    model_loaded = check_api_status()
    
    if not model_loaded:
        print("\n⚠️  Warning: Model not loaded!")
        print("   The prediction examples below will not work.")
        print("   To train the model, run: python scripts/train_and_convert.py")
        return
    
    print("\n" + "=" * 60)
    print("Example 1: Single Address Prediction")
    print("=" * 60)
    
    address1 = "Số 123 Đường Nguyễn Văn Linh, Quận 7, Thành phố Hồ Chí Minh"
    predict_address(address1)
    
    print("\n" + "=" * 60)
    print("Example 2: Another Single Prediction")
    print("=" * 60)
    
    address2 = "456 Lê Lai, Phường 1, Quận 5, TP HCM"
    predict_address(address2)
    
    print("\n" + "=" * 60)
    print("Example 3: Batch Prediction")
    print("=" * 60)
    
    addresses = [
        "789 Trần Hưng Đạo, Quận 1, Hà Nội",
        "12 Hai Bà Trưng, Phường Bến Nghé, Quận 1, TP HCM",
        "34 Lê Duẩn, Quận 1, Hà Nội",
    ]
    batch_predict(addresses)
    
    print("\n" + "=" * 60)
    print("✅ Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
