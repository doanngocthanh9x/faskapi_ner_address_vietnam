# NER Address Vietnam API

API FastAPI sử dụng ONNX Runtime để thực hiện Named Entity Recognition (NER) cho địa chỉ Việt Nam.

Dataset: 🤗 [dathuynh1108/ner-address-standard-dataset](https://huggingface.co/datasets/dathuynh1108/ner-address-standard-dataset)

## 🌟 Tính năng

- ⚡ **FastAPI** - API hiệu năng cao với tài liệu tự động
- 🚀 **ONNX Runtime** - Inference nhanh với ONNX
- 🏷️ **NER cho địa chỉ Việt Nam** - Trích xuất các thực thể:
  - Số nhà (NUMBER)
  - Đường (STREET)
  - Phường/Xã (WARD)
  - Quận/Huyện (DISTRICT)
  - Thành phố/Tỉnh (CITY)
- 🐳 **Docker Support** - Dễ dàng deploy với Docker
- 📝 **API Documentation** - Tự động tạo docs với Swagger UI

## 📋 Yêu cầu

- Python 3.10+
- pip
- Docker (tùy chọn)

## 🚀 Cài đặt

### Cách 1: Cài đặt thủ công

1. **Clone repository**
```bash
git clone https://github.com/doanngocthanh9x/faskapi_ner_address_vietnam.git
cd faskapi_ner_address_vietnam
```

2. **Tạo virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows
```

3. **Cài đặt dependencies**
```bash
pip install -r requirements.txt
```

4. **Tải model từ Google Drive**

Model files được lưu trữ trên Google Drive:
- Model file: https://drive.google.com/file/d/19wXYDJytJor4i5C_E4Q19aR4bDz5xd87/view?usp=drive_link
- Tokenizer và các file khác: https://drive.google.com/drive/folders/1U3Kb-Nmv_8dXfLu_GF6w7KZXHQ7KC43P

Chạy script tự động tải model:
```bash
python faskapi/download_model.py
```

5. **Chạy API server**
```bash
uvicorn faskapi.main:app --host 0.0.0.0 --port 8000 --reload
```

### Cách 2: Sử dụng script tự động

```bash
chmod +x run.sh
./run.sh
```

### Cách 3: Sử dụng Docker

1. **Build và chạy với Docker Compose**
```bash
# Tải model trước
python faskapi/download_model.py

# Build và chạy
docker-compose up -d
```

2. **Hoặc sử dụng Docker thủ công**
```bash
# Build image
docker build -t ner-address-vietnam .

# Chạy container
docker run -d -p 8000:8000 -v $(pwd)/models:/app/models ner-address-vietnam
```

## 📖 Sử dụng API

### 1. Truy cập API Documentation

Sau khi chạy server, mở trình duyệt và truy cập:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 2. API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Predict - Lấy tokens và labels
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "123 Đường Nguyễn Huệ, Phường Bến Nghé, Quận 1, Thành phố Hồ Chí Minh"}'
```

Kết quả:
```json
{
  "text": "123 Đường Nguyễn Huệ, Phường Bến Nghé, Quận 1, Thành phố Hồ Chí Minh",
  "tokens": [
    {"token": "123", "label": "B-NUMBER"},
    {"token": "Đường", "label": "O"},
    {"token": "Nguyễn", "label": "B-STREET"},
    {"token": "Huệ", "label": "I-STREET"},
    ...
  ],
  "entities": {
    "STREET": ["Nguyễn Huệ"],
    "WARD": ["Bến Nghé"],
    "DISTRICT": ["Quận 1"],
    "CITY": ["Thành phố Hồ Chí Minh"],
    "NUMBER": ["123"]
  }
}
```

#### Extract - Chỉ lấy entities
```bash
curl -X POST "http://localhost:8000/extract" \
  -H "Content-Type: application/json" \
  -d '{"text": "456 Lê Lợi, Phường 4, Quận 3, TP.HCM"}'
```

Kết quả:
```json
{
  "text": "456 Lê Lợi, Phường 4, Quận 3, TP.HCM",
  "entities": {
    "STREET": ["Lê Lợi"],
    "WARD": ["Phường 4"],
    "DISTRICT": ["Quận 3"],
    "CITY": ["TP.HCM"],
    "NUMBER": ["456"]
  }
}
```

### 3. Test API với Python

```python
import requests

# Test predict
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "123 Đường Nguyễn Huệ, Phường Bến Nghé, Quận 1, TP.HCM"}
)
print(response.json())
```

Hoặc sử dụng test script có sẵn:
```bash
python test_api.py
```

## 📁 Cấu trúc thư mục

```
faskapi_ner_address_vietnam/
├── faskapi/                    # Thư mục chính của API
│   ├── __init__.py            # Package init
│   ├── main.py                # FastAPI application
│   ├── config.py              # Configuration
│   ├── schemas.py             # Pydantic models
│   ├── ner_model.py           # ONNX NER model
│   └── download_model.py      # Script tải model
├── models/                     # Thư mục chứa model (tạo tự động)
│   ├── model.onnx             # ONNX model file
│   └── tokenizer/             # Tokenizer files
├── docker/                     # Docker related files
├── Dockerfile                  # Docker build file
├── docker-compose.yml          # Docker compose config
├── requirements.txt            # Python dependencies
├── run.sh                      # Script chạy tự động
├── test_api.py                # Test script
├── .gitignore                 # Git ignore
└── README.md                   # Documentation

```

## 🔧 Configuration

Các cấu hình có thể được thay đổi trong [faskapi/config.py](faskapi/config.py):

- `MAX_LENGTH`: Độ dài tối đa của input (mặc định: 128)
- `MODEL_DIR`: Thư mục chứa model
- Google Drive IDs cho model files

## 📊 Entity Labels

API hỗ trợ các loại entity sau:

| Label | Mô tả | Ví dụ |
|-------|-------|-------|
| NUMBER | Số nhà | 123, 456A |
| STREET | Tên đường | Nguyễn Huệ, Lê Lợi |
| WARD | Phường/Xã | Bến Nghé, Phường 4 |
| DISTRICT | Quận/Huyện | Quận 1, Quận 3 |
| CITY | Thành phố/Tỉnh | TP.HCM, Hà Nội |

## 🐛 Troubleshooting

### Model không tải được

Nếu gặp lỗi khi tải model:
1. Kiểm tra kết nối internet
2. Tải model thủ công từ Google Drive links
3. Đặt file model vào thư mục `models/`

### API không khởi động

1. Kiểm tra port 8000 có bị chiếm không
2. Kiểm tra đã cài đặt đủ dependencies
3. Xem logs để biết lỗi cụ thể

### ONNX Runtime lỗi

Nếu gặp lỗi với ONNX Runtime, cài đặt lại:
```bash
pip uninstall onnxruntime
pip install onnxruntime
```

## 📝 License

[MIT License](LICENSE)

## 👤 Author

**Doan Ngoc Thanh**
- GitHub: [@doanngocthanh9x](https://github.com/doanngocthanh9x)

## 🤝 Contributing

Contributions, issues và feature requests đều được chào đón!

## ⭐ Show your support

Nếu project này hữu ích, hãy cho một ⭐️!

## 📧 Contact

Nếu có câu hỏi, vui lòng tạo issue trên GitHub.
