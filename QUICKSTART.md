# Quick Start Guide

## 🚀 Using Docker (Recommended)

### Option 1: Pull from Docker Hub (Easiest)

```bash
# Pull image
docker pull doanngocthanh9x/faskapi_ner_address_vietnam:latest

# Create models directory
mkdir -p models

# Run container (models will be downloaded automatically on first start)
docker run -d \
  --name ner-api \
  -p 5000:8000 \
  -v $(pwd)/models:/app/models \
  doanngocthanh9x/faskapi_ner_address_vietnam:latest

# View logs
docker logs -f ner-api

# Access API
open http://localhost:5000/docs
```

### Option 2: Pull from GitHub Container Registry

```bash
# Pull image
docker pull ghcr.io/doanngocthanh9x/faskapi_ner_address_vietnam:latest

# Run container
docker run -d \
  --name ner-api \
  -p 5000:8000 \
  -v $(pwd)/models:/app/models \
  ghcr.io/doanngocthanh9x/faskapi_ner_address_vietnam:latest
```

### Option 3: Build from Source

```bash
# Clone repository
git clone https://github.com/doanngocthanh9x/faskapi_ner_address_vietnam.git
cd faskapi_ner_address_vietnam

# Build and run with Docker Compose
docker-compose up -d

# Access API
open http://localhost:5000/docs
```

## 📦 What Happens on First Start?

1. Container starts
2. **Entrypoint script checks for model files**
3. If not found, **automatically downloads from Google Drive**
4. Model is cached in mounted volume
5. FastAPI server starts
6. Ready to use! 🎉

**Note:** First start takes ~2-3 minutes to download model files (~1.4 GB).

## 🔍 Check Status

```bash
# Check health
curl http://localhost:5000/health

# View logs
docker logs ner-api

# Follow logs
docker logs -f ner-api
```

## 🧪 Test API

```bash
# Test predict endpoint
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "123 Đường Nguyễn Huệ, Phường Bến Nghé, Quận 1, TP.HCM"
  }'

# Test extract endpoint
curl -X POST http://localhost:5000/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "456 Lê Lợi, Phường 4, Quận 3, Hồ Chí Minh"
  }'
```

## 🛑 Stop & Clean Up

```bash
# Stop container
docker stop ner-api

# Remove container
docker rm ner-api

# Remove image (optional)
docker rmi doanngocthanh9x/faskapi_ner_address_vietnam:latest
```

## 🐍 Using Python Directly

```bash
# Clone repository
git clone https://github.com/doanngocthanh9x/faskapi_ner_address_vietnam.git
cd faskapi_ner_address_vietnam

# Install dependencies
pip install -r requirements.txt

# Download models
python faskapi/download_model.py

# Run server
uvicorn faskapi.main:app --host 0.0.0.0 --port 8000 --reload

# Access API
open http://localhost:8000/docs
```

## 📚 API Documentation

Once running, access:
- **Swagger UI**: http://localhost:5000/docs
- **ReDoc**: http://localhost:5000/redoc
- **OpenAPI JSON**: http://localhost:5000/openapi.json

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Welcome message |
| `/health` | GET | Health check |
| `/predict` | POST | Get tokens with labels and entities |
| `/extract` | POST | Get only extracted entities |

## 🔧 Troubleshooting

### Models Not Downloading

If automatic download fails, download manually:

```bash
# Create models directory
mkdir -p models

# Run download script
python faskapi/download_model.py

# Or download from Google Drive manually:
# https://drive.google.com/file/d/19wXYDJytJor4i5C_E4Q19aR4bDz5xd87/view
# https://drive.google.com/drive/folders/1U3Kb-Nmv_8dXfLu_GF6w7KZXHQ7KC43P
```

### Container Won't Start

```bash
# Check logs
docker logs ner-api

# Run interactively for debugging
docker run -it --rm \
  -v $(pwd)/models:/app/models \
  doanngocthanh9x/faskapi_ner_address_vietnam:latest \
  bash
```

### Port Already in Use

```bash
# Use different port
docker run -d -p 8080:8000 doanngocthanh9x/faskapi_ner_address_vietnam:latest

# Access at http://localhost:8080
```

## 💡 Tips

- **Persistent Models**: Always use `-v $(pwd)/models:/app/models` to keep models across container restarts
- **First Time**: Allow 2-3 minutes for model download on first start
- **Updates**: Pull latest image regularly: `docker pull doanngocthanh9x/faskapi_ner_address_vietnam:latest`
- **Production**: Use docker-compose for easier management

## 📞 Support

For issues or questions:
- GitHub Issues: https://github.com/doanngocthanh9x/faskapi_ner_address_vietnam/issues
- Check logs: `docker logs ner-api`
