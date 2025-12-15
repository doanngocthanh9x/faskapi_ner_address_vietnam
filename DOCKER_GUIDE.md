# Docker Deployment Guide

## 🐳 Quick Start

### Build and Test Locally

```bash
# Make script executable
chmod +x test_docker.sh

# Run test script
./test_docker.sh
```

### Manual Build and Run

```bash
# Build image
docker build -t ner-api:latest .

# Run container with model volume
docker run -d \
  --name ner-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  ner-api:latest

# View logs
docker logs -f ner-api

# Test API
curl http://localhost:8000/health
```

## 🚀 GitHub Container Registry (GHCR)

### Setup Secrets

**GitHub Token** được tự động cung cấp bởi GitHub Actions, không cần setup thêm!

### Auto Deploy

Workflow sẽ tự động chạy khi:
- Push lên branch `main` hoặc `develop`
- Tạo tag với format `v*.*.*` (vd: v1.0.0)
- Manual trigger từ Actions tab

### Pull Image

**From GitHub Container Registry:**
```bash
# Public repo
docker pull ghcr.io/doanngocthanh9x/faskapi_ner_address_vietnam:latest

# Private repo - cần đăng nhập trước
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin
docker pull ghcr.io/doanngocthanh9x/faskapi_ner_address_vietnam:latest
```

### Make Package Public (Optional)

Sau khi image được push lần đầu:
1. Đi tới: https://github.com/doanngocthanh9x?tab=packages
2. Click vào package `faskapi_ner_address_vietnam`
3. Package settings > Change visibility > Public

### Run Pulled Image

```bash
# Create models directory
mkdir -p models

# Download model files
python faskapi/download_model.py

# Run container
docker run -d \
  --name ner-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  ghcr.io/doanngocthanh9x/faskapi_ner_address_vietnam:latest

# Access API
open http://localhost:8000/docs
```

## 🔧 Docker Compose

```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📊 Container Management

### View running containers
```bash
docker ps
```

### Stop container
```bash
docker stop ner-api
```

### Remove container
```bash
docker rm ner-api
```

### Remove image
```bash
docker rmi ner-api:latest
```

### Clean up
```bash
# Remove stopped containers
docker container prune

# Remove unused images
docker image prune -a
```

## 🏷️ Image Tags

GitHub Actions tự động tạo các tags:

- `latest` - Latest build từ main branch
- `main` - Build từ main branch
- `develop` - Build từ develop branch
- `v1.0.0` - Version tags
- `v1.0` - Major.minor tags
- `v1` - Major version tags
- `main-abc1234` - Commit SHA tags

## 🌐 Environment Variables

```bash
docker run -d \
  --name ner-api \
  -p 8000:8000 \
  -e PYTHONUNBUFFERED=1 \
  -e MAX_LENGTH=128 \
  -v $(pwd)/models:/app/models \
  ner-api:latest
```

## 🔍 Debugging

### Check container status
```bash
docker inspect ner-api
```

### Execute command in container
```bash
docker exec -it ner-api bash
```

### View logs
```bash
docker logs ner-api
docker logs -f ner-api  # Follow logs
docker logs --tail 100 ner-api  # Last 100 lines
```

### Test inside container
```bash
docker exec -it ner-api python -c "import onnxruntime; print(onnxruntime.__version__)"
```

## 🛠️ Troubleshooting

### Port already in use
```bash
# Find process using port 8000
lsof -i :8000

# Or use different port
docker run -d -p 8080:8000 ner-api:latest
```

### Model not found
```bash
# Ensure models are downloaded
python faskapi/download_model.py

# Check models directory
ls -la models/

# Mount models correctly
docker run -d -v $(pwd)/models:/app/models ner-api:latest
```

### Container crashes
```bash
# Check logs
docker logs ner-api

# Run interactively
docker run -it --rm ner-api:latest bash
```

## 📈 Performance

### CPU Optimization
```bash
docker run -d \
  --cpus="2.0" \
  --memory="2g" \
  ner-api:latest
```

### Multi-platform Build
```bash
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t ner-api:latest \
  --push .
```

## 🔒 Security

### Run as non-root user
Already configured in Dockerfile with Python image defaults.

### Scan for vulnerabilities
```bash
docker scan ner-api:latest
```

### Use specific Python version
Update Dockerfile: `FROM python:3.10.12-slim`
