#!/bin/bash
# Simple Docker test script

set -e

echo "=========================================="
echo "Step 1: Building Docker image..."
echo "=========================================="
docker build -t ner-api:test .

echo ""
echo "=========================================="
echo "Step 2: Starting container..."
echo "=========================================="

# Stop existing container if any
docker stop ner-api-test 2>/dev/null || true
docker rm ner-api-test 2>/dev/null || true

# Run container
docker run -d \
  --name ner-api-test \
  -p 8001:8000 \
  -v "$(pwd)/models:/app/models" \
  ner-api:test

echo "Container started!"
echo ""
echo "Waiting for API to be ready (this may take 10-15 seconds)..."
sleep 5

# Wait for health check
for i in {1..20}; do
  if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
    echo "✅ API is ready!"
    break
  fi
  echo "Waiting... ($i/20)"
  sleep 2
done

echo ""
echo "=========================================="
echo "Testing Endpoints"
echo "=========================================="

echo ""
echo "1. Health Check:"
curl -s http://localhost:8001/health | jq '.'

echo ""
echo "2. Root Endpoint:"
curl -s http://localhost:8001/ | jq '.'

echo ""
echo "3. Predict Endpoint:"
curl -s -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "123 Đường Nguyễn Huệ, Phường Bến Nghé, Quận 1, TP.HCM"}' \
  | jq '.entities'

echo ""
echo "=========================================="
echo "Container Info"
echo "=========================================="
echo "Container ID: $(docker ps -q -f name=ner-api-test)"
echo "Access API at: http://localhost:8001"
echo "API Docs: http://localhost:8001/docs"
echo ""
echo "To view logs:"
echo "  docker logs -f ner-api-test"
echo ""
echo "To stop:"
echo "  docker stop ner-api-test && docker rm ner-api-test"
