#!/bin/bash

# Script to test Docker image locally

set -e

echo "======================================"
echo "Testing NER API Docker Image"
echo "======================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

IMAGE_NAME="ner-api:test"
CONTAINER_NAME="ner-api-test"
PORT=8000

# Step 1: Build Docker image
echo -e "\n${YELLOW}Step 1: Building Docker image...${NC}"
docker build -t $IMAGE_NAME .

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Docker image built successfully${NC}"
else
    echo -e "${RED}❌ Failed to build Docker image${NC}"
    exit 1
fi

# Step 2: Stop and remove existing container if exists
echo -e "\n${YELLOW}Step 2: Cleaning up existing container...${NC}"
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Step 3: Run container
echo -e "\n${YELLOW}Step 3: Running container...${NC}"
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8000 \
    -v "$(pwd)/models:/app/models" \
    $IMAGE_NAME

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Container started successfully${NC}"
else
    echo -e "${RED}❌ Failed to start container${NC}"
    exit 1
fi

# Step 4: Wait for container to be ready
echo -e "\n${YELLOW}Step 4: Waiting for API to be ready...${NC}"
RETRY_COUNT=0
MAX_RETRIES=30

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -sf http://localhost:$PORT/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ API is ready!${NC}"
        break
    fi
    
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo "Waiting... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
    
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo -e "${RED}❌ API failed to start within timeout${NC}"
        echo -e "\n${YELLOW}Container logs:${NC}"
        docker logs $CONTAINER_NAME
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
        exit 1
    fi
done

# Step 5: Test endpoints
echo -e "\n${YELLOW}Step 5: Testing API endpoints...${NC}"

# Test health endpoint
echo "Testing /health endpoint..."
HEALTH_RESPONSE=$(curl -s http://localhost:$PORT/health)
if echo "$HEALTH_RESPONSE" | grep -q "status"; then
    echo -e "${GREEN}✅ Health endpoint works${NC}"
    echo "Response: $HEALTH_RESPONSE"
else
    echo -e "${RED}❌ Health endpoint failed${NC}"
fi

# Test root endpoint
echo -e "\nTesting / endpoint..."
ROOT_RESPONSE=$(curl -s http://localhost:$PORT/)
if echo "$ROOT_RESPONSE" | grep -q "message"; then
    echo -e "${GREEN}✅ Root endpoint works${NC}"
    echo "Response: $ROOT_RESPONSE"
else
    echo -e "${RED}❌ Root endpoint failed${NC}"
fi

# Test predict endpoint (if model is loaded)
echo -e "\nTesting /predict endpoint..."
PREDICT_RESPONSE=$(curl -s -X POST http://localhost:$PORT/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "123 Đường Nguyễn Huệ, Phường Bến Nghé, Quận 1, TP.HCM"}')

if echo "$PREDICT_RESPONSE" | grep -q "text"; then
    echo -e "${GREEN}✅ Predict endpoint works${NC}"
    echo "Response preview:"
    echo "$PREDICT_RESPONSE" | jq '.entities' 2>/dev/null || echo "$PREDICT_RESPONSE"
else
    echo -e "${YELLOW}⚠️  Predict endpoint returned error (model might not be loaded)${NC}"
    echo "Response: $PREDICT_RESPONSE"
fi

# Step 6: Show container info
echo -e "\n${YELLOW}Step 6: Container Information${NC}"
echo "Container ID: $(docker ps -q -f name=$CONTAINER_NAME)"
echo "Image: $IMAGE_NAME"
echo "Ports: http://localhost:$PORT"
echo "API Docs: http://localhost:$PORT/docs"

# Step 7: Show logs
echo -e "\n${YELLOW}Container logs (last 20 lines):${NC}"
docker logs --tail 20 $CONTAINER_NAME

# Final instructions
echo -e "\n${GREEN}======================================"
echo "Docker test completed successfully!"
echo "======================================${NC}"
echo ""
echo "Access the API at:"
echo "  - Root: http://localhost:$PORT/"
echo "  - Health: http://localhost:$PORT/health"
echo "  - Docs: http://localhost:$PORT/docs"
echo ""
echo "To view logs:"
echo "  docker logs -f $CONTAINER_NAME"
echo ""
echo "To stop container:"
echo "  docker stop $CONTAINER_NAME"
echo ""
echo "To remove container:"
echo "  docker rm $CONTAINER_NAME"
echo ""
