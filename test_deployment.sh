#!/bin/bash
# Test script for deployed DeepSeek-OCR service

# Configuration
SERVICE_URL="${1:-http://localhost:8765}"  # Pass URL as first argument or use localhost

echo "=================================="
echo "Testing DeepSeek-OCR Deployment"
echo "=================================="
echo "Service URL: $SERVICE_URL"
echo ""

# Test 1: Health Check
echo "Test 1: Health Check"
echo "------------------------------------"
curl -s "$SERVICE_URL/health" | jq . || curl -s "$SERVICE_URL/health"
echo -e "\n"

# Test 2: List Models
echo "Test 2: List Available Models"
echo "------------------------------------"
curl -s "$SERVICE_URL/v1/models" | jq . || curl -s "$SERVICE_URL/v1/models"
echo -e "\n"

# Test 3: Simple Text Extraction (if you have a test image)
if [ -f "bausch_BRW283A4D6DFC2A_20180728_013026_010420.pdf" ]; then
    echo "Test 3: OCR Text Extraction"
    echo "------------------------------------"
    echo "Converting PDF to base64..."

    # Convert first page of PDF to PNG, then to base64
    if command -v convert &> /dev/null; then
        convert -density 300 "bausch_BRW283A4D6DFC2A_20180728_013026_010420.pdf[0]" -quality 100 /tmp/test_prescription.png
        BASE64_IMG=$(base64 -i /tmp/test_prescription.png)

        echo "Sending OCR request..."

        curl -s -X POST "$SERVICE_URL/v1/chat/completions" \
          -H "Content-Type: application/json" \
          -d "{
            \"model\": \"deepseek-ai/DeepSeek-OCR\",
            \"messages\": [
              {
                \"role\": \"system\",
                \"content\": \"You are an OCR assistant. Extract all text from the image.\"
              },
              {
                \"role\": \"user\",
                \"content\": [
                  {\"type\": \"input_text\", \"text\": \"Extract the text from this prescription.\"},
                  {\"type\": \"input_image\", \"image_url\": \"data:image/png;base64,$BASE64_IMG\"}
                ]
              }
            ],
            \"temperature\": 0.0,
            \"max_tokens\": 2000
          }" | jq '.choices[0].message.content' || echo "Failed"

        rm -f /tmp/test_prescription.png
    else
        echo "Skipping - ImageMagick not installed (needed to convert PDF)"
    fi
else
    echo "Test 3: Skipped (no test PDF found)"
fi

echo -e "\n=================================="
echo "Testing Complete"
echo "=================================="
