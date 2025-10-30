#!/bin/bash
# Monitor OCR service status

SERVICE_URL="https://nunoocr.opefitoo.com"
CHECK_INTERVAL=10  # seconds

echo "🔍 Monitoring DeepSeek-OCR Service"
echo "URL: $SERVICE_URL"
echo "Checking every ${CHECK_INTERVAL}s..."
echo "Press Ctrl+C to stop"
echo ""

while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

    # Get health status
    RESPONSE=$(curl -s "$SERVICE_URL/health" 2>/dev/null)

    if [ $? -eq 0 ]; then
        STATUS=$(echo "$RESPONSE" | jq -r '.status' 2>/dev/null)
        MODEL=$(echo "$RESPONSE" | jq -r '.model' 2>/dev/null)

        case "$STATUS" in
            "ok")
                echo "[$TIMESTAMP] ✅ READY - Model: $MODEL"
                echo ""
                echo "🎉 Service is ready to use!"
                echo "You can now use DeepSeek-OCR in Django."
                exit 0
                ;;
            "initializing")
                echo "[$TIMESTAMP] ⏳ INITIALIZING - Downloading model..."
                ;;
            *)
                echo "[$TIMESTAMP] ⚠️  Unknown status: $STATUS"
                ;;
        esac
    else
        echo "[$TIMESTAMP] ❌ Cannot connect to service"
    fi

    sleep $CHECK_INTERVAL
done
