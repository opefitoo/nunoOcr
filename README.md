# Nuno OCR Service

DeepSeek-OCR microservice for extracting text from medical prescription scans in the Nuno Django application.

## Overview

This service provides a standalone OCR API using DeepSeek-OCR (Rust implementation) that can be called from the Nuno Django application to extract text and structured data from medical prescription images and PDFs.

## Features

- **High-accuracy OCR** - DeepSeek-OCR model optimized for document text extraction
- **OpenAI-compatible API** - Easy integration with existing OpenAI client libraries
- **Multi-language support** - Works with French, English, German, and other European languages
- **Docker-based** - Easy deployment and scaling
- **GPU optional** - Works on CPU, with optional GPU acceleration
- **Model caching** - Downloads model once, cached locally
- **Health checks** - Built-in monitoring and health endpoints

## Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM available for the container
- ~4GB disk space for model weights (downloaded on first run)
- (Optional) NVIDIA GPU with nvidia-docker for acceleration

## Quick Start

### 1. Start the service

```bash
docker-compose up -d
```

The service will:
- Download the DeepSeek-OCR model (~3-4GB) on first run
- Start the API server on http://localhost:8765
- Cache the model in `./models/` directory

**Note:** First startup may take 5-10 minutes to download the model. Watch the logs:

```bash
docker-compose logs -f deepseek-ocr
```

### 2. Check service health

```bash
curl http://localhost:8765/health
```

Expected response: `{"status":"ok"}`

### 3. Test with a sample image

See the [Testing](#testing) section below.

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Available options:
- `API_KEY` - Optional API key for authentication (leave empty for no auth)

### GPU Support

If you have an NVIDIA GPU, uncomment the GPU sections in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
runtime: nvidia
```

## API Usage

### OpenAI-Compatible Chat Completions Endpoint

**Endpoint:** `POST http://localhost:8765/v1/chat/completions`

**Request Format:**

```bash
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-OCR",
    "messages": [
      {
        "role": "system",
        "content": "You are an OCR assistant for medical prescriptions. Extract all text accurately, preserving structure and formatting."
      },
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": "Extract the text from this prescription image."
          },
          {
            "type": "input_image",
            "image_url": "data:image/png;base64,<BASE64_ENCODED_IMAGE>"
          }
        ]
      }
    ],
    "max_tokens": 2000,
    "temperature": 0.0
  }'
```

**Response Format:**

```json
{
  "id": "chatcmpl-xyz",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "deepseek-ai/DeepSeek-OCR",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Extracted text from the prescription:\n\nDr. Jean DUPONT\nCode médecin: 901234-56\n..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 200,
    "total_tokens": 350
  }
}
```

### Structured Data Extraction

For medical prescriptions, use a specific system prompt to extract structured data:

```json
{
  "model": "deepseek-ai/DeepSeek-OCR",
  "messages": [
    {
      "role": "system",
      "content": "You are an OCR assistant for Luxembourg medical prescriptions. Extract data in JSON format with these fields: doctor_code, doctor_name, patient_matricule, patient_name, prescription_date, medications (list), notes."
    },
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Extract prescription data as JSON."},
        {"type": "input_image", "image_url": "data:image/png;base64,..."}
      ]
    }
  ],
  "temperature": 0.0
}
```

## Django Integration

### Install Python Client

Add the client library to your Django project:

```bash
pip install openai requests pillow
```

### Use in Django Views

```python
from nunoocr_client import DeepSeekOCRClient

# Initialize client
ocr_client = DeepSeekOCRClient(base_url="http://localhost:8765")

# Extract text from prescription image
with open('prescription.pdf', 'rb') as f:
    result = ocr_client.extract_text(f, file_type='pdf')

print(result['text'])  # Extracted text
print(result['confidence'])  # Confidence score if available
```

See `nunoocr_client.py` for the complete client implementation.

## Testing

### Manual Testing

Use the provided test script:

```bash
python test_ocr.py path/to/prescription.pdf
```

Or test with curl:

```bash
# Convert image to base64
base64 -i prescription.png -o prescription_b64.txt

# Make API request
curl http://localhost:8765/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d @request_payload.json
```

## Management

### View Logs

```bash
docker-compose logs -f deepseek-ocr
```

### Restart Service

```bash
docker-compose restart deepseek-ocr
```

### Stop Service

```bash
docker-compose down
```

### Update Model

To update to a newer model version:

```bash
# Remove cached model
rm -rf models/*

# Restart service (will download new model)
docker-compose restart deepseek-ocr
```

## Troubleshooting

### Service won't start

1. Check if port 8765 is available:
   ```bash
   lsof -i :8765
   ```

2. Check Docker logs:
   ```bash
   docker-compose logs deepseek-ocr
   ```

### Model download fails

- Ensure you have stable internet connection
- Check available disk space (need ~4GB)
- Try restarting the service

### Low accuracy results

- Ensure input images are high quality (300 DPI minimum for scans)
- Use appropriate system prompts for your document type
- Consider preprocessing images (deskew, denoise, etc.)

### Out of memory errors

- Increase Docker memory limit (8GB minimum recommended)
- Reduce batch size or image resolution
- Consider using GPU if available

## Performance Notes

### CPU Mode
- First inference: ~5-10 seconds per page
- Subsequent inferences: ~3-5 seconds per page
- Memory usage: ~4-6GB

### GPU Mode (NVIDIA)
- First inference: ~2-3 seconds per page
- Subsequent inferences: ~1-2 seconds per page
- Memory usage: ~6-8GB RAM + 2-4GB VRAM

## Network Configuration

The service creates a Docker network named `nunoocr_network`. To connect the Django app's Docker network to this service:

```yaml
# In your Django docker-compose.yml
networks:
  nunoocr_network:
    external: true
```

Then access the service at `http://deepseek-ocr:8000` from within Django containers.

## Security Considerations

- **No authentication by default** - Set `API_KEY` in `.env` for production
- **Internal network only** - Do not expose port 8765 to the internet
- **Use reverse proxy** - Add nginx/traefik with HTTPS for production
- **Rate limiting** - Consider adding rate limits for production use

## License

This service uses DeepSeek-OCR (MIT License) via the Rust implementation.

## Support

For issues specific to:
- This integration: Contact Nuno development team
- DeepSeek-OCR model: https://github.com/deepseek-ai/DeepSeek-OCR
- Rust server: https://github.com/timmyovo/deepseek-ocr.rs
