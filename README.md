# nunoOcr - Medical AI Service

Unified microservice for medical document analysis and wound assessment.

## Overview

nunoOcr is a centralized AI service that provides:
- **OCR for Prescriptions** - DeepSeek-OCR for extracting text from medical documents
- **Wound Analysis** - GPT-4 Vision / Claude Vision for analyzing wound images

This service acts as a secure gateway between your Django application and various AI providers (OpenAI, Anthropic), ensuring your Django app never needs to handle external AI API keys directly.

## Features

### OCR for Prescriptions
- **High-accuracy OCR** - DeepSeek-OCR model optimized for document text extraction
- **Multi-language support** - French, English, German, and other European languages
- **Self-hosted** - Unlimited usage, no per-request costs
- **GPU optional** - Works on CPU with optional GPU acceleration

### Wound Analysis (NEW!)
- **GPT-4 Vision / Claude Vision** - Professional medical image analysis
- **Structured French output** - JSON format with medical fields
- **Centralized API keys** - Django never sees OpenAI/Claude keys
- **Easy provider switching** - OpenAI ↔ Claude with one env var

### General
- **OpenAI-compatible API** - Easy integration with existing clients
- **Docker-based** - Easy deployment and scaling
- **Health checks** - Built-in monitoring endpoints
- **Microservice architecture** - Single service for all AI needs

## Prerequisites

- Docker and Docker Compose installed
- At least 8GB RAM available for the container
- ~4GB disk space for model weights (downloaded on first run)
- (Optional) NVIDIA GPU with nvidia-docker for acceleration

## 🚀 Quick Start

### For Microservice Architecture (Recommended)

See **[QUICK_START_MICROSERVICE.md](QUICK_START_MICROSERVICE.md)** for complete setup guide.

**TL;DR**:
1. Deploy nunoOcr service with `OPENAI_API_KEY`
2. Configure Django with `NUNOOCR_SERVICE_URL`
3. Django calls nunoOcr → nunoOcr calls OpenAI

### For Direct OpenAI Integration (Legacy)

If you want Django to call OpenAI directly (not recommended):

1. Configure Django with `OPENAI_API_KEY`
2. Use `nunoocr_client.py` with `vision_api_key` parameter

## 📚 Documentation

| Document | Description |
|----------|-------------|
| **[QUICK_START_MICROSERVICE.md](QUICK_START_MICROSERVICE.md)** | 🚀 Quick deployment guide (3 steps) |
| **[MICROSERVICE_ARCHITECTURE.md](MICROSERVICE_ARCHITECTURE.md)** | 🏗️ Complete architecture explanation |
| **[API_KEYS_EXPLAINED.md](API_KEYS_EXPLAINED.md)** | 🔑 Understanding the two types of API keys |
| **[API_KEY_SETUP.md](API_KEY_SETUP.md)** | 🔐 Setup API Key authentication in Django |
| **[API_SECURITY.md](API_SECURITY.md)** | 🛡️ Security and rate limiting guide |
| **[INTEGRATION_CHECKLIST.md](INTEGRATION_CHECKLIST.md)** | ✅ Complete integration checklist |

## 🎯 Service Endpoints

### OCR Endpoint (Prescriptions)
```bash
POST /v1/chat/completions
```
OpenAI-compatible endpoint for prescription OCR using DeepSeek-OCR

### Wound Analysis Endpoints (NEW!)
```bash
POST /v1/analyze-wound
```
Analyze single wound image (returns structured French JSON)

```bash
POST /v1/compare-wound-progress
```
Compare multiple wound images over time for progression tracking

### Health Check
```bash
GET /health
```
Returns service status and configuration

## Configuration

### Environment Variables

For nunoOcr service (in Dokploy or docker-compose):

```bash
# Required for wound analysis
OPENAI_API_KEY=sk-proj-xxxxx  # Your OpenAI API key
VISION_PROVIDER=openai         # or 'anthropic' for Claude

# Optional - OCR configuration
MODEL_NAME=deepseek-ai/DeepSeek-OCR
HOST=0.0.0.0
PORT=8000
```

For Django application:

```bash
# URL of the nunoOcr service
NUNOOCR_SERVICE_URL=http://localhost:8765  # or http://nunoocr:8000 in Docker

# NOT needed in Django:
# OPENAI_API_KEY - handled by nunoOcr service!
```

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

## 🏗️ Architecture

### Microservice Architecture (Recommended)

```
Client (Mobile/Web)
    ↓ Authorization: Bearer nuno_xxxxx (your API Key)
Django App (inur.opefitoo.com)
    ↓ Validates API Key + quota
    ↓ POST http://nunoocr:8765/v1/analyze-wound
nunoOcr Service
    ↓ Uses OPENAI_API_KEY (stored here)
OpenAI/Claude API
    ↓ Returns analysis
Client receives result
```

**Benefits**:
- ✅ Django never knows OpenAI keys
- ✅ Easy to switch AI providers
- ✅ Centralized monitoring and caching
- ✅ Better security

See [MICROSERVICE_ARCHITECTURE.md](MICROSERVICE_ARCHITECTURE.md) for details.

## Django Integration

### Microservice Mode (Recommended)

```python
from .nunoocr_client import NunoOcrServiceClient

# Initialize client pointing to nunoOcr service
client = NunoOcrServiceClient(
    base_url="http://nunoocr:8765"  # No API key needed!
)

# Analyze wound
result = client.analyze_wound(request.FILES['wound_image'])
```

### Direct Mode (Legacy)

```python
from nunoocr_client import DeepSeekOCRClient

# Initialize with OpenAI key directly (not recommended)
client = DeepSeekOCRClient(
    vision_api_key=settings.OPENAI_API_KEY,
    vision_provider='openai'
)

result = client.analyze_wound_from_uploaded_file(wound_image)
```

See `django_microservice_integration.py` for complete examples.

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
