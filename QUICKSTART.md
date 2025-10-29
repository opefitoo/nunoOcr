# Quick Start Guide

Get DeepSeek-OCR running in 5 minutes!

## 1. Start the Service (First Time)

```bash
# Navigate to this directory
cd /Users/mehdi/workspace/clients/inur-sur.lu/nuno/nunoOcr

# Start the service
docker-compose up -d

# Watch the logs (model downloads on first run, takes 5-10 min)
docker-compose logs -f deepseek-ocr
```

Wait until you see:
```
INFO Server running on 0.0.0.0:8000
```

## 2. Test the Service

```bash
# Check health
curl http://localhost:8765/health

# Should return: {"status":"ok"}
```

## 3. Test with Python (Optional)

```bash
# Install dependencies
pip install -r requirements.txt

# Run health check
python test_ocr.py --health-check

# Test with your prescription file
python test_ocr.py --extract-text /path/to/prescription.pdf
```

## 4. Integrate with Django

### Copy the client library to Django project:

```bash
cp nunoocr_client.py ../inur.django/invoices/services/
```

### Add to Django settings:

```python
# In invoices/settings/base.py
OCR_SERVICE_URL = env('OCR_SERVICE_URL', default='http://localhost:8765')
OCR_SERVICE_API_KEY = env('OCR_SERVICE_API_KEY', default='')
USE_DEEPSEEK_OCR = env.bool('USE_DEEPSEEK_OCR', default=True)
```

### Add to .env:

```bash
OCR_SERVICE_URL=http://localhost:8765
USE_DEEPSEEK_OCR=true
```

### Use in your code:

```python
from nunoocr_client import DjangoOCRService

# In your view
ocr_service = DjangoOCRService()

if ocr_service.is_available():
    result = ocr_service.extract_from_uploaded_file(
        request.FILES['prescription'],
        extract_structured=True
    )
    print(result)
```

## 5. Common Commands

```bash
# View logs
docker-compose logs -f deepseek-ocr

# Restart service
docker-compose restart deepseek-ocr

# Stop service
docker-compose down

# Stop and remove volumes (clears model cache)
docker-compose down -v

# Check status
docker-compose ps
```

## Next Steps

- Read [README.md](README.md) for detailed documentation
- Check [django_integration_example.py](django_integration_example.py) for integration patterns
- Test with your own prescription images

## Troubleshooting

### Service won't start?
```bash
# Check if port 8765 is in use
lsof -i :8765

# Check Docker status
docker ps -a

# View error logs
docker-compose logs deepseek-ocr
```

### Model download is slow?
This is normal. The model is ~3-4GB and only downloads once. Subsequent starts are instant.

### OCR results are poor?
- Ensure images are high quality (300 DPI minimum)
- Make sure images are not rotated
- Try preprocessing (deskew, denoise)

## Support

- GitHub Issues: https://github.com/timmyovo/deepseek-ocr.rs/issues
- DeepSeek-OCR Docs: https://github.com/deepseek-ai/DeepSeek-OCR
