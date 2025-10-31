#!/usr/bin/env python3
"""
Test deployed OCR service

Usage:
    python test_deployed_service.py <service_url> [pdf_file]

Examples:
    python test_deployed_service.py https://nunoocr.yourdomain.com
    python test_deployed_service.py http://123.45.67.89:8765 prescription.pdf
"""

import sys
import requests
from pathlib import Path

def test_health(service_url):
    """Test health endpoint."""
    print("=" * 80)
    print("Testing Health Endpoint")
    print("=" * 80)
    print(f"URL: {service_url}/health\n")

    try:
        response = requests.get(f"{service_url}/health", timeout=5)
        response.raise_for_status()

        data = response.json()
        status = data.get('status', 'unknown')

        if status == 'ok':
            print("✅ Service is HEALTHY and ready!")
            print(f"   Model: {data.get('model', 'N/A')}")
            return True
        elif status == 'initializing':
            print("⏳ Service is INITIALIZING (model still loading)")
            print("   Wait a few minutes and try again")
            return False
        else:
            print(f"⚠️  Unknown status: {status}")
            return False

    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to service")
        print("   Check if the URL is correct and service is running")
        return False
    except requests.exceptions.Timeout:
        print("❌ Connection timeout")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_ocr(service_url, pdf_path=None):
    """Test OCR with a PDF file."""
    if not pdf_path:
        print("\n⏭️  Skipping OCR test (no PDF provided)")
        return

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"\n❌ PDF file not found: {pdf_path}")
        return

    print("\n" + "=" * 80)
    print("Testing OCR Extraction")
    print("=" * 80)
    print(f"File: {pdf_path.name}")
    print(f"Size: {pdf_path.stat().st_size / 1024:.2f} KB\n")

    try:
        # Import dependencies
        from pdf2image import convert_from_path
        import base64
        import io

        print("Converting PDF to image...")
        images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)

        if not images:
            print("❌ Failed to convert PDF")
            return

        # Convert to base64
        img_byte_arr = io.BytesIO()
        images[0].save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        data_uri = f"data:image/png;base64,{b64_img}"

        print("Sending OCR request...")
        print("(This may take 10-30 seconds on first request)\n")

        payload = {
            "model": "deepseek-ai/DeepSeek-OCR",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an OCR assistant. Extract all text accurately."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Extract all text from this image."},
                        {"type": "input_image", "image_url": data_uri}
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 2000
        }

        response = requests.post(
            f"{service_url}/v1/chat/completions",
            json=payload,
            timeout=300  # 5 minutes timeout (OCR on CPU is slow)
        )
        response.raise_for_status()

        result = response.json()
        text = result['choices'][0]['message']['content']

        print("✅ OCR Extraction Successful!\n")
        print("-" * 80)
        print("EXTRACTED TEXT:")
        print("-" * 80)
        print(text)
        print("-" * 80)

        if 'usage' in result:
            print(f"\nTokens used: {result['usage'].get('total_tokens', 'N/A')}")

    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Install with: pip install pdf2image Pillow")
    except requests.exceptions.Timeout:
        print("❌ Request timeout (service may be slow or overloaded)")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_deployed_service.py <service_url> [pdf_file]")
        print("\nExamples:")
        print("  python test_deployed_service.py https://nunoocr.yourdomain.com")
        print("  python test_deployed_service.py http://123.45.67.89:8765 prescription.pdf")
        sys.exit(1)

    service_url = sys.argv[1].rstrip('/')
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"\nTesting DeepSeek-OCR Service")
    print(f"URL: {service_url}\n")

    # Test 1: Health check
    is_healthy = test_health(service_url)

    # Test 2: OCR (if healthy and PDF provided)
    if is_healthy and pdf_file:
        test_ocr(service_url, pdf_file)
    elif not is_healthy:
        print("\n⚠️  Service not ready. Fix health check before testing OCR.")

    print("\n" + "=" * 80)
    print("Testing Complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
