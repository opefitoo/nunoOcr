#!/usr/bin/env python3
"""
Test DeepSeek-OCR locally using transformers/vLLM

This script tests OCR without Docker using the HuggingFace model directly.
Requires: pip install transformers torch pillow pdf2image
"""

import sys
from pathlib import Path

def test_with_pytesseract_baseline(pdf_path: Path):
    """Test with pytesseract as baseline for comparison."""
    try:
        from pdf2image import convert_from_path
        import pytesseract

        print("=" * 80)
        print("Testing with Pytesseract (Baseline)")
        print("=" * 80)

        images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)

        if not images:
            print("Failed to convert PDF to image")
            return

        print(f"Converted PDF to {len(images)} image(s)")
        print("\nExtracting text with pytesseract (fra+eng)...")

        text = pytesseract.image_to_string(images[0], lang='fra+eng')

        print("\n" + "-" * 80)
        print("EXTRACTED TEXT (Pytesseract):")
        print("-" * 80)
        print(text)
        print("-" * 80)

        return text

    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install pytesseract pdf2image")
    except Exception as e:
        print(f"Error: {e}")

def test_with_openai_api(pdf_path: Path, base_url: str = "http://localhost:8765"):
    """Test using OpenAI-compatible API if service is running."""
    try:
        import requests
        import base64
        from pdf2image import convert_from_path
        import io

        print("\n" + "=" * 80)
        print("Testing with DeepSeek-OCR API")
        print("=" * 80)

        # Check if service is available
        try:
            response = requests.get(f"{base_url}/health", timeout=2)
            if response.status_code != 200:
                print(f"Service not available at {base_url}")
                return None
        except:
            print(f"Service not available at {base_url}")
            return None

        print("✅ Service is available")

        # Convert PDF to image
        images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)
        if not images:
            print("Failed to convert PDF")
            return None

        # Convert to base64
        img_byte_arr = io.BytesIO()
        images[0].save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()
        b64_img = base64.b64encode(img_bytes).decode('utf-8')
        data_uri = f"data:image/png;base64,{b64_img}"

        print("Sending request to DeepSeek-OCR API...")

        payload = {
            "model": "deepseek-ai/DeepSeek-OCR",
            "messages": [
                {
                    "role": "system",
                    "content": "You are an OCR assistant for Luxembourg medical prescriptions. Extract all text accurately, preserving structure."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Extract all text from this prescription."},
                        {"type": "input_image", "image_url": data_uri}
                    ]
                }
            ],
            "temperature": 0.0,
            "max_tokens": 2000
        }

        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()
        text = result['choices'][0]['message']['content']

        print("\n" + "-" * 80)
        print("EXTRACTED TEXT (DeepSeek-OCR):")
        print("-" * 80)
        print(text)
        print("-" * 80)

        if 'usage' in result:
            print(f"\nTokens used: {result['usage'].get('total_tokens', 'N/A')}")

        return text

    except Exception as e:
        print(f"Error: {e}")
        return None

def extract_prescription_data(text: str):
    """Extract structured data from prescription text."""
    import re

    print("\n" + "=" * 80)
    print("Extracting Structured Data")
    print("=" * 80)

    data = {}

    # Doctor code
    code_patterns = [
        r'\b(90\d{4}-\d{2})\b',
        r'\b(90\d{6})\b',
    ]
    for pattern in code_patterns:
        match = re.search(pattern, text)
        if match:
            data['doctor_code'] = match.group(1)
            break

    # Patient matricule (13 digits)
    matricule_match = re.search(r'\b(\d{13})\b', text)
    if matricule_match:
        data['patient_matricule'] = matricule_match.group(1)

    # Date
    date_match = re.search(r'(\d{1,2})[/\-](\d{1,2})[/\-](\d{4})', text)
    if date_match:
        data['date'] = f"{date_match.group(3)}-{date_match.group(2).zfill(2)}-{date_match.group(1).zfill(2)}"

    # Doctor name
    name_patterns = [
        r'Dr\.?\s+([A-Z][a-z]+)\s+([A-Z]+)',
        r'Prof\.\s+Dr\.?\s+([A-Z][a-z]+)\s+([A-Z]+)',
    ]
    for pattern in name_patterns:
        match = re.search(pattern, text)
        if match:
            data['doctor_name'] = f"{match.group(1)} {match.group(2)}"
            break

    # Patient name
    patient_patterns = [
        r'Patient[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'M(?:me|r|lle)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
    ]
    for pattern in patient_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            data['patient_name'] = match.group(1)
            break

    print("\nExtracted Data:")
    for key, value in data.items():
        print(f"  {key}: {value}")

    return data

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_local_ocr.py <pdf_file>")
        print("\nExample:")
        print("  python test_local_ocr.py bausch_BRW283A4D6DFC2A_20180728_013026_010420.pdf")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    print(f"Testing OCR with: {pdf_path.name}")
    print(f"File size: {pdf_path.stat().st_size / 1024:.2f} KB")

    # Test 1: Try DeepSeek-OCR API if available
    deepseek_text = test_with_openai_api(pdf_path)

    # Test 2: Baseline with pytesseract
    pytesseract_text = test_with_pytesseract_baseline(pdf_path)

    # Extract structured data from whichever worked
    if deepseek_text:
        extract_prescription_data(deepseek_text)
    elif pytesseract_text:
        extract_prescription_data(pytesseract_text)

    print("\n" + "=" * 80)
    print("Testing Complete")
    print("=" * 80)

if __name__ == '__main__':
    main()
