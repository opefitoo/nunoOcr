#!/usr/bin/env python3
"""
Test script for DeepSeek-OCR service

Usage:
    python test_ocr.py <image_or_pdf_path>
    python test_ocr.py --health-check
    python test_ocr.py --extract-text sample.pdf
    python test_ocr.py --extract-data prescription.pdf
"""

import sys
import argparse
import json
from pathlib import Path
from nunoocr_client import DeepSeekOCRClient, DeepSeekOCRError


def print_separator(title: str = ""):
    """Print a visual separator."""
    if title:
        print(f"\n{'=' * 80}")
        print(f" {title}")
        print('=' * 80)
    else:
        print('-' * 80)


def test_health_check(client: DeepSeekOCRClient):
    """Test if the OCR service is healthy."""
    print_separator("Health Check")
    print(f"Checking OCR service at: {client.base_url}")

    is_healthy = client.health_check()

    if is_healthy:
        print("✅ Service is healthy and responding")
        return True
    else:
        print("❌ Service is not responding")
        print("\nTroubleshooting:")
        print("1. Check if Docker container is running:")
        print("   docker-compose ps")
        print("2. Check container logs:")
        print("   docker-compose logs deepseek-ocr")
        print("3. Try starting the service:")
        print("   docker-compose up -d")
        return False


def test_extract_text(client: DeepSeekOCRClient, file_path: Path):
    """Test plain text extraction."""
    print_separator(f"Text Extraction: {file_path.name}")

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    print(f"File: {file_path}")
    print(f"Size: {file_path.stat().st_size / 1024:.2f} KB")
    print("\nExtracting text...")

    try:
        with open(file_path, 'rb') as f:
            result = client.extract_text(f, file_type='auto')

        print("\n✅ Text extraction successful!")
        print_separator("Extracted Text")
        print(result['text'])
        print_separator()
        print(f"\nMetadata:")
        print(f"  Tokens used: {result['metadata']['tokens_used']}")
        print(f"  Model: {result['metadata']['model']}")

    except DeepSeekOCRError as e:
        print(f"\n❌ Extraction failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


def test_extract_prescription_data(client: DeepSeekOCRClient, file_path: Path):
    """Test structured prescription data extraction."""
    print_separator(f"Prescription Data Extraction: {file_path.name}")

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    print(f"File: {file_path}")
    print(f"Size: {file_path.stat().st_size / 1024:.2f} KB")
    print("\nExtracting structured prescription data...")

    try:
        with open(file_path, 'rb') as f:
            data = client.extract_prescription_data(f, file_type='auto')

        print("\n✅ Data extraction successful!")
        print_separator("Extracted Data (JSON)")

        # Remove metadata for cleaner display
        metadata = data.pop('_metadata', None)

        # Pretty print the data
        print(json.dumps(data, indent=2, ensure_ascii=False))

        if metadata:
            print_separator("Metadata")
            print(f"Tokens used: {metadata['tokens_used']}")
            print(f"Model: {metadata['model']}")

        # Display key information
        print_separator("Summary")
        print(f"Doctor: {data.get('doctor_name', 'N/A')} ({data.get('doctor_code', 'N/A')})")
        print(f"Patient: {data.get('patient_name', 'N/A')} ({data.get('patient_matricule', 'N/A')})")
        print(f"Date: {data.get('prescription_date', 'N/A')}")
        print(f"Medications: {len(data.get('medications', []))} item(s)")

        if data.get('medications'):
            print("\nMedications:")
            for i, med in enumerate(data['medications'], 1):
                print(f"  {i}. {med}")

    except DeepSeekOCRError as e:
        print(f"\n❌ Extraction failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


def test_multi_page_pdf(client: DeepSeekOCRClient, file_path: Path):
    """Test multi-page PDF extraction."""
    print_separator(f"Multi-Page PDF Extraction: {file_path.name}")

    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    print(f"File: {file_path}")
    print(f"Size: {file_path.stat().st_size / 1024:.2f} KB")
    print("\nExtracting text from all pages...")

    try:
        result = client.extract_text_multi_page(file_path)

        print(f"\n✅ Extraction successful!")
        print(f"Total pages: {result['total_pages']}")

        for page_data in result['pages']:
            print_separator(f"Page {page_data['page_number']}")
            print(page_data['text'])
            print(f"\nTokens: {page_data['metadata']['tokens_used']}")

        print_separator("Full Text (All Pages)")
        print(result['full_text'])

    except DeepSeekOCRError as e:
        print(f"\n❌ Extraction failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Test DeepSeek-OCR service',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_ocr.py --health-check
  python test_ocr.py --extract-text sample.pdf
  python test_ocr.py --extract-data prescription.pdf
  python test_ocr.py --multi-page document.pdf
  python test_ocr.py --url http://ocr-service:8000 --extract-text sample.pdf
        """
    )

    parser.add_argument(
        '--url',
        default='http://localhost:8765',
        help='OCR service URL (default: http://localhost:8765)'
    )

    parser.add_argument(
        '--api-key',
        help='API key for authentication (optional)'
    )

    parser.add_argument(
        '--health-check',
        action='store_true',
        help='Check if OCR service is healthy'
    )

    parser.add_argument(
        '--extract-text',
        metavar='FILE',
        help='Extract plain text from image or PDF'
    )

    parser.add_argument(
        '--extract-data',
        metavar='FILE',
        help='Extract structured prescription data'
    )

    parser.add_argument(
        '--multi-page',
        metavar='FILE',
        help='Extract text from multi-page PDF'
    )

    args = parser.parse_args()

    # Initialize client
    client = DeepSeekOCRClient(
        base_url=args.url,
        api_key=args.api_key
    )

    print_separator("DeepSeek-OCR Test Script")
    print(f"Service URL: {client.base_url}")

    # Run tests based on arguments
    if args.health_check:
        success = test_health_check(client)
        sys.exit(0 if success else 1)

    elif args.extract_text:
        file_path = Path(args.extract_text)
        test_extract_text(client, file_path)

    elif args.extract_data:
        file_path = Path(args.extract_data)
        test_extract_prescription_data(client, file_path)

    elif args.multi_page:
        file_path = Path(args.multi_page)
        test_multi_page_pdf(client, file_path)

    else:
        # No specific test selected, run health check
        print("\nNo test specified. Running health check...\n")
        test_health_check(client)
        print("\nUse --help to see available test options.")


if __name__ == '__main__':
    main()
