#!/usr/bin/env python3
"""
Wound Analysis Example - Using DeepSeek-OCR for Medical Wound Assessment

This example demonstrates how to use the wound analysis feature to:
1. Analyze wound images and extract medical information
2. Get structured JSON data about wounds
3. Integrate wound analysis into Django applications

Usage:
    # Basic usage
    python wound_analysis_example.py path/to/wound_image.jpg

    # With custom service URL
    python wound_analysis_example.py path/to/wound_image.jpg --url http://localhost:8765

    # Get unstructured text instead of JSON
    python wound_analysis_example.py path/to/wound_image.jpg --unstructured
"""

import sys
import argparse
from pathlib import Path
from nunoocr_client import DeepSeekOCRClient, DjangoOCRService, DeepSeekOCRError
import json


def example_basic_wound_analysis(image_path: str, service_url: str = "http://localhost:8765"):
    """
    Example 1: Basic wound analysis with structured output
    """
    print("=" * 80)
    print("Example 1: Basic Wound Analysis (Structured JSON)")
    print("=" * 80)

    client = DeepSeekOCRClient(base_url=service_url)

    # Check service availability
    if not client.health_check():
        print(f"⚠️  Warning: OCR service at {service_url} is not responding")
        print("   Make sure the service is running with: docker compose up")
        return False

    print(f"✅ Connected to OCR service at {service_url}\n")

    try:
        with open(image_path, 'rb') as f:
            print(f"Analyzing wound image: {Path(image_path).name}")
            print("Please wait, this may take 30-60 seconds...\n")

            # Get structured wound analysis
            result = client.analyze_wound(f, return_structured=True)

            print("✅ Analysis Complete!\n")
            print("-" * 80)

            if result.get('structured', True):
                # Pretty print the structured data
                print("STRUCTURED ANALYSIS:")
                print("-" * 80)

                # Print main fields
                print(f"Wound Type: {result.get('wound_type', 'N/A')}")
                print(f"Location: {result.get('location', 'N/A')}")

                dimensions = result.get('dimensions', {})
                if dimensions:
                    length = dimensions.get('length_cm', 'N/A')
                    width = dimensions.get('width_cm', 'N/A')
                    print(f"Dimensions: {length} x {width} cm")

                print(f"Healing Stage: {result.get('healing_stage', 'N/A')}")
                print(f"Closure Method: {result.get('closure_method', 'N/A')}")
                print(f"Closure Count: {result.get('closure_count', 'N/A')}")

                infections = result.get('signs_of_infection', [])
                if infections:
                    print(f"Signs of Infection: {', '.join(infections)}")
                else:
                    print("Signs of Infection: None detected")

                complications = result.get('complications', [])
                if complications:
                    print(f"Complications: {', '.join(complications)}")
                else:
                    print("Complications: None detected")

                print(f"Overall Condition: {result.get('overall_condition', 'N/A')}")
                print(f"Confidence: {result.get('confidence', 'N/A')}")

                notes = result.get('notes', '')
                if notes:
                    print(f"\nAdditional Notes:\n{notes}")

                print("-" * 80)

                # Show full JSON
                print("\nFULL JSON RESPONSE:")
                print("-" * 80)
                # Remove metadata for cleaner display
                display_data = {k: v for k, v in result.items() if k != '_metadata'}
                print(json.dumps(display_data, indent=2))

            else:
                # Unstructured response
                print(result.get('analysis', 'No analysis available'))

            print("-" * 80)

            # Show metadata
            metadata = result.get('_metadata', {})
            print(f"\nTokens used: {metadata.get('tokens_used', 'N/A')}")
            print(f"Model: {metadata.get('model', 'N/A')}")

            return True

    except FileNotFoundError:
        print(f"❌ Error: Image file not found: {image_path}")
        return False
    except DeepSeekOCRError as e:
        print(f"❌ OCR Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_unstructured_analysis(image_path: str, service_url: str = "http://localhost:8765"):
    """
    Example 2: Wound analysis with unstructured text output
    """
    print("\n" + "=" * 80)
    print("Example 2: Wound Analysis (Unstructured Text)")
    print("=" * 80)

    client = DeepSeekOCRClient(base_url=service_url)

    try:
        with open(image_path, 'rb') as f:
            print(f"Analyzing wound image: {Path(image_path).name}")
            print("Getting detailed narrative analysis...\n")

            # Get unstructured text analysis
            result = client.analyze_wound(f, return_structured=False)

            print("✅ Analysis Complete!\n")
            print("-" * 80)
            print("ANALYSIS:")
            print("-" * 80)
            print(result.get('analysis', 'No analysis available'))
            print("-" * 80)

            return True

    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def example_django_integration():
    """
    Example 3: Django integration for wound analysis
    """
    print("\n" + "=" * 80)
    print("Example 3: Django Integration Example Code")
    print("=" * 80)

    django_code = '''
# In your Django views.py:

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from nunoocr_client import DjangoOCRService

@csrf_exempt
def analyze_wound_view(request):
    """
    API endpoint to analyze wound images.

    Accepts: POST with 'wound_image' file
    Returns: JSON with wound analysis
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    if 'wound_image' not in request.FILES:
        return JsonResponse({'error': 'No image provided'}, status=400)

    # Initialize OCR service
    ocr = DjangoOCRService()

    # Check if service is available
    if not ocr.is_available():
        return JsonResponse({
            'error': 'OCR service unavailable',
            'detail': 'Please check if the OCR service is running'
        }, status=503)

    try:
        # Analyze wound from uploaded file
        analysis = ocr.analyze_wound_from_uploaded_file(
            request.FILES['wound_image'],
            return_structured=True
        )

        return JsonResponse({
            'success': True,
            'analysis': analysis
        })

    except Exception as e:
        return JsonResponse({
            'error': 'Analysis failed',
            'detail': str(e)
        }, status=500)


# In your Django settings.py:

# OCR Service Configuration
OCR_SERVICE_URL = env('OCR_SERVICE_URL', default='http://localhost:8765')
OCR_SERVICE_API_KEY = env('OCR_SERVICE_API_KEY', default='')


# In your Django urls.py:

from django.urls import path
from . import views

urlpatterns = [
    path('api/analyze-wound/', views.analyze_wound_view, name='analyze_wound'),
]


# Example model for storing wound assessments:

from django.db import models

class WoundAssessment(models.Model):
    """Store wound analysis results"""
    patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='wounds/')

    # Analysis results
    wound_type = models.CharField(max_length=100)
    location = models.CharField(max_length=200)
    length_cm = models.FloatField(null=True, blank=True)
    width_cm = models.FloatField(null=True, blank=True)
    healing_stage = models.CharField(max_length=50)
    closure_method = models.CharField(max_length=50)
    closure_count = models.IntegerField(null=True, blank=True)
    overall_condition = models.TextField()
    notes = models.TextField(blank=True)

    # Metadata
    confidence = models.CharField(max_length=20)
    analyzed_at = models.DateTimeField(auto_now_add=True)
    raw_analysis = models.JSONField()

    class Meta:
        ordering = ['-analyzed_at']

    def __str__(self):
        return f"Wound Assessment for {self.patient} - {self.analyzed_at}"
'''

    print(django_code)
    print("=" * 80)
    print("\nYou can copy this code to integrate wound analysis into your Django app.")


def main():
    parser = argparse.ArgumentParser(
        description='Wound Analysis Examples using DeepSeek-OCR',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  %(prog)s wound.jpg
  %(prog)s wound.jpg --url https://nunoocrapi.opefitoo.com
  %(prog)s wound.jpg --unstructured
  %(prog)s --django-example
        '''
    )

    parser.add_argument('image', nargs='?', help='Path to wound image file')
    parser.add_argument('--url', default='http://localhost:8765',
                       help='OCR service URL (default: http://localhost:8765)')
    parser.add_argument('--unstructured', action='store_true',
                       help='Get unstructured text instead of JSON')
    parser.add_argument('--django-example', action='store_true',
                       help='Show Django integration example code')

    args = parser.parse_args()

    # Show Django example if requested
    if args.django_example:
        example_django_integration()
        return 0

    # Require image path for other examples
    if not args.image:
        parser.print_help()
        print("\nError: image path required (or use --django-example)")
        return 1

    # Validate image path
    if not Path(args.image).exists():
        print(f"❌ Error: Image file not found: {args.image}")
        return 1

    print("\n🏥 DeepSeek-OCR Wound Analysis Examples")
    print(f"Service URL: {args.url}")
    print(f"Image: {args.image}\n")

    # Run examples
    if args.unstructured:
        success = example_unstructured_analysis(args.image, args.url)
    else:
        success = example_basic_wound_analysis(args.image, args.url)

    if success:
        example_django_integration()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
