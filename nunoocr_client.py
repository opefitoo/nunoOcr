"""
DeepSeek-OCR Client for Django Integration

This module provides a Python client for interacting with the DeepSeek-OCR service
to extract text and structured data from medical prescription images and PDFs.

Usage:
    from nunoocr_client import DeepSeekOCRClient

    client = DeepSeekOCRClient(base_url="http://localhost:8765")

    # Extract plain text
    with open('prescription.pdf', 'rb') as f:
        result = client.extract_text(f, file_type='pdf')

    # Extract structured data
    with open('prescription.pdf', 'rb') as f:
        data = client.extract_prescription_data(f, file_type='pdf')
"""

import base64
import json
import io
from typing import Dict, Any, Optional, BinaryIO, Union
from pathlib import Path

import requests
from PIL import Image


class DeepSeekOCRError(Exception):
    """Base exception for DeepSeek-OCR client errors."""
    pass


class DeepSeekOCRClient:
    """
    Client for DeepSeek-OCR service with OpenAI-compatible API.

    Attributes:
        base_url: Base URL of the OCR service (e.g., http://localhost:8765)
        api_key: Optional API key for authentication
        timeout: Request timeout in seconds (default: 60)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        api_key: Optional[str] = None,
        timeout: int = 60
    ):
        """
        Initialize the OCR client.

        Args:
            base_url: Base URL of the OCR service
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
        self.health_endpoint = f"{self.base_url}/health"

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def health_check(self) -> bool:
        """
        Check if the OCR service is healthy and responding.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            response = requests.get(self.health_endpoint, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def _image_to_base64(
        self,
        file_obj: BinaryIO,
        file_type: str = 'auto'
    ) -> str:
        """
        Convert image or PDF to base64 encoded string.

        Args:
            file_obj: File object (can be image or PDF)
            file_type: File type ('image', 'pdf', or 'auto')

        Returns:
            Base64 encoded string with data URI prefix
        """
        file_obj.seek(0)
        file_bytes = file_obj.read()

        # Auto-detect file type if needed
        if file_type == 'auto':
            # Check magic bytes
            if file_bytes[:4] == b'%PDF':
                file_type = 'pdf'
            else:
                file_type = 'image'

        # For PDFs, convert first page to image
        if file_type == 'pdf':
            try:
                from pdf2image import convert_from_bytes
                images = convert_from_bytes(file_bytes, dpi=300, first_page=1, last_page=1)
                if images:
                    # Convert PIL image to bytes
                    img_byte_arr = io.BytesIO()
                    images[0].save(img_byte_arr, format='PNG')
                    file_bytes = img_byte_arr.getvalue()
                    mime_type = 'image/png'
                else:
                    raise DeepSeekOCRError("Failed to convert PDF to image")
            except ImportError:
                raise DeepSeekOCRError(
                    "pdf2image library required for PDF support. "
                    "Install with: pip install pdf2image"
                )
        else:
            # Detect image MIME type
            try:
                img = Image.open(io.BytesIO(file_bytes))
                format_lower = img.format.lower()
                mime_type = f'image/{format_lower}'
            except Exception:
                mime_type = 'image/png'  # Default fallback

        # Encode to base64
        b64_encoded = base64.b64encode(file_bytes).decode('utf-8')
        return f"data:{mime_type};base64,{b64_encoded}"

    def _make_request(
        self,
        system_prompt: str,
        user_prompt: str,
        image_data_uri: str,
        temperature: float = 0.0,
        max_tokens: int = 2000
    ) -> Dict[str, Any]:
        """
        Make a request to the OCR API.

        Args:
            system_prompt: System instruction for the model
            user_prompt: User prompt describing the task
            image_data_uri: Base64-encoded image data URI
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            API response dictionary

        Raises:
            DeepSeekOCRError: If request fails
        """
        payload = {
            "model": "deepseek-ai/DeepSeek-OCR",
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_prompt
                        },
                        {
                            "type": "input_image",
                            "image_url": image_data_uri
                        }
                    ]
                }
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(
                self.chat_endpoint,
                headers=self._get_headers(),
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise DeepSeekOCRError(f"OCR API request failed: {str(e)}")

    def extract_text(
        self,
        file_obj: BinaryIO,
        file_type: str = 'auto',
        preserve_formatting: bool = True
    ) -> Dict[str, Any]:
        """
        Extract plain text from image or PDF.

        Args:
            file_obj: File object (image or PDF)
            file_type: File type ('image', 'pdf', or 'auto')
            preserve_formatting: Whether to preserve text structure

        Returns:
            Dictionary with 'text' and 'metadata' keys

        Example:
            with open('prescription.pdf', 'rb') as f:
                result = client.extract_text(f)
                print(result['text'])
        """
        system_prompt = (
            "You are an OCR assistant specialized in extracting text from documents. "
            "Extract all visible text accurately, preserving structure and formatting. "
            "Include all details such as names, codes, dates, and medication information."
        )

        if preserve_formatting:
            user_prompt = (
                "Extract all text from this document, preserving the original "
                "formatting and structure as much as possible."
            )
        else:
            user_prompt = "Extract all text from this document."

        image_data_uri = self._image_to_base64(file_obj, file_type)
        response = self._make_request(system_prompt, user_prompt, image_data_uri)

        # Extract text from response
        extracted_text = response['choices'][0]['message']['content']

        return {
            'text': extracted_text,
            'metadata': {
                'tokens_used': response.get('usage', {}).get('total_tokens', 0),
                'model': response.get('model', 'unknown')
            }
        }

    def extract_prescription_data(
        self,
        file_obj: BinaryIO,
        file_type: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Extract structured data from a Luxembourg medical prescription.

        Args:
            file_obj: File object (image or PDF of prescription)
            file_type: File type ('image', 'pdf', or 'auto')

        Returns:
            Dictionary with extracted prescription data

        Example:
            with open('prescription.pdf', 'rb') as f:
                data = client.extract_prescription_data(f)
                print(data['doctor_code'])
                print(data['patient_matricule'])
                print(data['medications'])
        """
        system_prompt = (
            "You are an OCR assistant specialized in Luxembourg medical prescriptions. "
            "Extract data and return ONLY a valid JSON object with these fields:\n"
            "- doctor_code: string (format: 90XXXX-XX or 90XXXXXX)\n"
            "- doctor_name: string (full name)\n"
            "- doctor_specialty: string or null\n"
            "- doctor_phone: string or null\n"
            "- doctor_address: string or null\n"
            "- patient_matricule: string (13 digits)\n"
            "- patient_name: string\n"
            "- prescription_date: string (YYYY-MM-DD format)\n"
            "- medications: array of strings (each medication with dosage)\n"
            "- notes: string (additional notes)\n"
            "Return ONLY the JSON object, no additional text."
        )

        user_prompt = (
            "Extract all prescription data from this image and return as JSON. "
            "Be thorough and accurate."
        )

        image_data_uri = self._image_to_base64(file_obj, file_type)
        response = self._make_request(
            system_prompt,
            user_prompt,
            image_data_uri,
            temperature=0.0,
            max_tokens=2000
        )

        # Extract and parse JSON from response
        content = response['choices'][0]['message']['content']

        # Try to extract JSON from response (model might add extra text)
        try:
            # Find JSON object in response
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
                data = json.loads(json_str)
            else:
                # Fallback: try parsing entire content
                data = json.loads(content)

            # Add metadata
            data['_metadata'] = {
                'tokens_used': response.get('usage', {}).get('total_tokens', 0),
                'model': response.get('model', 'unknown'),
                'raw_response': content
            }

            return data

        except json.JSONDecodeError as e:
            raise DeepSeekOCRError(
                f"Failed to parse JSON from OCR response. "
                f"Error: {str(e)}. Raw content: {content[:200]}"
            )

    def extract_text_multi_page(
        self,
        pdf_path: Union[str, Path],
        preserve_formatting: bool = True
    ) -> Dict[str, Any]:
        """
        Extract text from all pages of a multi-page PDF.

        Args:
            pdf_path: Path to PDF file
            preserve_formatting: Whether to preserve text structure

        Returns:
            Dictionary with 'pages' list and 'full_text' string

        Example:
            result = client.extract_text_multi_page('prescription.pdf')
            for i, page in enumerate(result['pages']):
                print(f"Page {i+1}: {page['text']}")
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            raise DeepSeekOCRError(
                "pdf2image library required for multi-page PDF support. "
                "Install with: pip install pdf2image"
            )

        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300)

        pages = []
        full_text = []

        for i, image in enumerate(images):
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)

            # Extract text from this page
            result = self.extract_text(
                img_byte_arr,
                file_type='image',
                preserve_formatting=preserve_formatting
            )

            pages.append({
                'page_number': i + 1,
                'text': result['text'],
                'metadata': result['metadata']
            })

            full_text.append(result['text'])

        return {
            'pages': pages,
            'full_text': '\n\n--- Page Break ---\n\n'.join(full_text),
            'total_pages': len(pages)
        }


# Django integration helper
class DjangoOCRService:
    """
    Django-specific wrapper for OCR service.

    Usage in Django settings:
        OCR_SERVICE_URL = env('OCR_SERVICE_URL', default='http://localhost:8765')
        OCR_SERVICE_API_KEY = env('OCR_SERVICE_API_KEY', default='')

    Usage in views:
        from nunoocr_client import DjangoOCRService

        ocr = DjangoOCRService()
        if ocr.is_available():
            result = ocr.extract_from_uploaded_file(request.FILES['prescription'])
    """

    def __init__(self):
        """Initialize with settings from Django configuration."""
        from django.conf import settings

        self.base_url = getattr(settings, 'OCR_SERVICE_URL', 'http://localhost:8765')
        self.api_key = getattr(settings, 'OCR_SERVICE_API_KEY', None)
        self.client = DeepSeekOCRClient(
            base_url=self.base_url,
            api_key=self.api_key
        )

    def is_available(self) -> bool:
        """Check if OCR service is available."""
        return self.client.health_check()

    def extract_from_uploaded_file(
        self,
        uploaded_file,
        extract_structured: bool = False
    ) -> Dict[str, Any]:
        """
        Extract text from Django UploadedFile object.

        Args:
            uploaded_file: Django UploadedFile from request.FILES
            extract_structured: If True, extract structured prescription data

        Returns:
            Dictionary with extracted data
        """
        file_type = 'pdf' if uploaded_file.name.lower().endswith('.pdf') else 'image'

        if extract_structured:
            return self.client.extract_prescription_data(uploaded_file, file_type)
        else:
            return self.client.extract_text(uploaded_file, file_type)
