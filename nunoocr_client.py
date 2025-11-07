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
from typing import Dict, Any, Optional, BinaryIO, Union, List
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
        vision_api_key: Optional API key for vision models (GPT-4V/Claude)
        vision_provider: Vision model provider ('openai' or 'anthropic')
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        api_key: Optional[str] = None,
        timeout: int = 60,
        vision_api_key: Optional[str] = None,
        vision_provider: str = "openai"
    ):
        """
        Initialize the OCR client.

        Args:
            base_url: Base URL of the OCR service (for prescriptions)
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            vision_api_key: API key for vision model (GPT-4V or Claude)
            vision_provider: 'openai' for GPT-4 Vision or 'anthropic' for Claude
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
        self.health_endpoint = f"{self.base_url}/health"

        # Vision API configuration
        self.vision_api_key = vision_api_key
        self.vision_provider = vision_provider.lower()

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

    def _analyze_wound_with_vision_api(
        self,
        file_obj: BinaryIO,
        file_type: str = 'auto',
        return_structured: bool = True
    ) -> Dict[str, Any]:
        """Use GPT-4 Vision or Claude for wound analysis."""
        if not self.vision_api_key:
            raise DeepSeekOCRError(
                "Vision API key required for wound analysis. "
                "Initialize client with vision_api_key parameter."
            )

        # Convert image to base64
        file_obj.seek(0)
        if file_type == 'pdf':
            # Convert PDF first page to image
            try:
                from pdf2image import convert_from_bytes
                file_obj.seek(0)
                images = convert_from_bytes(file_obj.read(), dpi=300, first_page=1, last_page=1)
                if images:
                    img_byte_arr = io.BytesIO()
                    images[0].save(img_byte_arr, format='PNG')
                    img_bytes = img_byte_arr.getvalue()
                else:
                    raise DeepSeekOCRError("Failed to convert PDF to image")
            except ImportError:
                raise DeepSeekOCRError("pdf2image required for PDF support")
        else:
            file_obj.seek(0)
            img_bytes = file_obj.read()

        b64_image = base64.b64encode(img_bytes).decode('utf-8')

        if self.vision_provider == 'openai':
            return self._analyze_with_openai_vision(b64_image, return_structured)
        elif self.vision_provider == 'anthropic':
            return self._analyze_with_claude_vision(b64_image, return_structured)
        else:
            raise DeepSeekOCRError(f"Unknown vision provider: {self.vision_provider}")

    def _analyze_with_openai_vision(self, b64_image: str, return_structured: bool) -> Dict[str, Any]:
        """Analyze wound using GPT-4 Vision API."""
        if return_structured:
            system_prompt = (
                "Vous êtes un assistant d'aide à la documentation médicale pour professionnels de santé. "
                "Vous aidez à documenter et décrire des plaies pour le suivi médical. "
                "Ceci est utilisé par des médecins et infirmières qualifiés dans un contexte professionnel. "
                "Retournez UNIQUEMENT un objet JSON EN FRANÇAIS avec une description objective de la plaie visible."
            )
            user_prompt = (
                "Je suis un professionnel de santé documentant cette plaie pour le dossier patient. "
                "Aidez-moi à documenter cette image en retournant un JSON avec ces champs descriptifs:\n"
                "- type_plaie: string (ex: 'ulcère', 'lacération', 'plaie chirurgicale')\n"
                "- localisation: string (localisation anatomique visible)\n"
                "- dimensions: {longueur_cm: number approximatif, largeur_cm: number approximatif}\n"
                "- stade_cicatrisation: string ('fraîche', 'en cours de cicatrisation', 'cicatrisée')\n"
                "- methode_fermeture: string ('points de suture', 'agrafes', 'adhésif', 'aucune')\n"
                "- nombre_points: number ou null (si visible)\n"
                "- signes_infection: array de strings (observations visuelles)\n"
                "- complications: array de strings (observations visuelles)\n"
                "- etat_general: string (description objective de l'apparence)\n"
                "- confiance: string ('élevée', 'moyenne', 'faible')\n"
                "- notes: string (observations supplémentaires)\n"
                "Retournez UNIQUEMENT le JSON, sans texte supplémentaire."
            )
        else:
            system_prompt = "Vous êtes un expert médical spécialisé dans l'évaluation des plaies."
            user_prompt = (
                "Analysez cette plaie EN FRANÇAIS:\n"
                "1. Type et localisation\n"
                "2. Dimensions approximatives\n"
                "3. Stade de cicatrisation\n"
                "4. Signes d'infection\n"
                "5. État général\n"
                "6. Recommandations"
            )

        payload = {
            "model": "gpt-4o",  # or gpt-4-turbo, gpt-4o-mini for cost savings
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_image}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.0
        }

        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.vision_api_key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            content = result['choices'][0]['message']['content']

            if return_structured:
                try:
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        data = json.loads(json_str)
                    else:
                        data = json.loads(content)

                    data['_metadata'] = {
                        'tokens_used': result.get('usage', {}).get('total_tokens', 0),
                        'model': result.get('model', 'gpt-4-vision'),
                        'provider': 'openai'
                    }
                    return data
                except json.JSONDecodeError as e:
                    return {
                        'analysis': content,
                        'structured': False,
                        '_metadata': {
                            'tokens_used': result.get('usage', {}).get('total_tokens', 0),
                            'model': result.get('model', 'gpt-4-vision'),
                            'provider': 'openai',
                            'parse_error': str(e)
                        }
                    }
            else:
                return {
                    'analysis': content,
                    'structured': False,
                    '_metadata': {
                        'tokens_used': result.get('usage', {}).get('total_tokens', 0),
                        'model': result.get('model', 'gpt-4-vision'),
                        'provider': 'openai'
                    }
                }

        except requests.exceptions.RequestException as e:
            raise DeepSeekOCRError(f"OpenAI Vision API request failed: {str(e)}")

    def _analyze_with_claude_vision(self, b64_image: str, return_structured: bool) -> Dict[str, Any]:
        """Analyze wound using Claude Vision API."""
        if return_structured:
            system_prompt = (
                "Vous êtes un expert médical spécialisé dans l'évaluation des plaies. "
                "Analysez l'image et retournez UNIQUEMENT un objet JSON EN FRANÇAIS."
            )
            user_prompt = (
                "Analysez cette plaie et retournez un JSON avec ces champs:\n"
                "- type_plaie, localisation, dimensions {longueur_cm, largeur_cm}, "
                "stade_cicatrisation, methode_fermeture, nombre_points, "
                "signes_infection (array), complications (array), etat_general, confiance, notes"
            )
        else:
            system_prompt = "Vous êtes un expert médical spécialisé dans l'évaluation des plaies."
            user_prompt = "Analysez cette plaie en détail EN FRANÇAIS."

        payload = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 2000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{system_prompt}\n\n{user_prompt}"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64_image
                            }
                        }
                    ]
                }
            ]
        }

        try:
            response = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.vision_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            result = response.json()
            content = result['content'][0]['text']

            if return_structured:
                try:
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    if start_idx != -1 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        data = json.loads(json_str)
                    else:
                        data = json.loads(content)

                    data['_metadata'] = {
                        'tokens_used': result.get('usage', {}).get('input_tokens', 0) +
                                     result.get('usage', {}).get('output_tokens', 0),
                        'model': result.get('model', 'claude-3-sonnet'),
                        'provider': 'anthropic'
                    }
                    return data
                except json.JSONDecodeError as e:
                    return {
                        'analysis': content,
                        'structured': False,
                        '_metadata': {
                            'tokens_used': result.get('usage', {}).get('input_tokens', 0) +
                                         result.get('usage', {}).get('output_tokens', 0),
                            'model': result.get('model', 'claude-3-sonnet'),
                            'provider': 'anthropic',
                            'parse_error': str(e)
                        }
                    }
            else:
                return {
                    'analysis': content,
                    'structured': False,
                    '_metadata': {
                        'tokens_used': result.get('usage', {}).get('input_tokens', 0) +
                                     result.get('usage', {}).get('output_tokens', 0),
                        'model': result.get('model', 'claude-3-sonnet'),
                        'provider': 'anthropic'
                    }
                }

        except requests.exceptions.RequestException as e:
            raise DeepSeekOCRError(f"Claude Vision API request failed: {str(e)}")

    def analyze_wound(
        self,
        file_obj: BinaryIO,
        file_type: str = 'auto',
        return_structured: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a wound image and extract medical information in French.
        Uses GPT-4 Vision or Claude Vision API for accurate analysis.

        Args:
            file_obj: File object (image of wound)
            file_type: File type ('image', 'pdf', or 'auto')
            return_structured: If True, attempt to parse JSON response

        Returns:
            Dictionary with wound analysis data (in French)

        Example:
            # With OpenAI GPT-4 Vision
            client = DeepSeekOCRClient(
                vision_api_key="sk-...",
                vision_provider="openai"
            )
            with open('wound.jpg', 'rb') as f:
                analysis = client.analyze_wound(f)
                print(analysis['type_plaie'])
                print(analysis['dimensions'])
                print(analysis['etat_general'])
        """
        # Use vision API for wound analysis
        return self._analyze_wound_with_vision_api(file_obj, file_type, return_structured)

    def compare_wound_progress(
        self,
        wound_images: List[Dict[str, Any]],
        return_structured: bool = True
    ) -> Dict[str, Any]:
        """
        Compare multiple wound images over time to evaluate healing progress.

        Args:
            wound_images: List of dicts with keys:
                - 'file_obj': BinaryIO (image file object)
                - 'date': str (date in YYYY-MM-DD format or ISO format)
                - 'file_type': str (optional, 'image' or 'pdf', default 'auto')
                - 'notes': str (optional, additional context)
            return_structured: If True, attempt to parse JSON response

        Returns:
            Dictionary with progression analysis data (in French)

        Example:
            images = [
                {'file_obj': open('wound_day1.jpg', 'rb'), 'date': '2025-01-01'},
                {'file_obj': open('wound_day7.jpg', 'rb'), 'date': '2025-01-07'},
                {'file_obj': open('wound_day14.jpg', 'rb'), 'date': '2025-01-14'}
            ]
            progress = client.compare_wound_progress(images)
            print(progress['evolution_globale'])
            print(progress['ameliorations'])
            print(progress['preoccupations'])
        """
        if not wound_images or len(wound_images) < 2:
            raise DeepSeekOCRError(
                "Au moins 2 images sont requises pour l'analyse de progression"
            )

        # Sort images by date
        sorted_images = sorted(wound_images, key=lambda x: x['date'])

        # Analyze each image individually first
        analyses = []
        for i, img_data in enumerate(sorted_images):
            file_obj = img_data['file_obj']
            date = img_data['date']
            file_type = img_data.get('file_type', 'auto')

            # Reset file pointer
            file_obj.seek(0)

            # Get individual analysis
            analysis = self.analyze_wound(file_obj, file_type, return_structured=True)
            analyses.append({
                'date': date,
                'analysis': analysis,
                'notes': img_data.get('notes', '')
            })

        # Now create a comprehensive comparison prompt
        comparison_text = "ANALYSE CHRONOLOGIQUE DES PLAIES:\n\n"
        for i, item in enumerate(analyses, 1):
            comparison_text += f"Image {i} - Date: {item['date']}\n"
            if item['notes']:
                comparison_text += f"Notes: {item['notes']}\n"
            comparison_text += f"Type: {item['analysis'].get('type_plaie', 'N/A')}\n"
            comparison_text += f"Localisation: {item['analysis'].get('localisation', 'N/A')}\n"

            dims = item['analysis'].get('dimensions', {})
            if dims:
                comparison_text += f"Dimensions: {dims.get('longueur_cm', 'N/A')} x {dims.get('largeur_cm', 'N/A')} cm\n"

            comparison_text += f"Stade: {item['analysis'].get('stade_cicatrisation', 'N/A')}\n"
            comparison_text += f"État: {item['analysis'].get('etat_general', 'N/A')}\n"

            infections = item['analysis'].get('signes_infection', [])
            if infections:
                comparison_text += f"Signes d'infection: {', '.join(infections)}\n"

            comparison_text += "\n"

        if return_structured:
            system_prompt = (
                "Vous êtes un expert médical spécialisé dans l'évaluation de la progression des plaies. "
                "Analysez l'évolution des plaies à travers le temps et retournez UNIQUEMENT un objet JSON EN FRANÇAIS avec ces champs:\n"
                "- periode_analyse: string (ex: '14 jours', '3 semaines')\n"
                "- nombre_evaluations: number (nombre d'images analysées)\n"
                "- evolution_globale: string ('amélioration significative', 'amélioration modérée', 'stable', 'détérioration')\n"
                "- ameliorations: array de strings (aspects qui se sont améliorés)\n"
                "- preoccupations: array de strings (aspects préoccupants ou qui se sont détériorés)\n"
                "- changement_dimensions: object avec 'evolution' (string: 'réduction', 'stable', 'augmentation'), 'pourcentage' (number ou null)\n"
                "- cicatrisation_progression: string (description de la progression de cicatrisation)\n"
                "- infection_evolution: string ('amélioration', 'stable', 'détérioration', 'aucune infection')\n"
                "- recommandations: array de strings (recommandations médicales basées sur l'évolution)\n"
                "- prochain_controle: string (recommandation pour le prochain contrôle, ex: '3-5 jours', '1 semaine')\n"
                "- notes_progression: string (observations détaillées sur l'évolution)\n"
                "Retournez UNIQUEMENT l'objet JSON. TOUT doit être en français."
            )

            user_prompt = (
                f"{comparison_text}\n\n"
                "Comparez ces observations chronologiques et évaluez la progression de la plaie. "
                "Fournissez une analyse détaillée de l'évolution EN FRANÇAIS sous forme de JSON."
            )
        else:
            system_prompt = (
                "Vous êtes un expert médical spécialisé dans l'évaluation de la progression des plaies. "
                "Fournissez des analyses détaillées et précises de l'évolution des plaies en français."
            )

            user_prompt = (
                f"{comparison_text}\n\n"
                "Comparez ces observations chronologiques et fournissez EN FRANÇAIS:\n"
                "1. Évaluation globale de l'évolution (amélioration/stable/détérioration)\n"
                "2. Aspects qui se sont améliorés\n"
                "3. Aspects préoccupants ou détériorés\n"
                "4. Changements dans les dimensions de la plaie\n"
                "5. Progression de la cicatrisation\n"
                "6. Évolution des signes d'infection\n"
                "7. Recommandations médicales\n"
                "8. Recommandation pour le prochain contrôle\n"
                "9. Observations détaillées sur l'évolution"
            )

        # Use the first image as reference for the API call
        # (We're mainly using the text comparison here)
        sorted_images[0]['file_obj'].seek(0)
        image_data_uri = self._image_to_base64(
            sorted_images[0]['file_obj'],
            sorted_images[0].get('file_type', 'auto')
        )

        response = self._make_request(
            system_prompt,
            user_prompt,
            image_data_uri,
            temperature=0.0,
            max_tokens=3000
        )

        # Extract content from response
        content = response['choices'][0]['message']['content']

        if return_structured:
            try:
                # Find JSON object in response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1

                if start_idx != -1 and end_idx > start_idx:
                    json_str = content[start_idx:end_idx]
                    data = json.loads(json_str)
                else:
                    data = json.loads(content)

                # Add analyses and metadata
                data['analyses_individuelles'] = analyses
                data['_metadata'] = {
                    'tokens_used': response.get('usage', {}).get('total_tokens', 0),
                    'model': response.get('model', 'unknown'),
                    'raw_response': content
                }

                return data

            except json.JSONDecodeError as e:
                return {
                    'comparison': content,
                    'analyses_individuelles': analyses,
                    'structured': False,
                    '_metadata': {
                        'tokens_used': response.get('usage', {}).get('total_tokens', 0),
                        'model': response.get('model', 'unknown'),
                        'parse_error': str(e)
                    }
                }
        else:
            return {
                'comparison': content,
                'analyses_individuelles': analyses,
                'structured': False,
                '_metadata': {
                    'tokens_used': response.get('usage', {}).get('total_tokens', 0),
                    'model': response.get('model', 'unknown')
                }
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

        # Vision API configuration for wound analysis
        self.vision_api_key = getattr(settings, 'VISION_API_KEY', None)
        self.vision_provider = getattr(settings, 'VISION_PROVIDER', 'openai')

        self.client = DeepSeekOCRClient(
            base_url=self.base_url,
            api_key=self.api_key,
            vision_api_key=self.vision_api_key,
            vision_provider=self.vision_provider
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

    def analyze_wound_from_uploaded_file(
        self,
        uploaded_file,
        return_structured: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze wound from Django UploadedFile object.

        Args:
            uploaded_file: Django UploadedFile from request.FILES
            return_structured: If True, return structured JSON data

        Returns:
            Dictionary with wound analysis data

        Example:
            ocr = DjangoOCRService()
            analysis = ocr.analyze_wound_from_uploaded_file(
                request.FILES['wound_image']
            )
        """
        file_type = 'pdf' if uploaded_file.name.lower().endswith('.pdf') else 'image'
        return self.client.analyze_wound(uploaded_file, file_type, return_structured)

    def compare_wound_progress_from_model(
        self,
        wound_assessments_queryset,
        return_structured: bool = True
    ) -> Dict[str, Any]:
        """
        Compare wound progression from Django model queryset.

        Args:
            wound_assessments_queryset: Django queryset of wound assessment objects
                                       Must have 'image' and 'analyzed_at' or 'date' field
            return_structured: If True, return structured JSON data

        Returns:
            Dictionary with progression analysis

        Example:
            from myapp.models import WoundAssessment

            # Get all assessments for a patient, ordered by date
            assessments = WoundAssessment.objects.filter(
                patient=patient
            ).order_by('analyzed_at')

            ocr = DjangoOCRService()
            progress = ocr.compare_wound_progress_from_model(assessments)
        """
        if wound_assessments_queryset.count() < 2:
            raise DeepSeekOCRError(
                "Au moins 2 évaluations sont requises pour l'analyse de progression"
            )

        wound_images = []
        for assessment in wound_assessments_queryset:
            # Get date from model (try common field names)
            date = None
            for date_field in ['analyzed_at', 'date', 'created_at', 'assessment_date']:
                if hasattr(assessment, date_field):
                    date_value = getattr(assessment, date_field)
                    if date_value:
                        # Convert to string format
                        if hasattr(date_value, 'isoformat'):
                            date = date_value.isoformat()
                        else:
                            date = str(date_value)
                        break

            if not date:
                raise DeepSeekOCRError(
                    "Unable to find date field in wound assessment model. "
                    "Expected fields: analyzed_at, date, created_at, or assessment_date"
                )

            # Open image file
            if hasattr(assessment.image, 'open'):
                assessment.image.open()
                file_obj = assessment.image.file
            else:
                file_obj = assessment.image

            wound_images.append({
                'file_obj': file_obj,
                'date': date,
                'file_type': 'image',
                'notes': getattr(assessment, 'notes', '')
            })

        return self.client.compare_wound_progress(wound_images, return_structured)
