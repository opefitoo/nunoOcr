"""
Django Integration avec nunoOcr comme Microservice Centralisé

Architecture:
    Django → nunoOcr Service (port 8765) → OpenAI/Claude

Avantages:
- Django ne connaît jamais les clés OpenAI/Claude
- Changement de technologie? Seulement modifier nunoOcr
- Service centralisé pour tous les clients
- Cache possible dans nunoOcr
"""

import requests
from typing import Dict, Any, Optional, BinaryIO, List
from django.core.files.uploadedfile import UploadedFile


class NunoOcrServiceError(Exception):
    """Exception pour les erreurs du service nunoOcr."""
    pass


class NunoOcrServiceClient:
    """
    Client pour le service nunoOcr centralisé.

    Ce client appelle le service nunoOcr qui gère:
    - OCR pour prescriptions (DeepSeek-OCR)
    - Analyse de plaies (GPT-4 Vision / Claude)

    Usage dans Django:
        client = NunoOcrServiceClient(base_url="http://nunoocr:8765")
        result = client.analyze_wound(request.FILES['wound_image'])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8765",
        service_api_key: Optional[str] = None,
        timeout: int = 120
    ):
        """
        Initialize client.

        Args:
            base_url: URL du service nunoOcr (ex: http://nunoocr:8765)
            service_api_key: API Key pour authentifier auprès du service nunoOcr
            timeout: Timeout en secondes
        """
        self.base_url = base_url.rstrip('/')
        self.service_api_key = service_api_key
        self.timeout = timeout

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers including service API key if configured."""
        headers = {}
        if self.service_api_key:
            headers['Authorization'] = f'Bearer {self.service_api_key}'
        return headers

    def health_check(self) -> Dict[str, Any]:
        """
        Vérifier l'état du service nunoOcr.

        Returns:
            Status du service avec infos sur OCR et Vision APIs
        """
        try:
            response = requests.get(
                f"{self.base_url}/health",
                timeout=5
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise NunoOcrServiceError(f"Service health check failed: {e}")

    def analyze_wound(
        self,
        wound_image: UploadedFile
    ) -> Dict[str, Any]:
        """
        Analyser une plaie via le service nunoOcr.

        Le service nunoOcr appelle OpenAI/Claude en interne.
        Django n'a jamais besoin de connaître les clés API vision.

        Args:
            wound_image: Fichier Django UploadedFile

        Returns:
            Dict avec l'analyse structurée en français:
            {
                "success": True,
                "data": {
                    "type_plaie": "ulcère de pression",
                    "localisation": "cheville gauche",
                    "dimensions": {"longueur_cm": 2.5, "largeur_cm": 2.0},
                    "stade_cicatrisation": "en cours",
                    "signes_infection": [...],
                    "etat_general": "...",
                    "confiance": "élevée"
                }
            }

        Raises:
            NunoOcrServiceError: Si l'analyse échoue
        """
        try:
            # Préparer le fichier pour l'upload
            files = {
                'wound_image': (
                    wound_image.name,
                    wound_image.read(),
                    wound_image.content_type
                )
            }

            # Appeler le service nunoOcr avec auth
            response = requests.post(
                f"{self.base_url}/v1/analyze-wound",
                files=files,
                headers=self._get_headers(),
                timeout=self.timeout
            )

            response.raise_for_status()
            return response.json()

        except requests.exceptions.Timeout:
            raise NunoOcrServiceError("Service timeout - l'analyse a pris trop de temps")
        except requests.exceptions.ConnectionError:
            raise NunoOcrServiceError("Impossible de se connecter au service nunoOcr")
        except requests.exceptions.HTTPError as e:
            error_detail = e.response.text if hasattr(e, 'response') else str(e)
            raise NunoOcrServiceError(f"Service error: {error_detail}")
        except Exception as e:
            raise NunoOcrServiceError(f"Unexpected error: {e}")

    def compare_wound_progress(
        self,
        images_with_dates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Comparer plusieurs images de plaies dans le temps.

        Args:
            images_with_dates: Liste de dict avec 'file' (UploadedFile) et 'date' (str)
                Exemple:
                [
                    {'file': uploaded_file1, 'date': '2025-01-01'},
                    {'file': uploaded_file2, 'date': '2025-01-07'},
                ]

        Returns:
            Dict avec progression:
            {
                "success": True,
                "data": {
                    "total_images": 2,
                    "images": [
                        {"date": "2025-01-01", "analysis": {...}},
                        {"date": "2025-01-07", "analysis": {...}}
                    ],
                    "progression_notes": "..."
                }
            }
        """
        try:
            # Préparer les fichiers
            files = []
            dates = []

            for item in images_with_dates:
                uploaded_file = item['file']
                files.append((
                    'images',
                    (uploaded_file.name, uploaded_file.read(), uploaded_file.content_type)
                ))
                dates.append(item.get('date', ''))

            # Données
            data = {
                'dates': ','.join(dates)
            }

            # Appeler le service avec auth
            response = requests.post(
                f"{self.base_url}/v1/compare-wound-progress",
                files=files,
                data=data,
                headers=self._get_headers(),
                timeout=self.timeout * 2  # Plus de temps pour plusieurs images
            )

            response.raise_for_status()
            return response.json()

        except Exception as e:
            raise NunoOcrServiceError(f"Progression analysis failed: {e}")

    def extract_prescription_text(
        self,
        prescription_file: UploadedFile,
        return_structured: bool = True
    ) -> Dict[str, Any]:
        """
        Extraire le texte d'une prescription (OCR).

        Utilise DeepSeek-OCR du service nunoOcr.

        Args:
            prescription_file: Image/PDF de prescription
            return_structured: Si True, retourne des données structurées

        Returns:
            Texte extrait ou données structurées
        """
        # TODO: Implement prescription OCR endpoint in nunoOcr service
        raise NotImplementedError("Prescription OCR endpoint not yet implemented in service")


# ============================================================================
# DJANGO VIEWS - Exemples d'Intégration
# ============================================================================

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings


# Créer une instance globale du client
# Configuré avec l'URL et la clé de service
NUNOOCR_SERVICE = NunoOcrServiceClient(
    base_url=getattr(settings, 'NUNOOCR_SERVICE_URL', 'http://localhost:8765'),
    service_api_key=getattr(settings, 'NUNOOCR_SERVICE_API_KEY', None)
)


@csrf_exempt
@require_POST
def analyze_wound_view(request):
    """
    View Django pour analyser une plaie.

    Usage:
        POST /api/analyze-wound/
        Body: multipart/form-data avec wound_image

    L'analyse est faite via le service nunoOcr qui appelle OpenAI/Claude.
    Django n'a jamais besoin de la clé API vision!
    """
    if 'wound_image' not in request.FILES:
        return JsonResponse({'error': 'Image requise'}, status=400)

    wound_image = request.FILES['wound_image']

    # Validation taille
    if wound_image.size > 5 * 1024 * 1024:
        return JsonResponse({'error': 'Image trop grande (max 5MB)'}, status=400)

    try:
        # Appeler le service nunoOcr
        result = NUNOOCR_SERVICE.analyze_wound(wound_image)

        # Vérifier le résultat
        if not result.get('success'):
            return JsonResponse({
                'error': 'Analyse échouée',
                'detail': result.get('error')
            }, status=500)

        # Retourner les données
        return JsonResponse({
            'success': True,
            'data': result['data']
        })

    except NunoOcrServiceError as e:
        return JsonResponse({
            'error': 'Service nunoOcr error',
            'detail': str(e)
        }, status=503)
    except Exception as e:
        return JsonResponse({
            'error': 'Unexpected error',
            'detail': str(e)
        }, status=500)


@csrf_exempt
@require_POST
def compare_wounds_view(request):
    """
    View Django pour comparer plusieurs plaies dans le temps.

    Usage:
        POST /api/compare-wounds/
        Body: multipart/form-data
            - wound_1, wound_2, wound_3 (files)
            - date_1, date_2, date_3 (strings)
    """
    # Récupérer les images
    images_with_dates = []

    i = 1
    while f'wound_{i}' in request.FILES:
        wound_file = request.FILES[f'wound_{i}']
        date = request.POST.get(f'date_{i}', f'Image {i}')

        images_with_dates.append({
            'file': wound_file,
            'date': date
        })
        i += 1

    if len(images_with_dates) < 2:
        return JsonResponse({
            'error': 'Au moins 2 images requises'
        }, status=400)

    try:
        # Appeler le service
        result = NUNOOCR_SERVICE.compare_wound_progress(images_with_dates)

        return JsonResponse(result)

    except NunoOcrServiceError as e:
        return JsonResponse({
            'error': 'Service error',
            'detail': str(e)
        }, status=503)


# ============================================================================
# DJANGO VIEWS AVEC AUTHENTIFICATION API KEY
# ============================================================================

from functools import wraps


def require_api_key(view_func):
    """
    Decorator pour vérifier l'API Key avant d'appeler le service nunoOcr.

    Usage:
        @require_api_key
        def my_view(request, api_key):
            # api_key est l'objet APIKey validé
            ...
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        # Récupérer la clé
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')

        if not auth_header.startswith('Bearer '):
            return JsonResponse({
                'error': 'Authorization header requis',
                'format': 'Authorization: Bearer nuno_xxxxx'
            }, status=401)

        api_key_string = auth_header[7:]

        # Valider la clé (importer votre modèle APIKey)
        try:
            from .models import APIKey

            api_key = APIKey.objects.select_related('user').get(
                key=api_key_string,
                is_active=True
            )
        except APIKey.DoesNotExist:
            return JsonResponse({
                'error': 'API Key invalide'
            }, status=401)

        # Vérifier quota
        can_use, error = api_key.can_make_request()
        if not can_use:
            return JsonResponse({
                'error': error,
                'daily_limit': api_key.daily_limit,
                'calls_today': api_key.calls_today
            }, status=429)

        # Enregistrer l'utilisation
        api_key.record_usage()

        # Injecter dans request
        request.api_key = api_key
        request.user = api_key.user

        return view_func(request, api_key=api_key, *args, **kwargs)

    return wrapper


@csrf_exempt
@require_POST
@require_api_key
def analyze_wound_protected(request, api_key):
    """
    View protégée avec API Key + appel au service nunoOcr.

    Usage:
        curl -X POST https://inur.opefitoo.com/api/analyze-wound/ \
             -H "Authorization: Bearer nuno_xxxxx" \
             -F "wound_image=@wound.jpg"

    Architecture:
        1. Django vérifie l'API Key (quota, validité)
        2. Django appelle le service nunoOcr
        3. nunoOcr appelle OpenAI/Claude
        4. Django retourne le résultat
    """
    if 'wound_image' not in request.FILES:
        return JsonResponse({'error': 'Image requise'}, status=400)

    wound_image = request.FILES['wound_image']

    # Validation
    if wound_image.size > 5 * 1024 * 1024:
        return JsonResponse({'error': 'Image trop grande (max 5MB)'}, status=400)

    try:
        # Appeler le service nunoOcr (qui appelle OpenAI/Claude)
        result = NUNOOCR_SERVICE.analyze_wound(wound_image)

        if not result.get('success'):
            return JsonResponse({
                'error': 'Analyse échouée',
                'detail': result.get('error')
            }, status=500)

        # Retourner avec info API Key
        return JsonResponse({
            'success': True,
            'data': result['data'],
            'api_key_name': api_key.name,
            'remaining_calls_today': api_key.daily_limit - api_key.calls_today
        })

    except NunoOcrServiceError as e:
        return JsonResponse({
            'error': 'Service nunoOcr unavailable',
            'detail': str(e)
        }, status=503)
    except Exception as e:
        return JsonResponse({
            'error': 'Unexpected error',
            'detail': str(e)
        }, status=500)


# ============================================================================
# CONFIGURATION SETTINGS.PY
# ============================================================================

"""
Ajouter dans settings.py:

# URL du service nunoOcr
NUNOOCR_SERVICE_URL = os.getenv('NUNOOCR_SERVICE_URL', 'http://localhost:8765')

# Service API Key pour authentifier auprès de nunoOcr
NUNOOCR_SERVICE_API_KEY = os.getenv('NUNOOCR_SERVICE_API_KEY')  # REQUIS!

# Exemples selon déploiement:
# - Si déployé avec Docker Compose: 'http://nunoocr:8765'
# - Si service externe: 'http://46.224.6.193:8765'
# - Si via domain: 'https://nunoocr.opefitoo.com'
"""


# ============================================================================
# CONFIGURATION DOCKER COMPOSE
# ============================================================================

"""
docker-compose.yml:

version: '3.8'

services:
  nunoocr:
    image: nunoocr:latest
    ports:
      - "8765:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VISION_PROVIDER=openai
      - MODEL_NAME=deepseek-ai/DeepSeek-OCR
    volumes:
      - model-cache:/root/.cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  django:
    build: .
    depends_on:
      - nunoocr
    environment:
      - NUNOOCR_SERVICE_URL=http://nunoocr:8000
      # Pas besoin de OPENAI_API_KEY ici! C'est dans nunoocr
    ports:
      - "8000:8000"

volumes:
  model-cache:
"""


# ============================================================================
# TESTS
# ============================================================================

def test_service_connection():
    """Tester la connexion au service nunoOcr."""
    client = NunoOcrServiceClient(base_url="http://localhost:8765")

    try:
        health = client.health_check()
        print("Service nunoOcr status:")
        print(f"  OCR ready: {health.get('ocr_ready')}")
        print(f"  Vision configured: {health.get('vision_configured')}")
        print(f"  Vision provider: {health.get('vision_provider')}")
        return True
    except NunoOcrServiceError as e:
        print(f"Service error: {e}")
        return False


if __name__ == '__main__':
    # Test
    print("Testing nunoOcr service connection...")
    test_service_connection()
