"""
Authentification par API Key pour l'analyse de plaies

Installation:
pip install djangorestframework

Ou sans DRF (version simple avec modèle custom)
"""

# ============================================================================
# MÉTHODE 1: Avec Django REST Framework (Recommandé)
# ============================================================================

from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from django_ratelimit.decorators import ratelimit
from django.views.decorators.csrf import csrf_exempt
from nunoocr_client import DjangoOCRService

@csrf_exempt
@api_view(['POST'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
@ratelimit(key='user', rate='10/d', method='POST')
def analyze_wound_api(request):
    """
    Endpoint API avec authentification par token

    Header requis:
        Authorization: Token abc123def456...

    Body:
        - wound_image: fichier image

    Usage:
        curl -X POST https://api.example.com/analyze-wound/ \
             -H "Authorization: Token abc123def456..." \
             -F "wound_image=@wound.jpg"
    """
    # Check rate limit
    if getattr(request, 'limited', False):
        return Response({
            'error': 'Quota quotidien dépassé',
            'detail': 'Maximum 10 analyses par jour',
            'retry_after': '24 heures'
        }, status=status.HTTP_429_TOO_MANY_REQUESTS)

    # Vérifier le fichier
    if 'wound_image' not in request.FILES:
        return Response({
            'error': 'Image requise',
            'detail': 'Le champ "wound_image" est obligatoire'
        }, status=status.HTTP_400_BAD_REQUEST)

    file = request.FILES['wound_image']

    # Validation taille
    if file.size > 5 * 1024 * 1024:  # 5MB
        return Response({
            'error': 'Fichier trop volumineux',
            'detail': 'Taille maximum: 5MB',
            'size_received': f'{file.size / 1024 / 1024:.2f}MB'
        }, status=status.HTTP_400_BAD_REQUEST)

    try:
        # Analyse
        ocr = DjangoOCRService()
        result = ocr.analyze_wound_from_uploaded_file(
            file,
            return_structured=True
        )

        # Nettoyer la réponse (retirer metadata interne)
        clean_result = {
            'type_plaie': result.get('type_plaie'),
            'localisation': result.get('localisation'),
            'dimensions': result.get('dimensions'),
            'stade_cicatrisation': result.get('stade_cicatrisation'),
            'methode_fermeture': result.get('methode_fermeture'),
            'nombre_points': result.get('nombre_points'),
            'signes_infection': result.get('signes_infection', []),
            'complications': result.get('complications', []),
            'etat_general': result.get('etat_general'),
            'confiance': result.get('confiance'),
            'notes': result.get('notes'),
        }

        return Response({
            'success': True,
            'data': clean_result,
            'user': request.user.username
        }, status=status.HTTP_200_OK)

    except Exception as e:
        return Response({
            'error': 'Analyse échouée',
            'detail': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# ============================================================================
# MÉTHODE 2: Custom API Key (Sans DRF)
# ============================================================================

from django.db import models
from django.contrib.auth.models import User
import secrets
from django.utils import timezone

class APIKey(models.Model):
    """
    Modèle pour gérer les API Keys
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='api_keys')
    key = models.CharField(max_length=64, unique=True, db_index=True)
    name = models.CharField(max_length=100, help_text="Nom descriptif (ex: 'Production Server')")

    # Quotas
    daily_limit = models.IntegerField(default=10)
    calls_today = models.IntegerField(default=0)
    last_reset = models.DateField(auto_now_add=True)

    # Statistiques
    total_calls = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used = models.DateTimeField(null=True, blank=True)

    # Statut
    is_active = models.BooleanField(default=True)
    expires_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.user.username})"

    @staticmethod
    def generate_key():
        """Générer une clé API aléatoire"""
        return f"nuno_{secrets.token_urlsafe(32)}"

    def reset_daily_count_if_needed(self):
        """Réinitialiser le compteur quotidien si nécessaire"""
        from datetime import date
        if self.last_reset < date.today():
            self.calls_today = 0
            self.last_reset = date.today()
            self.save()

    def can_make_request(self):
        """Vérifier si la clé peut faire une requête"""
        if not self.is_active:
            return False, "Clé API désactivée"

        if self.expires_at and self.expires_at < timezone.now():
            return False, "Clé API expirée"

        self.reset_daily_count_if_needed()

        if self.calls_today >= self.daily_limit:
            return False, f"Quota quotidien dépassé ({self.daily_limit}/jour)"

        return True, None

    def record_usage(self):
        """Enregistrer l'utilisation de la clé"""
        self.calls_today += 1
        self.total_calls += 1
        self.last_used = timezone.now()
        self.save()


# Decorator pour authentification par API Key
from functools import wraps
from django.http import JsonResponse

def require_api_key(view_func):
    """
    Decorator pour requérir une API Key valide

    Usage:
        @require_api_key
        def my_view(request, api_key):
            # api_key est l'objet APIKey validé
            ...
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        # Récupérer la clé depuis le header
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')

        if not auth_header.startswith('Bearer '):
            return JsonResponse({
                'error': 'Authentification requise',
                'detail': 'Header "Authorization: Bearer YOUR_API_KEY" manquant'
            }, status=401)

        api_key_string = auth_header[7:]  # Retirer "Bearer "

        # Chercher la clé
        try:
            api_key = APIKey.objects.select_related('user').get(
                key=api_key_string
            )
        except APIKey.DoesNotExist:
            return JsonResponse({
                'error': 'API Key invalide',
                'detail': 'Cette clé n\'existe pas ou a été révoquée'
            }, status=401)

        # Vérifier si la clé peut être utilisée
        can_use, error_message = api_key.can_make_request()
        if not can_use:
            return JsonResponse({
                'error': 'Accès refusé',
                'detail': error_message
            }, status=429 if 'Quota' in error_message else 403)

        # Enregistrer l'utilisation
        api_key.record_usage()

        # Ajouter la clé à la request
        request.api_key = api_key
        request.user = api_key.user

        # Appeler la view
        return view_func(request, api_key=api_key, *args, **kwargs)

    return wrapper


# View avec custom API Key
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

@csrf_exempt
@require_POST
@require_api_key
def analyze_wound_with_api_key(request, api_key):
    """
    Endpoint avec authentification par API Key custom

    Usage:
        curl -X POST https://api.example.com/api/analyze-wound/ \
             -H "Authorization: Bearer nuno_abc123def456..." \
             -F "wound_image=@wound.jpg"
    """
    if 'wound_image' not in request.FILES:
        return JsonResponse({
            'error': 'Image requise'
        }, status=400)

    file = request.FILES['wound_image']

    # Validation taille
    if file.size > 5 * 1024 * 1024:
        return JsonResponse({
            'error': 'Fichier trop volumineux (max 5MB)'
        }, status=400)

    try:
        ocr = DjangoOCRService()
        result = ocr.analyze_wound_from_uploaded_file(file)

        return JsonResponse({
            'success': True,
            'data': result,
            'api_key_name': api_key.name,
            'remaining_calls_today': api_key.daily_limit - api_key.calls_today
        })

    except Exception as e:
        return JsonResponse({
            'error': 'Analyse échouée',
            'detail': str(e)
        }, status=500)


# ============================================================================
# COMMANDES DE GESTION DES API KEYS
# ============================================================================

# management/commands/create_api_key.py
from django.core.management.base import BaseCommand

class Command(BaseCommand):
    help = 'Créer une nouvelle API Key pour un utilisateur'

    def add_arguments(self, parser):
        parser.add_argument('username', type=str, help='Username')
        parser.add_argument('--name', type=str, default='Default Key', help='Nom de la clé')
        parser.add_argument('--limit', type=int, default=10, help='Limite quotidienne')

    def handle(self, *args, **options):
        from django.contrib.auth.models import User

        try:
            user = User.objects.get(username=options['username'])
        except User.DoesNotExist:
            self.stdout.write(self.style.ERROR(f'User {options["username"]} not found'))
            return

        api_key = APIKey.objects.create(
            user=user,
            key=APIKey.generate_key(),
            name=options['name'],
            daily_limit=options['limit']
        )

        self.stdout.write(self.style.SUCCESS(f'API Key created successfully!'))
        self.stdout.write(f'Key: {api_key.key}')
        self.stdout.write(f'User: {user.username}')
        self.stdout.write(f'Daily limit: {api_key.daily_limit}')


# ============================================================================
# ADMIN INTERFACE
# ============================================================================

from django.contrib import admin

@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'is_active', 'calls_today', 'daily_limit',
                    'total_calls', 'last_used', 'created_at']
    list_filter = ['is_active', 'created_at']
    search_fields = ['name', 'user__username', 'key']
    readonly_fields = ['key', 'created_at', 'last_used', 'total_calls', 'calls_today']

    fieldsets = (
        ('Informations', {
            'fields': ('user', 'name', 'key')
        }),
        ('Quotas', {
            'fields': ('daily_limit', 'calls_today', 'last_reset', 'total_calls')
        }),
        ('Statut', {
            'fields': ('is_active', 'expires_at', 'last_used', 'created_at')
        }),
    )

    def save_model(self, request, obj, form, change):
        if not change:  # Nouvelle clé
            obj.key = APIKey.generate_key()
        super().save_model(request, obj, form, change)


# ============================================================================
# URLS
# ============================================================================

"""
# urls.py

from django.urls import path
from . import views

urlpatterns = [
    # Avec DRF Token
    path('api/analyze-wound/', views.analyze_wound_api),

    # Avec Custom API Key
    path('api/v2/analyze-wound/', views.analyze_wound_with_api_key),
]
"""


# ============================================================================
# TESTS
# ============================================================================

"""
# Test avec curl

# 1. Créer une API Key
python manage.py create_api_key myusername --name "Test Key" --limit 10

# 2. Tester l'endpoint
curl -X POST http://localhost:8000/api/v2/analyze-wound/ \
     -H "Authorization: Bearer nuno_abc123def456..." \
     -F "wound_image=@wound.jpg"

# Réponse:
{
    "success": true,
    "data": {
        "type_plaie": "ulcère de pression",
        "localisation": "cheville gauche",
        ...
    },
    "api_key_name": "Test Key",
    "remaining_calls_today": 9
}
"""
