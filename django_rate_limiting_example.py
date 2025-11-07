"""
Exemples de Rate Limiting pour protéger l'API Vision

Installs requis:
pip install django-ratelimit django-redis
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django_ratelimit.decorators import ratelimit
from nunoocr_client import DjangoOCRService
import time

# ============================================================================
# MÉTHODE 1: Rate Limit Simple avec django-ratelimit
# ============================================================================

@csrf_exempt
@login_required  # Nécessite authentification
@ratelimit(key='user', rate='10/h', method='POST')  # 10 requêtes par heure par utilisateur
def analyze_wound_with_ratelimit(request):
    """
    Analyse de plaie avec rate limiting strict
    - 10 analyses par heure par utilisateur
    - Authentification obligatoire
    """
    was_limited = getattr(request, 'limited', False)
    if was_limited:
        return JsonResponse({
            'error': 'Trop de requêtes',
            'detail': 'Maximum 10 analyses par heure. Réessayez plus tard.',
            'retry_after': '1 heure'
        }, status=429)

    if 'wound_image' not in request.FILES:
        return JsonResponse({'error': 'Image requise'}, status=400)

    ocr = DjangoOCRService()
    result = ocr.analyze_wound_from_uploaded_file(
        request.FILES['wound_image'],
        return_structured=True
    )

    return JsonResponse({'success': True, 'data': result})


# ============================================================================
# MÉTHODE 2: Rate Limit avec Cache Manuel (Plus de Contrôle)
# ============================================================================

@csrf_exempt
@login_required
def analyze_wound_with_custom_limit(request):
    """
    Rate limiting personnalisé avec suivi des coûts
    - Limite par utilisateur ET par IP
    - Tracking du nombre de tokens utilisés
    - Limites différentes selon le rôle utilisateur
    """
    user = request.user
    user_id = user.id
    ip_address = get_client_ip(request)

    # Vérifier limite par utilisateur (10/jour pour users normaux, 100/jour pour staff)
    daily_limit = 100 if user.is_staff else 10
    cache_key_user = f'wound_analysis_count_user_{user_id}_{time.strftime("%Y%m%d")}'
    user_count = cache.get(cache_key_user, 0)

    if user_count >= daily_limit:
        return JsonResponse({
            'error': 'Quota quotidien dépassé',
            'detail': f'Limite de {daily_limit} analyses par jour atteinte',
            'used': user_count,
            'limit': daily_limit,
            'reset': 'minuit'
        }, status=429)

    # Vérifier limite par IP (protection anti-abus)
    cache_key_ip = f'wound_analysis_ip_{ip_address}_{time.strftime("%Y%m%d%H")}'
    ip_count = cache.get(cache_key_ip, 0)

    if ip_count >= 20:  # Max 20 par IP par heure
        return JsonResponse({
            'error': 'Trop de requêtes depuis cette IP',
            'retry_after': '1 heure'
        }, status=429)

    # Effectuer l'analyse
    if 'wound_image' not in request.FILES:
        return JsonResponse({'error': 'Image requise'}, status=400)

    try:
        ocr = DjangoOCRService()
        result = ocr.analyze_wound_from_uploaded_file(
            request.FILES['wound_image'],
            return_structured=True
        )

        # Incrémenter les compteurs
        cache.set(cache_key_user, user_count + 1, 86400)  # 24h
        cache.set(cache_key_ip, ip_count + 1, 3600)  # 1h

        # Logger l'usage pour analytics
        log_api_usage(user, 'wound_analysis', result.get('_metadata', {}).get('tokens_used', 0))

        return JsonResponse({
            'success': True,
            'data': result,
            'usage': {
                'remaining_today': daily_limit - user_count - 1,
                'limit': daily_limit
            }
        })

    except Exception as e:
        return JsonResponse({
            'error': 'Analyse échouée',
            'detail': str(e)
        }, status=500)


# ============================================================================
# MÉTHODE 3: Système de Crédits
# ============================================================================

from django.db import models

class UserAPICredit(models.Model):
    """Modèle pour gérer les crédits API par utilisateur"""
    user = models.OneToOneField('auth.User', on_delete=models.CASCADE)
    credits_remaining = models.IntegerField(default=100)  # Crédits initiaux
    credits_used = models.IntegerField(default=0)
    last_refill = models.DateTimeField(auto_now_add=True)

    def has_credits(self, required=1):
        return self.credits_remaining >= required

    def use_credits(self, amount=1):
        if self.has_credits(amount):
            self.credits_remaining -= amount
            self.credits_used += amount
            self.save()
            return True
        return False

    def refill_credits(self, amount):
        self.credits_remaining += amount
        self.save()


@csrf_exempt
@login_required
def analyze_wound_with_credits(request):
    """
    Système de crédits:
    - Chaque utilisateur a un quota de crédits
    - 1 analyse = 1 crédit
    - Recharge mensuelle automatique ou manuelle
    """
    user = request.user

    # Récupérer ou créer les crédits utilisateur
    credits, created = UserAPICredit.objects.get_or_create(
        user=user,
        defaults={'credits_remaining': 100}
    )

    # Vérifier si l'utilisateur a des crédits
    if not credits.has_credits():
        return JsonResponse({
            'error': 'Crédits insuffisants',
            'credits_remaining': 0,
            'detail': 'Contactez l\'administrateur pour recharger vos crédits'
        }, status=402)  # 402 Payment Required

    if 'wound_image' not in request.FILES:
        return JsonResponse({'error': 'Image requise'}, status=400)

    try:
        ocr = DjangoOCRService()
        result = ocr.analyze_wound_from_uploaded_file(
            request.FILES['wound_image'],
            return_structured=True
        )

        # Déduire un crédit
        credits.use_credits(1)

        return JsonResponse({
            'success': True,
            'data': result,
            'credits': {
                'remaining': credits.credits_remaining,
                'used_total': credits.credits_used
            }
        })

    except Exception as e:
        return JsonResponse({
            'error': 'Analyse échouée',
            'detail': str(e)
        }, status=500)


# ============================================================================
# MÉTHODE 4: Permission par Rôle
# ============================================================================

from django.contrib.auth.models import Group

@csrf_exempt
@login_required
def analyze_wound_with_permissions(request):
    """
    Restreindre l'accès par rôle:
    - Seuls certains groupes peuvent utiliser l'API
    - Médecins, infirmières = accès
    - Autres = refusé
    """
    user = request.user

    # Vérifier si l'utilisateur a la permission
    allowed_groups = ['Medecins', 'Infirmieres', 'Personnel_Soignant']
    user_groups = user.groups.values_list('name', flat=True)

    if not any(group in allowed_groups for group in user_groups):
        return JsonResponse({
            'error': 'Accès refusé',
            'detail': 'Cette fonctionnalité est réservée au personnel soignant'
        }, status=403)

    # Procéder avec l'analyse...
    if 'wound_image' not in request.FILES:
        return JsonResponse({'error': 'Image requise'}, status=400)

    ocr = DjangoOCRService()
    result = ocr.analyze_wound_from_uploaded_file(
        request.FILES['wound_image'],
        return_structured=True
    )

    return JsonResponse({'success': True, 'data': result})


# ============================================================================
# HELPERS
# ============================================================================

def get_client_ip(request):
    """Obtenir l'IP réelle du client (avec support proxy)"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


def log_api_usage(user, endpoint, tokens_used):
    """Logger l'usage de l'API pour analytics et facturation"""
    from django.utils import timezone

    # Créer un log d'usage
    APIUsageLog.objects.create(
        user=user,
        endpoint=endpoint,
        tokens_used=tokens_used,
        cost_usd=tokens_used * 0.00001,  # Estimation coût
        timestamp=timezone.now()
    )


class APIUsageLog(models.Model):
    """Modèle pour tracker l'usage de l'API"""
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    endpoint = models.CharField(max_length=100)
    tokens_used = models.IntegerField()
    cost_usd = models.DecimalField(max_digits=10, decimal_places=6)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['user', '-timestamp']),
        ]


# ============================================================================
# CONFIGURATION SETTINGS.PY
# ============================================================================

"""
# Dans settings.py, ajoutez:

# Rate Limiting avec Redis
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
        'OPTIONS': {
            'CLIENT_CLASS': 'django_redis.client.DefaultClient',
        }
    }
}

# Limites globales
API_RATE_LIMITS = {
    'wound_analysis': {
        'anonymous': '0/day',  # Anonyme = interdit
        'authenticated': '10/day',  # Users normaux = 10/jour
        'staff': '100/day',  # Staff = 100/jour
        'superuser': '1000/day',  # Admin = 1000/jour
    }
}

# Budget mensuel maximum (en USD)
MAX_MONTHLY_API_COST = 100.00  # Alerter si dépassé

# Email d'alerte
API_ALERT_EMAIL = 'admin@example.com'
"""
