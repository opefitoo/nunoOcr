# 🔒 Sécurisation de l'API Vision - Guide Complet

Ce guide explique comment protéger votre API d'analyse de plaies contre les abus et contrôler les coûts.

## ⚠️ Risques Sans Protection

Sans protection, vous risquez:
- 💸 **Coûts incontrôlés** - Quelqu'un peut épuiser vos crédits OpenAI en quelques heures
- 🚨 **Abus** - Utilisation massive non autorisée
- 🐌 **Performance** - Surcharge du système
- 📊 **Pas de traçabilité** - Impossible de savoir qui utilise quoi

## 🛡️ Solutions de Protection

### 1️⃣ Authentification Obligatoire (ESSENTIEL)

**Toujours** exiger l'authentification:

```python
from django.contrib.auth.decorators import login_required

@login_required  # ← ESSENTIEL
def analyze_wound_view(request):
    # Seuls les utilisateurs connectés peuvent accéder
    ...
```

### 2️⃣ Rate Limiting (CRITIQUE)

#### Option A: Simple avec django-ratelimit

**Installation:**
```bash
pip install django-ratelimit
```

**Usage:**
```python
from django_ratelimit.decorators import ratelimit

@ratelimit(key='user', rate='10/h', method='POST')
def analyze_wound_view(request):
    """
    Limite: 10 analyses par heure par utilisateur
    """
    was_limited = getattr(request, 'limited', False)
    if was_limited:
        return JsonResponse({
            'error': 'Trop de requêtes. Maximum 10/heure.'
        }, status=429)
```

**Limites recommandées:**
```python
# Pour utilisateurs normaux
@ratelimit(key='user', rate='10/day')  # 10 par jour

# Pour staff
@ratelimit(key='user', rate='100/day')  # 100 par jour
```

#### Option B: Cache Manuel (Plus de contrôle)

```python
from django.core.cache import cache

def check_rate_limit(user_id, limit_per_day=10):
    """Vérifier la limite quotidienne"""
    cache_key = f'wound_analysis_{user_id}_{date.today()}'
    count = cache.get(cache_key, 0)

    if count >= limit_per_day:
        return False, count

    cache.set(cache_key, count + 1, 86400)  # 24h
    return True, count + 1

@login_required
def analyze_wound_view(request):
    allowed, count = check_rate_limit(request.user.id)

    if not allowed:
        return JsonResponse({
            'error': 'Quota quotidien dépassé',
            'used': count,
            'limit': 10,
            'reset': 'minuit'
        }, status=429)
```

### 3️⃣ Système de Crédits

**Migration:**
```python
# models.py
class UserAPICredit(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    credits_remaining = models.IntegerField(default=100)
    credits_used = models.IntegerField(default=0)

    def can_use(self):
        return self.credits_remaining > 0

    def consume(self, amount=1):
        if self.can_use():
            self.credits_remaining -= amount
            self.credits_used += amount
            self.save()
            return True
        return False
```

**View:**
```python
@login_required
def analyze_wound_view(request):
    credits = UserAPICredit.objects.get(user=request.user)

    if not credits.can_use():
        return JsonResponse({
            'error': 'Crédits insuffisants',
            'credits_remaining': 0
        }, status=402)

    # Analyse...
    credits.consume(1)
```

### 4️⃣ Permissions par Rôle

```python
from django.contrib.auth.models import Group

@login_required
def analyze_wound_view(request):
    # Seuls médecins et infirmières
    if not request.user.groups.filter(
        name__in=['Medecins', 'Infirmieres']
    ).exists():
        return JsonResponse({
            'error': 'Accès refusé - Personnel soignant uniquement'
        }, status=403)
```

### 5️⃣ Validation de Fichier

```python
def validate_wound_image(file):
    """Valider l'image uploadée"""
    # Taille max: 5MB
    if file.size > 5 * 1024 * 1024:
        raise ValidationError('Image trop grande (max 5MB)')

    # Types autorisés
    allowed_types = ['image/jpeg', 'image/png']
    if file.content_type not in allowed_types:
        raise ValidationError('Format non supporté (JPEG/PNG uniquement)')

    # Vérifier que c'est vraiment une image
    try:
        from PIL import Image
        img = Image.open(file)
        img.verify()
    except:
        raise ValidationError('Fichier corrompu ou invalide')
```

### 6️⃣ Logging et Monitoring

```python
# models.py
class APIUsageLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    endpoint = models.CharField(max_length=100)
    tokens_used = models.IntegerField()
    cost_usd = models.DecimalField(max_digits=10, decimal_places=6)
    timestamp = models.DateTimeField(auto_now_add=True)

# Dans la view
def log_usage(user, tokens_used):
    APIUsageLog.objects.create(
        user=user,
        endpoint='wound_analysis',
        tokens_used=tokens_used,
        cost_usd=tokens_used * 0.00001  # Estimation
    )
```

### 7️⃣ Budget Alert

```python
# settings.py
MAX_MONTHLY_COST = 50.00  # USD

# Dans une tâche cron quotidienne
def check_monthly_budget():
    from django.utils import timezone
    from django.core.mail import send_mail

    current_month = timezone.now().month
    total_cost = APIUsageLog.objects.filter(
        timestamp__month=current_month
    ).aggregate(Sum('cost_usd'))['cost_usd__sum'] or 0

    if total_cost >= MAX_MONTHLY_COST * 0.8:  # 80% du budget
        send_mail(
            'Alerte Budget API Vision',
            f'Budget à {total_cost/MAX_MONTHLY_COST*100:.0f}%',
            'noreply@example.com',
            ['admin@example.com'],
        )
```

## 📋 Configuration Recommandée

### Pour Production

**settings.py:**
```python
# Rate Limiting
INSTALLED_APPS += ['django_ratelimit']

# Cache (requis pour rate limiting)
CACHES = {
    'default': {
        'BACKEND': 'django_redis.cache.RedisCache',
        'LOCATION': 'redis://127.0.0.1:6379/1',
    }
}

# Limites API
API_WOUND_ANALYSIS_LIMITS = {
    'default': 10,      # 10/jour pour users normaux
    'staff': 100,       # 100/jour pour staff
    'superuser': 1000,  # 1000/jour pour admin
}

# Budget
MAX_MONTHLY_API_COST_USD = 50.00
API_COST_ALERT_THRESHOLD = 0.80  # Alerter à 80%

# Alertes
API_ALERT_EMAILS = ['admin@example.com', 'tech@example.com']
```

**View finale protégée:**
```python
from django.contrib.auth.decorators import login_required
from django_ratelimit.decorators import ratelimit
from django.core.cache import cache

@login_required
@ratelimit(key='user', rate='10/d', method='POST')  # 10 par jour
def analyze_wound_view(request):
    """View complètement protégée"""

    # Check rate limit
    if getattr(request, 'limited', False):
        return JsonResponse({
            'error': 'Quota quotidien dépassé (10/jour)'
        }, status=429)

    # Check permissions
    if not request.user.groups.filter(
        name__in=['Medecins', 'Infirmieres', 'Personnel_Soignant']
    ).exists():
        return JsonResponse({
            'error': 'Accès refusé'
        }, status=403)

    # Validate file
    if 'wound_image' not in request.FILES:
        return JsonResponse({'error': 'Image requise'}, status=400)

    file = request.FILES['wound_image']

    # Taille max 5MB
    if file.size > 5 * 1024 * 1024:
        return JsonResponse({
            'error': 'Image trop grande (max 5MB)'
        }, status=400)

    try:
        # Analyse
        ocr = DjangoOCRService()
        result = ocr.analyze_wound_from_uploaded_file(file)

        # Log usage
        tokens = result.get('_metadata', {}).get('tokens_used', 0)
        APIUsageLog.objects.create(
            user=request.user,
            endpoint='wound_analysis',
            tokens_used=tokens,
            cost_usd=tokens * 0.00001
        )

        return JsonResponse({
            'success': True,
            'data': result
        })

    except Exception as e:
        return JsonResponse({
            'error': 'Analyse échouée',
            'detail': str(e)
        }, status=500)
```

## 🚨 Checklist Sécurité

Avant de déployer en production:

- [ ] ✅ Authentification obligatoire (`@login_required`)
- [ ] ✅ Rate limiting activé (django-ratelimit ou cache)
- [ ] ✅ Permissions par rôle vérifiées
- [ ] ✅ Validation des fichiers (taille, type)
- [ ] ✅ Logging de l'usage activé
- [ ] ✅ Budget mensuel défini
- [ ] ✅ Alertes email configurées
- [ ] ✅ Redis/cache configuré
- [ ] ✅ Variables d'env protégées (OPENAI_API_KEY)
- [ ] ✅ HTTPS obligatoire en production

## 📊 Monitoring

### Dashboard Admin

Créez une vue admin pour monitorer:

```python
# admin.py
from django.contrib import admin
from django.db.models import Sum, Count
from django.utils.html import format_html

@admin.register(APIUsageLog)
class APIUsageLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'endpoint', 'tokens_used', 'cost_usd', 'timestamp']
    list_filter = ['endpoint', 'timestamp']
    date_hierarchy = 'timestamp'

    def changelist_view(self, request, extra_context=None):
        # Stats mensuelles
        from django.utils import timezone
        current_month = timezone.now().month

        stats = APIUsageLog.objects.filter(
            timestamp__month=current_month
        ).aggregate(
            total_calls=Count('id'),
            total_tokens=Sum('tokens_used'),
            total_cost=Sum('cost_usd')
        )

        extra_context = extra_context or {}
        extra_context['monthly_stats'] = stats

        return super().changelist_view(request, extra_context=extra_context)
```

### Commandes de Gestion

```python
# management/commands/check_api_usage.py
from django.core.management.base import BaseCommand
from myapp.models import APIUsageLog

class Command(BaseCommand):
    def handle(self, *args, **options):
        from django.utils import timezone

        # Usage du mois
        current_month = timezone.now().month
        monthly = APIUsageLog.objects.filter(
            timestamp__month=current_month
        ).aggregate(
            total=Sum('cost_usd')
        )['total'] or 0

        self.stdout.write(f"Coût ce mois: ${monthly:.2f}")
```

## 💡 Conseils Pratiques

1. **Commencez strict**: Limites basses au début, augmentez si nécessaire
2. **Différenciez les rôles**: Staff = plus de quota
3. **Alertes précoces**: Email à 50% et 80% du budget
4. **Review mensuel**: Analysez l'usage chaque mois
5. **Cache agressif**: Cachier les résultats identiques
6. **Compression images**: Réduire la taille avant envoi API

## 🔗 Ressources

- [django-ratelimit docs](https://django-ratelimit.readthedocs.io/)
- [OpenAI Rate Limits](https://platform.openai.com/docs/guides/rate-limits)
- [Django Cache Framework](https://docs.djangoproject.com/en/stable/topics/cache/)

---

**Version**: 1.0.0
**Dernière mise à jour**: 2025-01-07
