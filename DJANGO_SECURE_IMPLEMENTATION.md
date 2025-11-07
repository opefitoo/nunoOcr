# 🔐 Implémentation Sécurisée Django - Guide Complet

## 🎯 Architecture de Sécurité

```
Client
  ↓ Authorization: Bearer nuno_xxxxx (API Key - Niveau 1)
Django (vérifie quota)
  ↓ Appelle nunoOcr (réseau interne - pas d'auth)
nunoOcr
  ↓ Utilise OPENAI_API_KEY (Niveau 2)
OpenAI
```

## 📋 Étapes d'Implémentation

### 1️⃣ Ajouter le Modèle APIKey dans Django

**Fichier**: `inur/models.py` (ou créez `inur/api_models.py`)

```python
from django.db import models
from django.contrib.auth.models import User
import secrets
from django.utils import timezone
from datetime import date

class APIKey(models.Model):
    """
    Modèle pour gérer les API Keys avec quotas.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='api_keys')
    key = models.CharField(max_length=64, unique=True, db_index=True)
    name = models.CharField(
        max_length=100,
        help_text="Nom descriptif (ex: 'Mobile App Production')"
    )

    # Quotas
    daily_limit = models.IntegerField(default=10, help_text="Nombre d'appels autorisés par jour")
    calls_today = models.IntegerField(default=0)
    last_reset = models.DateField(auto_now_add=True)

    # Statistiques
    total_calls = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used = models.DateTimeField(null=True, blank=True)

    # Statut
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = "API Key"
        verbose_name_plural = "API Keys"

    def __str__(self):
        return f"{self.name} ({self.user.username})"

    @staticmethod
    def generate_key():
        """Générer une clé API aléatoire unique."""
        return f"nuno_{secrets.token_urlsafe(40)}"

    def reset_daily_count_if_needed(self):
        """Réinitialiser le compteur quotidien si on a changé de jour."""
        if self.last_reset < date.today():
            self.calls_today = 0
            self.last_reset = date.today()
            self.save()

    def can_make_request(self):
        """
        Vérifier si la clé peut faire une requête.

        Returns:
            tuple: (bool, str) - (autorisation, message d'erreur si refusé)
        """
        if not self.is_active:
            return False, "Clé API désactivée"

        self.reset_daily_count_if_needed()

        if self.calls_today >= self.daily_limit:
            return False, f"Quota quotidien dépassé ({self.daily_limit}/jour)"

        return True, None

    def record_usage(self):
        """Enregistrer l'utilisation de la clé."""
        self.calls_today += 1
        self.total_calls += 1
        self.last_used = timezone.now()
        self.save(update_fields=['calls_today', 'total_calls', 'last_used'])
```

### 2️⃣ Créer la Migration

```bash
cd /path/to/inur.django
python manage.py makemigrations
python manage.py migrate
```

### 3️⃣ Créer le Decorator de Sécurité

**Fichier**: `inur/decorators.py`

```python
from functools import wraps
from django.http import JsonResponse
from .models import APIKey

def require_api_key(view_func):
    """
    Decorator pour requérir une API Key valide.

    Usage:
        @require_api_key
        def my_view(request, api_key):
            # api_key est l'objet APIKey validé
            ...
    """
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        # Récupérer le header Authorization
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')

        if not auth_header.startswith('Bearer '):
            return JsonResponse({
                'error': 'Authorization header requis',
                'format': 'Authorization: Bearer YOUR_API_KEY'
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
                'detail': error_message,
                'daily_limit': api_key.daily_limit,
                'calls_today': api_key.calls_today
            }, status=429 if 'Quota' in error_message else 403)

        # Enregistrer l'utilisation
        api_key.record_usage()

        # Ajouter la clé à la request
        request.api_key = api_key
        request.user = api_key.user

        # Appeler la view
        return view_func(request, api_key=api_key, *args, **kwargs)

    return wrapper
```

### 4️⃣ Créer la View Sécurisée

**Fichier**: `inur/api_views.py`

```python
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings
from .decorators import require_api_key
from .nunoocr_client import NunoOcrServiceClient, NunoOcrServiceError

# Client global pour le service nunoOcr
NUNOOCR_SERVICE = NunoOcrServiceClient(
    base_url=getattr(settings, 'NUNOOCR_SERVICE_URL', 'http://localhost:8765')
)


@csrf_exempt
@require_POST
@require_api_key
def analyze_wound_api(request, api_key):
    """
    Endpoint sécurisé pour analyser une plaie.

    Sécurité:
    - Requiert API Key valide (Authorization: Bearer nuno_xxxxx)
    - Vérifie quota quotidien (ex: 10/jour)
    - Enregistre chaque utilisation

    Architecture:
    - Django vérifie l'API Key
    - Django appelle nunoOcr (http://nunoocr:8765/v1/analyze-wound)
    - nunoOcr appelle OpenAI (avec OPENAI_API_KEY stockée là-bas)

    Usage:
        curl -X POST https://inur.opefitoo.com/api/analyze-wound/ \
             -H "Authorization: Bearer nuno_xxxxx" \
             -F "wound_image=@wound.jpg"

    Returns:
        {
            "success": true,
            "data": {
                "type_plaie": "...",
                "localisation": "...",
                ...
            },
            "api_key_name": "Mobile App",
            "remaining_calls_today": 9
        }
    """
    # Vérifier le fichier
    if 'wound_image' not in request.FILES:
        return JsonResponse({
            'error': 'Image requise',
            'detail': 'Le champ "wound_image" est obligatoire'
        }, status=400)

    wound_image = request.FILES['wound_image']

    # Validation taille (5MB max)
    if wound_image.size > 5 * 1024 * 1024:
        return JsonResponse({
            'error': 'Fichier trop volumineux',
            'detail': 'Taille maximum: 5MB',
            'size_received': f'{wound_image.size / 1024 / 1024:.2f}MB'
        }, status=400)

    # Validation type
    allowed_types = ['image/jpeg', 'image/jpg', 'image/png']
    if wound_image.content_type not in allowed_types:
        return JsonResponse({
            'error': 'Type de fichier non supporté',
            'detail': 'Formats acceptés: JPEG, PNG',
            'type_received': wound_image.content_type
        }, status=400)

    try:
        # Appeler le service nunoOcr
        result = NUNOOCR_SERVICE.analyze_wound(wound_image)

        # Vérifier le résultat
        if not result.get('success'):
            return JsonResponse({
                'error': 'Analyse échouée',
                'detail': result.get('error', 'Erreur inconnue')
            }, status=500)

        # Retourner avec info API Key
        return JsonResponse({
            'success': True,
            'data': result['data'],
            'api_key_name': api_key.name,
            'remaining_calls_today': api_key.daily_limit - api_key.calls_today,
            'user': api_key.user.username
        })

    except NunoOcrServiceError as e:
        return JsonResponse({
            'error': 'Service nunoOcr indisponible',
            'detail': str(e)
        }, status=503)
    except Exception as e:
        return JsonResponse({
            'error': 'Erreur inattendue',
            'detail': str(e)
        }, status=500)


@csrf_exempt
@require_POST
@require_api_key
def compare_wounds_api(request, api_key):
    """
    Endpoint sécurisé pour comparer plusieurs plaies dans le temps.

    Usage:
        curl -X POST https://inur.opefitoo.com/api/compare-wounds/ \
             -H "Authorization: Bearer nuno_xxxxx" \
             -F "wound_1=@day1.jpg" \
             -F "date_1=2025-01-01" \
             -F "wound_2=@day7.jpg" \
             -F "date_2=2025-01-07"
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
            'error': 'Au moins 2 images requises pour la comparaison'
        }, status=400)

    try:
        # Appeler le service
        result = NUNOOCR_SERVICE.compare_wound_progress(images_with_dates)

        return JsonResponse({
            'success': True,
            'data': result['data'],
            'api_key_name': api_key.name,
            'remaining_calls_today': api_key.daily_limit - api_key.calls_today
        })

    except NunoOcrServiceError as e:
        return JsonResponse({
            'error': 'Service error',
            'detail': str(e)
        }, status=503)


@csrf_exempt
@require_api_key
def api_key_info(request, api_key):
    """
    Endpoint pour voir les infos de l'API Key.

    Usage:
        curl https://inur.opefitoo.com/api/key-info/ \
             -H "Authorization: Bearer nuno_xxxxx"

    Returns:
        {
            "name": "Mobile App",
            "user": "mehdi",
            "daily_limit": 10,
            "calls_today": 5,
            "remaining_today": 5,
            "total_calls": 127,
            "created_at": "2025-01-07T10:00:00Z",
            "last_used": "2025-01-07T15:30:00Z"
        }
    """
    return JsonResponse({
        'name': api_key.name,
        'user': api_key.user.username,
        'daily_limit': api_key.daily_limit,
        'calls_today': api_key.calls_today,
        'remaining_today': api_key.daily_limit - api_key.calls_today,
        'total_calls': api_key.total_calls,
        'created_at': api_key.created_at,
        'last_used': api_key.last_used
    })
```

### 5️⃣ Ajouter les URLs

**Fichier**: `inur/urls.py`

```python
from django.urls import path
from . import api_views

urlpatterns = [
    # ... vos URLs existantes ...

    # API Wound Analysis (sécurisée avec API Key)
    path('api/analyze-wound/', api_views.analyze_wound_api, name='analyze_wound_api'),
    path('api/compare-wounds/', api_views.compare_wounds_api, name='compare_wounds_api'),
    path('api/key-info/', api_views.api_key_info, name='api_key_info'),
]
```

### 6️⃣ Configurer l'Admin

**Fichier**: `inur/admin.py`

```python
from django.contrib import admin
from .models import APIKey

@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = [
        'name', 'user', 'is_active',
        'calls_today', 'daily_limit',
        'total_calls', 'last_used', 'created_at'
    ]
    list_filter = ['is_active', 'created_at', 'last_used']
    search_fields = ['name', 'user__username', 'key']
    readonly_fields = ['key', 'created_at', 'last_used', 'total_calls', 'calls_today', 'last_reset']

    fieldsets = (
        ('Informations', {
            'fields': ('user', 'name', 'key')
        }),
        ('Quotas', {
            'fields': ('daily_limit', 'calls_today', 'last_reset', 'total_calls')
        }),
        ('Statut', {
            'fields': ('is_active', 'last_used', 'created_at')
        }),
    )

    def save_model(self, request, obj, form, change):
        if not change:  # Nouvelle clé
            obj.key = APIKey.generate_key()
        super().save_model(request, obj, form, change)
```

### 7️⃣ Configurer Settings

**Fichier**: `inur/settings.py`

```python
# URL du service nunoOcr
NUNOOCR_SERVICE_URL = os.getenv(
    'NUNOOCR_SERVICE_URL',
    'http://localhost:8765'  # ou http://46.224.6.193:8765
)

# PAS de OPENAI_API_KEY ici! Elle est dans nunoOcr
```

## 🔑 Créer des API Keys

### Option 1: Via Django Admin (Recommandé)

1. Aller dans Django Admin → API Keys
2. Cliquer "Ajouter API Key"
3. Sélectionner l'utilisateur
4. Nom: "Production Mobile App"
5. Daily limit: 50
6. Sauvegarder
7. **Copier la clé générée** (commence par `nuno_`)

### Option 2: Via Management Command

Créer `inur/management/commands/create_api_key.py`:

```python
from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from inur.models import APIKey

class Command(BaseCommand):
    help = 'Créer une nouvelle API Key pour un utilisateur'

    def add_arguments(self, parser):
        parser.add_argument('username', type=str)
        parser.add_argument('--name', type=str, default='Default Key')
        parser.add_argument('--limit', type=int, default=10)

    def handle(self, *args, **options):
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

        self.stdout.write(self.style.SUCCESS('API Key created!'))
        self.stdout.write(f'Key: {api_key.key}')
        self.stdout.write(f'User: {user.username}')
        self.stdout.write(f'Daily limit: {api_key.daily_limit}')
```

Utilisation:
```bash
python manage.py create_api_key mehdi --name "Mobile App" --limit 50
```

### Option 3: Via Django Shell

```bash
python manage.py shell
```

```python
from django.contrib.auth.models import User
from inur.models import APIKey

# Créer une clé pour l'utilisateur "mehdi"
user = User.objects.get(username='mehdi')
api_key = APIKey.objects.create(
    user=user,
    key=APIKey.generate_key(),
    name="Production API",
    daily_limit=100
)

print(f"API Key: {api_key.key}")
# Output: API Key: nuno_abc123def456...
```

## 🧪 Tester l'API Sécurisée

### Test 1: Sans API Key (doit échouer)

```bash
curl -X POST https://inur.opefitoo.com/api/analyze-wound/ \
     -F "wound_image=@wound.jpg"

# Réponse: 401 Unauthorized
{
  "error": "Authorization header requis",
  "format": "Authorization: Bearer YOUR_API_KEY"
}
```

### Test 2: Avec API Key valide (doit réussir)

```bash
curl -X POST https://inur.opefitoo.com/api/analyze-wound/ \
     -H "Authorization: Bearer nuno_abc123def456..." \
     -F "wound_image=@wound.jpg"

# Réponse: 200 OK
{
  "success": true,
  "data": {
    "type_plaie": "ulcère de pression",
    "localisation": "cheville gauche",
    ...
  },
  "api_key_name": "Production API",
  "remaining_calls_today": 9,
  "user": "mehdi"
}
```

### Test 3: Quota dépassé (doit échouer)

Après 10 appels:

```bash
curl -X POST https://inur.opefitoo.com/api/analyze-wound/ \
     -H "Authorization: Bearer nuno_abc123def456..." \
     -F "wound_image=@wound.jpg"

# Réponse: 429 Too Many Requests
{
  "error": "Accès refusé",
  "detail": "Quota quotidien dépassé (10/jour)",
  "daily_limit": 10,
  "calls_today": 10
}
```

### Test 4: Info API Key

```bash
curl https://inur.opefitoo.com/api/key-info/ \
     -H "Authorization: Bearer nuno_abc123def456..."

# Réponse:
{
  "name": "Production API",
  "user": "mehdi",
  "daily_limit": 10,
  "calls_today": 5,
  "remaining_today": 5,
  "total_calls": 127,
  "created_at": "2025-01-07T10:00:00Z",
  "last_used": "2025-01-07T15:30:00Z"
}
```

## ✅ Checklist Complète

- [ ] Modèle `APIKey` ajouté dans `models.py`
- [ ] Migration créée et exécutée: `python manage.py migrate`
- [ ] Decorator `require_api_key` créé dans `decorators.py`
- [ ] Views sécurisées créées dans `api_views.py`
- [ ] Client nunoOcr copié: `django_microservice_integration.py → nunoocr_client.py`
- [ ] URLs ajoutées dans `urls.py`
- [ ] Admin interface configurée dans `admin.py`
- [ ] `NUNOOCR_SERVICE_URL` configurée dans `settings.py`
- [ ] API Key créée pour test
- [ ] Test sans API Key (doit échouer)
- [ ] Test avec API Key (doit réussir)
- [ ] Test quota dépassé (doit échouer)

## 🎉 Résultat Final

Vous avez maintenant:
- ✅ API sécurisée avec API Keys
- ✅ Quotas quotidiens (10/jour par défaut)
- ✅ Django appelle nunoOcr
- ✅ nunoOcr appelle OpenAI
- ✅ Django ne connaît JAMAIS la clé OpenAI
- ✅ Facile de changer de technologie

**Architecture complète sécurisée!** 🔐

---

**Version**: 2.0.0
**Date**: 2025-01-07
