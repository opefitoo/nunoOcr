# 🔑 Configuration API Key Authentication - Guide Complet

Guide pour sécuriser votre API d'analyse de plaies avec des API Keys.

## 📋 Pourquoi API Key au lieu de Login/Password?

✅ **Avantages API Key:**
- Pas besoin de session/cookies
- Révocable sans changer le mot de passe
- Différentes clés pour différents services
- Quotas par clé
- Traçabilité précise
- Compatible avec tous les clients (mobile, web, CLI)

❌ **Problèmes Login/Password:**
- Nécessite session Django
- Difficile à utiliser depuis une app mobile
- Pas de granularité (un seul quota par user)
- Moins sécure (mot de passe exposé)

## 🚀 Installation Rapide

### 1. Ajouter le Modèle API Key

**models.py:**
```python
from django.db import models
from django.contrib.auth.models import User
import secrets
from django.utils import timezone

class APIKey(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    key = models.CharField(max_length=64, unique=True, db_index=True)
    name = models.CharField(max_length=100)

    daily_limit = models.IntegerField(default=10)
    calls_today = models.IntegerField(default=0)
    last_reset = models.DateField(auto_now_add=True)

    total_calls = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used = models.DateTimeField(null=True, blank=True)

    @staticmethod
    def generate_key():
        return f"nuno_{secrets.token_urlsafe(32)}"

    def can_make_request(self):
        from datetime import date
        # Reset quotidien
        if self.last_reset < date.today():
            self.calls_today = 0
            self.last_reset = date.today()
            self.save()

        if not self.is_active:
            return False, "Clé désactivée"

        if self.calls_today >= self.daily_limit:
            return False, "Quota dépassé"

        return True, None

    def record_usage(self):
        self.calls_today += 1
        self.total_calls += 1
        self.last_used = timezone.now()
        self.save()
```

### 2. Créer la Migration

```bash
python manage.py makemigrations
python manage.py migrate
```

### 3. Créer le Decorator

**decorators.py:**
```python
from functools import wraps
from django.http import JsonResponse
from .models import APIKey

def require_api_key(view_func):
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

        # Valider la clé
        try:
            api_key = APIKey.objects.get(key=api_key_string)
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

        # Injecter dans la request
        request.api_key = api_key
        request.user = api_key.user

        return view_func(request, *args, **kwargs)

    return wrapper
```

### 4. Créer la View API

**views.py:**
```python
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.http import JsonResponse
from .decorators import require_api_key
from nunoocr_client import DjangoOCRService

@csrf_exempt
@require_POST
@require_api_key
def analyze_wound_api(request):
    """
    POST /api/analyze-wound/
    Header: Authorization: Bearer nuno_xxxxx
    Body: wound_image (file)
    """
    if 'wound_image' not in request.FILES:
        return JsonResponse({'error': 'Image requise'}, status=400)

    file = request.FILES['wound_image']

    # Validation
    if file.size > 5 * 1024 * 1024:
        return JsonResponse({'error': 'Max 5MB'}, status=400)

    try:
        ocr = DjangoOCRService()
        result = ocr.analyze_wound_from_uploaded_file(file)

        return JsonResponse({
            'success': True,
            'data': result,
            'remaining_today': request.api_key.daily_limit - request.api_key.calls_today
        })

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
```

### 5. Ajouter l'URL

**urls.py:**
```python
from django.urls import path
from . import views

urlpatterns = [
    path('api/analyze-wound/', views.analyze_wound_api, name='analyze_wound_api'),
]
```

### 6. Admin Interface

**admin.py:**
```python
from django.contrib import admin
from .models import APIKey

@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'is_active', 'calls_today', 'daily_limit', 'created_at']
    readonly_fields = ['key', 'created_at', 'last_used', 'total_calls']

    def save_model(self, request, obj, form, change):
        if not change:  # Nouvelle clé
            obj.key = APIKey.generate_key()
        super().save_model(request, obj, form, change)
```

## 🎯 Utilisation

### Créer une API Key

**Option 1: Via Admin Django**
1. Aller dans Django Admin → API Keys
2. Cliquer "Add API Key"
3. Sélectionner l'utilisateur
4. Donner un nom (ex: "Production Server")
5. Définir le quota quotidien (ex: 10)
6. Sauvegarder
7. **Copier la clé générée** (commence par `nuno_`)

**Option 2: Via Management Command**
```bash
python manage.py shell

from django.contrib.auth.models import User
from myapp.models import APIKey

user = User.objects.get(username='mehdi')
api_key = APIKey.objects.create(
    user=user,
    key=APIKey.generate_key(),
    name="Production API",
    daily_limit=100
)

print(f"API Key: {api_key.key}")
```

**Option 3: Via Script**
```python
# create_api_key.py
from django.contrib.auth.models import User
from myapp.models import APIKey

def create_key(username, name="Default", limit=10):
    user = User.objects.get(username=username)
    key = APIKey.objects.create(
        user=user,
        key=APIKey.generate_key(),
        name=name,
        daily_limit=limit
    )
    return key.key

# Usage
key = create_key('mehdi', 'Mobile App', limit=50)
print(f"New API Key: {key}")
```

### Tester l'API

**Avec curl:**
```bash
curl -X POST https://inur.opefitoo.com/api/analyze-wound/ \
     -H "Authorization: Bearer nuno_abc123def456..." \
     -F "wound_image=@wound.jpg"
```

**Avec Python:**
```python
import requests

url = "https://inur.opefitoo.com/api/analyze-wound/"
headers = {
    "Authorization": "Bearer nuno_abc123def456..."
}
files = {
    "wound_image": open("wound.jpg", "rb")
}

response = requests.post(url, headers=headers, files=files)
print(response.json())
```

**Avec JavaScript:**
```javascript
const formData = new FormData();
formData.append('wound_image', fileInput.files[0]);

fetch('https://inur.opefitoo.com/api/analyze-wound/', {
    method: 'POST',
    headers: {
        'Authorization': 'Bearer nuno_abc123def456...'
    },
    body: formData
})
.then(res => res.json())
.then(data => console.log(data));
```

## 📊 Réponses API

### Succès (200)
```json
{
    "success": true,
    "data": {
        "type_plaie": "ulcère de pression",
        "localisation": "cheville gauche",
        "dimensions": {
            "longueur_cm": 2.5,
            "largeur_cm": 2.0
        },
        "stade_cicatrisation": "en cours de cicatrisation",
        "signes_infection": ["rougeur périphérique"],
        "etat_general": "Plaie en voie de cicatrisation",
        "confiance": "élevée",
        "notes": "Pansement hydrocolloïde visible"
    },
    "remaining_today": 9
}
```

### Erreur: Pas d'API Key (401)
```json
{
    "error": "Authorization header requis",
    "format": "Authorization: Bearer YOUR_API_KEY"
}
```

### Erreur: Quota Dépassé (429)
```json
{
    "error": "Quota dépassé",
    "daily_limit": 10,
    "calls_today": 10
}
```

### Erreur: Image Manquante (400)
```json
{
    "error": "Image requise"
}
```

## 🔐 Sécurité

### Bonnes Pratiques

1. **Ne jamais exposer la clé côté client**
```javascript
// ❌ MAUVAIS - clé visible dans le code frontend
const API_KEY = "nuno_abc123...";

// ✅ BON - clé sur le serveur backend
// Frontend → Backend → API Wound Analysis
```

2. **HTTPS Obligatoire**
```python
# settings.py
SECURE_SSL_REDIRECT = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
```

3. **Rotation des Clés**
```python
# Désactiver l'ancienne clé
old_key.is_active = False
old_key.save()

# Créer nouvelle clé
new_key = APIKey.objects.create(...)
```

4. **Monitoring**
```python
# Alerter si usage suspect
if api_key.calls_today > api_key.daily_limit * 0.9:
    send_alert_email(api_key.user, "Quota bientôt atteint")
```

## 📈 Quotas Recommandés

| Type d'Utilisateur | Quota/Jour | Usage Typique |
|-------------------|------------|---------------|
| Développement | 10-20 | Tests |
| Infirmière | 20-50 | Usage quotidien |
| Service/API | 100-500 | Intégration système |
| Admin | 1000 | Monitoring |

## 🔄 Gestion des Clés

### Lister les Clés
```python
from myapp.models import APIKey

# Toutes les clés actives
active_keys = APIKey.objects.filter(is_active=True)

# Clés d'un utilisateur
user_keys = APIKey.objects.filter(user__username='mehdi')

# Clés peu utilisées
unused = APIKey.objects.filter(total_calls__lt=10)
```

### Révoquer une Clé
```python
key = APIKey.objects.get(key='nuno_abc123...')
key.is_active = False
key.save()
```

### Augmenter le Quota
```python
key = APIKey.objects.get(key='nuno_abc123...')
key.daily_limit = 50
key.save()
```

## 🎁 Bonus: Endpoint Info

Ajoutez un endpoint pour voir le statut de la clé:

```python
@csrf_exempt
@require_api_key
def api_key_info(request):
    """GET /api/key-info/"""
    key = request.api_key
    return JsonResponse({
        'name': key.name,
        'user': key.user.username,
        'daily_limit': key.daily_limit,
        'calls_today': key.calls_today,
        'remaining_today': key.daily_limit - key.calls_today,
        'total_calls': key.total_calls,
        'created_at': key.created_at,
        'last_used': key.last_used
    })
```

Test:
```bash
curl https://inur.opefitoo.com/api/key-info/ \
     -H "Authorization: Bearer nuno_abc123..."
```

## 📝 Checklist Déploiement

- [ ] Modèle APIKey créé et migré
- [ ] Decorator `@require_api_key` implémenté
- [ ] View API créée et testée
- [ ] Admin interface configurée
- [ ] HTTPS activé (obligatoire!)
- [ ] API Keys créées pour utilisateurs
- [ ] Documentation fournie aux utilisateurs
- [ ] Monitoring des quotas en place
- [ ] Tests effectués (curl/Postman)

---

**Version**: 1.0.0
**Date**: 2025-01-07
**Sécurité**: Production-ready ✅
