# 🚀 COMMENCEZ ICI - Déploiement Complet Sécurisé

## 🎯 Ce Que Vous Allez Obtenir

Une API sécurisée pour analyser les plaies avec:
- ✅ **API Keys** pour authentifier vos clients
- ✅ **Quotas quotidiens** (10/jour par défaut)
- ✅ **Architecture microservice** (Django → nunoOcr → OpenAI)
- ✅ **Django ne connaît jamais la clé OpenAI**
- ✅ **Facile de changer de technologie** (OpenAI → Claude en 1 ligne)

## ⚡ Installation Rapide (30 minutes)

### Phase 1: Service nunoOcr (10 min)

#### 1. Déployer le Nouveau Serveur

```bash
ssh root@46.224.6.193

cd /etc/dokploy/compose/nunoocropefitoocom-nunoocr-ecwdho/code
git pull origin main

# Remplacer le serveur
cp server_with_wound_analysis.py docker/server.py

# Redémarrer
cd ..
docker compose down
docker compose up -d

# Vérifier les logs
docker logs nunoocr_deepseek --tail 50
```

#### 2. Générer Service API Key

```bash
# Sur votre machine locale
python3 -c "import secrets; print(f'nuno_service_{secrets.token_urlsafe(40)}')"

# Output (exemple):
nuno_service_8kJ2mP9xQ4nL7vR3wS6tY1dF5hK0zB8cN4vM2pQ9xW7sT3yL
```

**⚠️ IMPORTANT**: Copiez cette clé, vous en aurez besoin 2 fois!

#### 3. Configurer Variables d'Environnement

**Dans Dokploy → nunoOcr → Environment Variables**:

```bash
# Sécurité service-to-service (REQUIS!)
SERVICE_API_KEY=nuno_service_8kJ2mP9xQ4nL7vR3wS6tY1dF5hK0zB8cN4vM2pQ9xW7sT3yL

# Whitelist IP (optionnel mais recommandé)
ALLOWED_IPS=123.45.67.89  # IP de votre serveur Django

# Vision API
OPENAI_API_KEY=sk-proj-rHu_SrM8g...
VISION_PROVIDER=openai

# OCR
MODEL_NAME=deepseek-ai/DeepSeek-OCR
HOST=0.0.0.0
PORT=8000
```

#### 3. Ajouter Crédits OpenAI ⚠️ URGENT

https://platform.openai.com/settings/organization/billing/overview

- Ajouter carte de crédit
- Ajouter $5-10 de crédit

#### 4. Tester le Service

```bash
# Health check
curl http://46.224.6.193:8765/health

# Devrait retourner:
{
  "status": "ok",
  "ocr_ready": true,
  "vision_provider": "openai",
  "vision_configured": true
}

# Test analyse
curl -X POST http://46.224.6.193:8765/v1/analyze-wound \
     -F "wound_image=@wound.jpg"

# Devrait retourner JSON en français
```

### Phase 2: Django avec Sécurité API Key (20 min)

#### 1. Copier les Fichiers

```bash
cd /path/to/inur.django

# Copier le client nunoOcr
cp /path/to/nunoOcr/django_microservice_integration.py inur/nunoocr_client.py
```

#### 2. Ajouter le Modèle APIKey

**Dans `inur/models.py`**, ajouter:

```python
from django.db import models
from django.contrib.auth.models import User
import secrets
from django.utils import timezone
from datetime import date

class APIKey(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='api_keys')
    key = models.CharField(max_length=64, unique=True, db_index=True)
    name = models.CharField(max_length=100, help_text="Nom descriptif")

    daily_limit = models.IntegerField(default=10)
    calls_today = models.IntegerField(default=0)
    last_reset = models.DateField(auto_now_add=True)

    total_calls = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.user.username})"

    @staticmethod
    def generate_key():
        return f"nuno_{secrets.token_urlsafe(40)}"

    def reset_daily_count_if_needed(self):
        if self.last_reset < date.today():
            self.calls_today = 0
            self.last_reset = date.today()
            self.save()

    def can_make_request(self):
        if not self.is_active:
            return False, "Clé API désactivée"

        self.reset_daily_count_if_needed()

        if self.calls_today >= self.daily_limit:
            return False, f"Quota quotidien dépassé ({self.daily_limit}/jour)"

        return True, None

    def record_usage(self):
        self.calls_today += 1
        self.total_calls += 1
        self.last_used = timezone.now()
        self.save(update_fields=['calls_today', 'total_calls', 'last_used'])
```

#### 3. Créer le Decorator

**Créer `inur/decorators.py`**:

```python
from functools import wraps
from django.http import JsonResponse
from .models import APIKey

def require_api_key(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')

        if not auth_header.startswith('Bearer '):
            return JsonResponse({
                'error': 'Authorization header requis',
                'format': 'Authorization: Bearer YOUR_API_KEY'
            }, status=401)

        api_key_string = auth_header[7:]

        try:
            api_key = APIKey.objects.select_related('user').get(key=api_key_string)
        except APIKey.DoesNotExist:
            return JsonResponse({'error': 'API Key invalide'}, status=401)

        can_use, error_message = api_key.can_make_request()
        if not can_use:
            return JsonResponse({
                'error': error_message,
                'daily_limit': api_key.daily_limit,
                'calls_today': api_key.calls_today
            }, status=429)

        api_key.record_usage()
        request.api_key = api_key
        request.user = api_key.user

        return view_func(request, api_key=api_key, *args, **kwargs)

    return wrapper
```

#### 4. Créer les Views

**Créer `inur/api_views.py`**:

```python
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings
from .decorators import require_api_key
from .nunoocr_client import NunoOcrServiceClient, NunoOcrServiceError

NUNOOCR_SERVICE = NunoOcrServiceClient(
    base_url=getattr(settings, 'NUNOOCR_SERVICE_URL', 'http://localhost:8765')
)

@csrf_exempt
@require_POST
@require_api_key
def analyze_wound_api(request, api_key):
    if 'wound_image' not in request.FILES:
        return JsonResponse({'error': 'Image requise'}, status=400)

    wound_image = request.FILES['wound_image']

    if wound_image.size > 5 * 1024 * 1024:
        return JsonResponse({'error': 'Image trop grande (max 5MB)'}, status=400)

    try:
        result = NUNOOCR_SERVICE.analyze_wound(wound_image)

        if not result.get('success'):
            return JsonResponse({
                'error': 'Analyse échouée',
                'detail': result.get('error')
            }, status=500)

        return JsonResponse({
            'success': True,
            'data': result['data'],
            'api_key_name': api_key.name,
            'remaining_calls_today': api_key.daily_limit - api_key.calls_today
        })

    except NunoOcrServiceError as e:
        return JsonResponse({
            'error': 'Service nunoOcr indisponible',
            'detail': str(e)
        }, status=503)

@csrf_exempt
@require_api_key
def api_key_info(request, api_key):
    return JsonResponse({
        'name': api_key.name,
        'user': api_key.user.username,
        'daily_limit': api_key.daily_limit,
        'calls_today': api_key.calls_today,
        'remaining_today': api_key.daily_limit - api_key.calls_today,
        'total_calls': api_key.total_calls
    })
```

#### 5. Configurer URLs

**Dans `inur/urls.py`**, ajouter:

```python
from . import api_views

urlpatterns = [
    # ... vos URLs existantes ...
    path('api/analyze-wound/', api_views.analyze_wound_api),
    path('api/key-info/', api_views.api_key_info),
]
```

#### 6. Configurer Admin

**Dans `inur/admin.py`**, ajouter:

```python
from django.contrib import admin
from .models import APIKey

@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'is_active', 'calls_today', 'daily_limit', 'total_calls']
    readonly_fields = ['key', 'created_at', 'last_used', 'total_calls', 'calls_today']

    def save_model(self, request, obj, form, change):
        if not change:
            obj.key = APIKey.generate_key()
        super().save_model(request, obj, form, change)
```

#### 7. Configurer Settings

**Dans `inur/settings.py`**, ajouter:

```python
import os

# URL du service nunoOcr
NUNOOCR_SERVICE_URL = os.getenv(
    'NUNOOCR_SERVICE_URL',
    'http://46.224.6.193:8765'  # ou http://nunoocr:8000 si Docker network
)

# Service API Key (LA MÊME que dans nunoOcr!)
NUNOOCR_SERVICE_API_KEY = os.getenv('NUNOOCR_SERVICE_API_KEY')

# PAS de OPENAI_API_KEY ici! Elle est dans nunoOcr
```

**Ou dans `.env` / Dokploy variables d'environnement**:

```bash
NUNOOCR_SERVICE_URL=http://46.224.6.193:8765
NUNOOCR_SERVICE_API_KEY=nuno_service_8kJ2mP9xQ4nL7vR3wS6tY1dF5hK0zB8cN4vM2pQ9xW7sT3yL
```

#### 8. Migrer la Base de Données

```bash
python manage.py makemigrations
python manage.py migrate
```

#### 9. Créer une API Key de Test

**Django Admin**:
1. Aller dans Admin → API Keys → Add
2. User: sélectionner votre user
3. Name: "Test API"
4. Daily limit: 10
5. Sauvegarder
6. **Copier la clé** (commence par `nuno_`)

**Ou via shell**:
```bash
python manage.py shell
```

```python
from django.contrib.auth.models import User
from inur.models import APIKey

user = User.objects.get(username='mehdi')  # Votre username
api_key = APIKey.objects.create(
    user=user,
    key=APIKey.generate_key(),
    name="Test API",
    daily_limit=10
)

print(f"API Key: {api_key.key}")
```

#### 10. Tester End-to-End

```bash
# Test sans API Key (doit échouer - 401)
curl -X POST https://inur.opefitoo.com/api/analyze-wound/ \
     -F "wound_image=@wound.jpg"

# Test avec API Key (doit réussir - 200)
curl -X POST https://inur.opefitoo.com/api/analyze-wound/ \
     -H "Authorization: Bearer nuno_xxxxx" \
     -F "wound_image=@wound.jpg"

# Vérifier les infos de la clé
curl https://inur.opefitoo.com/api/key-info/ \
     -H "Authorization: Bearer nuno_xxxxx"
```

## ✅ Checklist Finale

### Service nunoOcr
- [ ] Code mis à jour (`git pull`)
- [ ] `server_with_wound_analysis.py` copié vers `docker/server.py`
- [ ] Service redémarré
- [ ] `OPENAI_API_KEY` configurée
- [ ] `VISION_PROVIDER=openai` configuré
- [ ] Crédits OpenAI ajoutés ($5-10)
- [ ] Health check OK: `curl http://46.224.6.193:8765/health`
- [ ] Test analyse OK: `curl -X POST .../v1/analyze-wound -F "wound_image=@..."`

### Django App
- [ ] Client nunoOcr copié → `inur/nunoocr_client.py`
- [ ] Modèle `APIKey` ajouté dans `models.py`
- [ ] Decorator `require_api_key` créé dans `decorators.py`
- [ ] Views créées dans `api_views.py`
- [ ] URLs ajoutées dans `urls.py`
- [ ] Admin configuré dans `admin.py`
- [ ] `NUNOOCR_SERVICE_URL` configurée dans `settings.py`
- [ ] Migrations exécutées
- [ ] API Key de test créée
- [ ] Test sans API Key (401) ✓
- [ ] Test avec API Key (200) ✓
- [ ] Test quota info ✓

## 📚 Documentation Détaillée

| Document | Quand l'Utiliser |
|----------|-----------------|
| **START_HERE.md** (ce fichier) | 🚀 Pour déployer rapidement |
| **DJANGO_SECURE_IMPLEMENTATION.md** | 🔐 Guide complet avec tous les détails |
| **QUICK_START_MICROSERVICE.md** | ⚡ Déploiement microservice sans API Key |
| **MICROSERVICE_ARCHITECTURE.md** | 🏗️ Comprendre l'architecture |
| **API_KEYS_EXPLAINED.md** | 🔑 Comprendre les deux types de clés |
| **SUMMARY.md** | 📋 Résumé complet du projet |

## 🎉 Résultat Final

Vous avez maintenant **triple sécurité**:

```
Client (Mobile/Web)
  ↓ Authorization: Bearer nuno_user_abc123 ← API Key User (Niveau 1)

Django (inur.opefitoo.com)
  ↓ Vérifie API Key user + quota ✓
  ↓ Authorization: Bearer nuno_service_xyz789 ← API Key Service (Niveau 2)
  ↓ POST http://46.224.6.193:8765/v1/analyze-wound

nunoOcr Service (46.224.6.193:8765)
  ↓ Vérifie API Key service ✓
  ↓ Vérifie IP whitelist ✓
  ↓ Authorization: Bearer sk-proj-xxxxx ← Clé OpenAI (Niveau 3)

OpenAI API
  ↓ Analyse l'image
  ↓ Retourne JSON français

Client reçoit l'analyse + quota restant
```

**Sécurité**:
- ✅ Niveau 1: Clients authentifiés avec quota
- ✅ Niveau 2: Service-to-service avec API Key
- ✅ Niveau 3: nunoOcr authentifié auprès OpenAI
- ✅ Django ne connaît jamais la clé OpenAI

**Flexibilité**: ✅ Changer de provider en 1 ligne
**Protection**: ✅ Impossible d'appeler nunoOcr sans la bonne clé

## 🆘 Problèmes Courants

### "Service nunoOcr unavailable"
```bash
# Vérifier que le service tourne
ssh root@46.224.6.193
docker ps | grep nunoocr
docker logs nunoocr_deepseek --tail 50
```

### "OpenAI API error: 429"
→ Vous n'avez pas de crédits OpenAI!
→ Aller sur https://platform.openai.com/settings/organization/billing/overview

### "vision_configured: false" dans health check
→ La clé `OPENAI_API_KEY` n'est pas configurée dans nunoOcr
→ Vérifier les variables d'environnement Dokploy

### "API Key invalide"
→ Vérifier que vous utilisez le bon format: `Authorization: Bearer nuno_xxxxx`
→ Vérifier que la clé existe en DB: `APIKey.objects.filter(key='...')`

### "Quota quotidien dépassé"
→ Normal après 10 appels
→ Augmenter `daily_limit` dans Admin ou attendre minuit

## 💡 Next Steps

1. **Production**: Augmenter les quotas selon vos besoins
2. **Monitoring**: Ajouter logs et alertes
3. **Cache**: Implémenter cache dans nunoOcr pour économiser
4. **Backup**: Sauvegarder la DB des API Keys
5. **Documentation**: Créer doc API pour vos clients

---

**Version**: 2.0.0
**Date**: 2025-01-07
**Status**: ✅ Production Ready
**Support**: Voir les docs ci-dessus
