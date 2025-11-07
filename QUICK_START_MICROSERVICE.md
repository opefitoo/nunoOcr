# 🚀 Quick Start - Architecture Microservice

Guide rapide pour déployer nunoOcr comme service centralisé.

## 🎯 Objectif

Django appelle nunoOcr → nunoOcr appelle OpenAI

**Avantage**: Django ne connaît jamais la clé OpenAI!

## ⚡ Installation Rapide (3 Étapes)

### 1️⃣ Déployer le Service nunoOcr

#### Sur votre serveur Dokploy (46.224.6.193):

```bash
# Se connecter
ssh root@46.224.6.193

# Aller dans le projet nunoOcr
cd /etc/dokploy/compose/nunoocropefitoocom-nunoocr-ecwdho/code

# Récupérer la dernière version
git pull origin main

# Remplacer le serveur actuel
cp server_with_wound_analysis.py docker/server.py

# Redémarrer le service
cd /etc/dokploy/compose/nunoocropefitoocom-nunoocr-ecwdho
docker compose down
docker compose up -d
```

#### Configurer les Variables d'Environnement

Dans Dokploy → nunoOcr → Environment Variables:

```bash
OPENAI_API_KEY=sk-proj-rHu_SrM8g...  # Votre clé OpenAI
VISION_PROVIDER=openai
MODEL_NAME=deepseek-ai/DeepSeek-OCR
HOST=0.0.0.0
PORT=8000
```

**⚠️ IMPORTANT**: Ajouter des crédits OpenAI!
- Aller sur https://platform.openai.com/settings/organization/billing/overview
- Ajouter carte de crédit + $5-10

### 2️⃣ Tester le Service nunoOcr

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
```

```bash
# Test analyse de plaie
curl -X POST http://46.224.6.193:8765/v1/analyze-wound \
     -F "wound_image=@wound.jpg"

# Devrait retourner:
{
  "success": true,
  "data": {
    "type_plaie": "...",
    "localisation": "...",
    ...
  }
}
```

### 3️⃣ Configurer Django

#### Dans votre app Django `inur`:

**Copier le client**:
```bash
cp django_microservice_integration.py /path/to/inur.django/inur/nunoocr_client.py
```

**Configurer `settings.py`**:
```python
# Service nunoOcr URL
NUNOOCR_SERVICE_URL = os.getenv(
    'NUNOOCR_SERVICE_URL',
    'http://46.224.6.193:8765'  # Ou http://nunoocr:8000 si Docker Compose
)

# PAS de OPENAI_API_KEY dans Django!
```

**Créer la view** (`inur/api_views.py`):
```python
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.conf import settings
from .nunoocr_client import NunoOcrServiceClient

# Client global
NUNOOCR = NunoOcrServiceClient(base_url=settings.NUNOOCR_SERVICE_URL)

@csrf_exempt
@require_POST
def analyze_wound_api(request):
    """Analyser une plaie via le service nunoOcr."""
    if 'wound_image' not in request.FILES:
        return JsonResponse({'error': 'Image requise'}, status=400)

    try:
        result = NUNOOCR.analyze_wound(request.FILES['wound_image'])
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
```

**Ajouter l'URL** (`inur/urls.py`):
```python
from . import api_views

urlpatterns = [
    # ... vos URLs existantes ...
    path('api/analyze-wound/', api_views.analyze_wound_api),
]
```

## ✅ Test End-to-End

```bash
# Depuis votre machine locale
curl -X POST https://inur.opefitoo.com/api/analyze-wound/ \
     -F "wound_image=@wound.jpg"

# Devrait retourner l'analyse!
```

## 🔐 Ajouter l'Authentification API Key (Optionnel)

Si vous voulez protéger l'endpoint:

1. **Ajouter le modèle APIKey** (voir `API_KEY_SETUP.md`)
2. **Modifier la view**:

```python
from .decorators import require_api_key

@csrf_exempt
@require_POST
@require_api_key
def analyze_wound_api(request, api_key):
    """Protected endpoint."""
    if 'wound_image' not in request.FILES:
        return JsonResponse({'error': 'Image requise'}, status=400)

    try:
        result = NUNOOCR.analyze_wound(request.FILES['wound_image'])

        return JsonResponse({
            'success': True,
            'data': result['data'],
            'remaining_calls_today': api_key.daily_limit - api_key.calls_today
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
```

3. **Utiliser avec API Key**:
```bash
curl -X POST https://inur.opefitoo.com/api/analyze-wound/ \
     -H "Authorization: Bearer nuno_xxxxx" \
     -F "wound_image=@wound.jpg"
```

## 📊 Architecture Finale

```
Client
  ↓ Authorization: Bearer nuno_xxxxx
Django (inur.opefitoo.com)
  ↓ Vérifie API Key + quota
  ↓ Appelle nunoOcr
nunoOcr Service (46.224.6.193:8765)
  ↓ Utilise OPENAI_API_KEY
OpenAI API
  ↓ Analyse l'image
Retour au client
```

## 🔍 Troubleshooting

### Erreur: "Service nunoOcr unavailable"

**Vérifier que le service tourne**:
```bash
ssh root@46.224.6.193
docker ps | grep nunoocr
```

**Vérifier les logs**:
```bash
docker logs nunoocr_deepseek --tail 50
```

### Erreur: "OpenAI API error: 429"

**Vous n'avez pas de crédits OpenAI!**
- Aller sur https://platform.openai.com/settings/organization/billing/overview
- Ajouter carte + crédits

### Erreur: "vision_configured: false"

**La clé OpenAI n'est pas configurée dans nunoOcr**:
```bash
# Vérifier les variables d'environnement
ssh root@46.224.6.193
docker exec nunoocr_deepseek env | grep OPENAI

# Si vide, ajouter dans Dokploy → Environment Variables
```

## 📚 Documentation Complète

- `MICROSERVICE_ARCHITECTURE.md` - Architecture détaillée
- `API_KEYS_EXPLAINED.md` - Comprendre les deux types de clés
- `API_KEY_SETUP.md` - Setup API Key authentication
- `INTEGRATION_CHECKLIST.md` - Checklist complète

## 🎉 C'est Tout!

Vous avez maintenant:
- ✅ Service nunoOcr qui gère OpenAI
- ✅ Django qui appelle nunoOcr
- ✅ Séparation des clés API (sécurité++)
- ✅ Facile de changer de technologie

**Prochaine étape**: Ajouter l'authentification API Key pour protéger l'endpoint!

---

**Version**: 2.0.0
**Date**: 2025-01-07
