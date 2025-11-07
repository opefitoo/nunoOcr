# 🏗️ Architecture Microservice - nunoOcr comme Service Centralisé

## 📋 Vue d'Ensemble

Au lieu que Django appelle directement OpenAI, on utilise nunoOcr comme **service centralisé** qui gère tous les appels aux APIs externes.

## 🎯 Avantages de Cette Architecture

### ✅ Séparation des Responsabilités
- **Django**: Gestion des utilisateurs, API Keys, quotas, business logic
- **nunoOcr**: Gestion des modèles AI (OCR, Vision)

### ✅ Sécurité
- Django ne connaît JAMAIS les clés OpenAI/Claude
- Les clés sont stockées uniquement dans le service nunoOcr
- Impossible de les exposer par erreur dans le code Django

### ✅ Flexibilité Technologique
- Changer d'OpenAI à Claude? Modifier seulement nunoOcr
- Ajouter un nouveau modèle? Seulement dans nunoOcr
- Django reste inchangé

### ✅ Cache Centralisé (Future)
- Possibilité d'ajouter un cache dans nunoOcr
- Éviter les appels redondants à OpenAI
- Économies de coûts

### ✅ Monitoring Centralisé
- Tous les appels AI passent par un seul point
- Facile de monitorer l'usage et les coûts
- Logs centralisés

## 🔄 Flow d'Appel Complet

```
┌─────────────────────────────────────────────────────────────────┐
│ CLIENT (Mobile App / Web Browser / Postman)                     │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ POST /api/analyze-wound/
                         │ Authorization: Bearer nuno_xxxxx
                         │ Body: wound_image=<file>
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ DJANGO APP (inur.opefitoo.com)                                  │
│                                                                  │
│  1. Decorator @require_api_key                                  │
│     - Vérifie que nuno_xxxxx existe en DB                       │
│     - Vérifie quota (ex: 10/jour)                               │
│     - Si OK, continue                                            │
│                                                                  │
│  2. View analyze_wound_protected()                              │
│     - Reçoit le fichier wound_image                             │
│     - Appelle le service nunoOcr                                │
│                                                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ POST http://nunoocr:8765/v1/analyze-wound
                         │ Content-Type: multipart/form-data
                         │ Body: wound_image=<file>
                         │ (PAS de clé API ici - c'est interne!)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ NUNOOCR SERVICE (nunoocr:8765)                                  │
│                                                                  │
│  1. Endpoint /v1/analyze-wound                                  │
│     - Reçoit l'image                                            │
│     - Convertit en base64                                        │
│     - Prépare le prompt en français                             │
│                                                                  │
│  2. Appelle OpenAI/Claude                                       │
│     - Utilise OPENAI_API_KEY (variable d'env interne)           │
│     - Envoie à GPT-4 Vision ou Claude                           │
│                                                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ POST https://api.openai.com/v1/chat/completions
                         │ Authorization: Bearer sk-proj-xxxxx
                         │ (Clé OpenAI stockée dans nunoOcr)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ OPENAI API                                                       │
│                                                                  │
│  - Analyse l'image avec GPT-4o Vision                           │
│  - Retourne JSON structuré en français                          │
│  - Facture sur compte OpenAI                                    │
│                                                                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Response JSON
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ NUNOOCR SERVICE                                                  │
│  - Reçoit la réponse OpenAI                                     │
│  - Parse le JSON                                                 │
│  - Retourne à Django                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Response: {"success": true, "data": {...}}
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ DJANGO APP                                                       │
│  - Reçoit le résultat                                           │
│  - Incrémente api_key.calls_today                               │
│  - Retourne au client                                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         │ Response JSON + remaining quota
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ CLIENT                                                           │
│  - Reçoit l'analyse structurée                                  │
│  - Affiche les résultats                                        │
└─────────────────────────────────────────────────────────────────┘
```

## 🔑 Gestion des Clés

### Deux Niveaux d'Authentification

#### Niveau 1: Client → Django (API Key `nuno_xxxxx`)
```python
# Dans Django models
class APIKey(models.Model):
    key = "nuno_abc123..."  # Généré par Django
    user = ForeignKey(User)
    daily_limit = 10
    calls_today = 5
```

**But**: Contrôler qui peut utiliser votre API et combien

#### Niveau 2: nunoOcr → OpenAI (clé OpenAI `sk-proj-xxx`)
```bash
# Dans nunoOcr service (variable d'environnement)
OPENAI_API_KEY=sk-proj-rHu_SrM8g...
```

**But**: Authentifier le service nunoOcr auprès d'OpenAI

### Qui Connaît Quoi?

| Composant | Connaît API Key Django | Connaît Clé OpenAI |
|-----------|------------------------|-------------------|
| **Client** | ✅ Oui (`nuno_xxx`) | ❌ Non |
| **Django** | ✅ Oui (vérifie en DB) | ❌ Non |
| **nunoOcr** | ❌ Non | ✅ Oui (var d'env) |
| **OpenAI** | ❌ Non | ✅ Oui (vérifie) |

## 📦 Déploiement

### Option 1: Docker Compose (Recommandé)

```yaml
# docker-compose.yml
version: '3.8'

services:
  # Service nunoOcr (gestion des modèles AI)
  nunoocr:
    build:
      context: ./nunoOcr
      dockerfile: Dockerfile
    container_name: nunoocr_service
    ports:
      - "8765:8000"
    environment:
      # Clés API - UNIQUEMENT dans nunoOcr!
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - VISION_PROVIDER=openai
      - MODEL_NAME=deepseek-ai/DeepSeek-OCR
      - HOST=0.0.0.0
      - PORT=8000
    volumes:
      - model-cache:/root/.cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Application Django
  django:
    build:
      context: ./inur.django
      dockerfile: Dockerfile
    container_name: django_app
    depends_on:
      - nunoocr
      - postgres
    environment:
      # URL du service nunoOcr (réseau Docker)
      - NUNOOCR_SERVICE_URL=http://nunoocr:8000
      # PAS de OPENAI_API_KEY ici!
      - DATABASE_URL=postgresql://...
      - SECRET_KEY=...
    ports:
      - "8000:8000"

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=inur
      - POSTGRES_USER=...
      - POSTGRES_PASSWORD=...
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  model-cache:
  postgres-data:
```

### Option 2: Services Séparés (Dokploy)

Si nunoOcr et Django sont sur des serveurs différents:

**nunoOcr** (sur serveur GPU):
```bash
# Variables d'environnement
OPENAI_API_KEY=sk-proj-xxxxx
VISION_PROVIDER=openai
PORT=8000
```

**Django** (sur serveur standard):
```bash
# Variables d'environnement
NUNOOCR_SERVICE_URL=https://nunoocr.opefitoo.com
# PAS de OPENAI_API_KEY!
```

## 🔧 Configuration

### 1. Service nunoOcr

**Remplacer** `docker/server.py` par `server_with_wound_analysis.py`:

```bash
cd /path/to/nunoOcr/
cp server_with_wound_analysis.py docker/server.py
```

**Variables d'environnement** (Dokploy ou Docker Compose):
```bash
OPENAI_API_KEY=sk-proj-xxxxx
VISION_PROVIDER=openai  # ou anthropic
MODEL_NAME=deepseek-ai/DeepSeek-OCR
HOST=0.0.0.0
PORT=8000
```

### 2. Application Django

**Copier** `django_microservice_integration.py` dans votre app Django:

```bash
cp django_microservice_integration.py /path/to/inur.django/inur/nunoocr_client.py
```

**settings.py**:
```python
# URL du service nunoOcr
NUNOOCR_SERVICE_URL = os.getenv(
    'NUNOOCR_SERVICE_URL',
    'http://localhost:8765'
)

# PAS de OPENAI_API_KEY ici!
```

**views.py**:
```python
from .nunoocr_client import NunoOcrServiceClient, require_api_key
from django.conf import settings

# Client global
NUNOOCR_SERVICE = NunoOcrServiceClient(
    base_url=settings.NUNOOCR_SERVICE_URL
)

@csrf_exempt
@require_POST
@require_api_key
def analyze_wound_api(request, api_key):
    """Analyser une plaie via le service nunoOcr."""
    wound_image = request.FILES['wound_image']

    # Appeler le service nunoOcr
    result = NUNOOCR_SERVICE.analyze_wound(wound_image)

    return JsonResponse({
        'success': True,
        'data': result['data'],
        'remaining_calls_today': api_key.daily_limit - api_key.calls_today
    })
```

## 🧪 Tests

### 1. Tester le Service nunoOcr Directement

```bash
# Health check
curl http://localhost:8765/health

# Analyser une plaie
curl -X POST http://localhost:8765/v1/analyze-wound \
     -F "wound_image=@wound.jpg"
```

### 2. Tester via Django

```bash
# Avec API Key
curl -X POST https://inur.opefitoo.com/api/analyze-wound/ \
     -H "Authorization: Bearer nuno_xxxxx" \
     -F "wound_image=@wound.jpg"
```

### 3. Test de Connection Python

```python
from django_microservice_integration import NunoOcrServiceClient

# Créer le client
client = NunoOcrServiceClient(base_url="http://localhost:8765")

# Vérifier le service
health = client.health_check()
print(f"OCR ready: {health['ocr_ready']}")
print(f"Vision configured: {health['vision_configured']}")

# Tester analyse
with open('wound.jpg', 'rb') as f:
    # Simuler UploadedFile Django
    from django.core.files.uploadedfile import SimpleUploadedFile
    uploaded = SimpleUploadedFile("wound.jpg", f.read(), content_type="image/jpeg")

    result = client.analyze_wound(uploaded)
    print(result)
```

## 🔄 Migration depuis l'Ancienne Architecture

Si vous utilisiez déjà `nunoocr_client.py` avec appels directs à OpenAI:

### Avant (Django appelle OpenAI directement):
```python
# ❌ Ancien code
from nunoocr_client import DjangoOCRService

ocr = DjangoOCRService(
    vision_api_key=settings.OPENAI_API_KEY,  # Clé dans Django!
    vision_provider='openai'
)
result = ocr.analyze_wound_from_uploaded_file(wound_image)
```

### Après (Django appelle nunoOcr):
```python
# ✅ Nouveau code
from .nunoocr_client import NunoOcrServiceClient

client = NunoOcrServiceClient(
    base_url=settings.NUNOOCR_SERVICE_URL  # Pas de clé API!
)
result = client.analyze_wound(wound_image)
```

**Changements requis**:
1. Déplacer `OPENAI_API_KEY` de Django → nunoOcr
2. Remplacer `server.py` dans nunoOcr
3. Mettre à jour les views Django
4. Tester la connexion

## 💰 Coûts

Identiques à l'ancienne architecture:
- **GPT-4o Vision**: ~$0.01-0.03 par image
- **Quota 10/jour**: ~$0.10-0.30/jour max
- **Budget mensuel**: ~$3-9/mois (10 images/jour)

**Avantage**: Plus facile de monitorer et cacher les coûts dans nunoOcr!

## 🚀 Avantages Futurs

### Cache Intelligent (TODO)
```python
# Dans nunoOcr - cacher les images identiques
if image_hash in cache:
    return cached_result
else:
    result = call_openai(image)
    cache[image_hash] = result
    return result
```

### Retry Logic (TODO)
```python
# Dans nunoOcr - retry automatique si OpenAI timeout
for attempt in range(3):
    try:
        return call_openai(image)
    except Timeout:
        if attempt < 2:
            time.sleep(2 ** attempt)  # Exponential backoff
        else:
            raise
```

### Rate Limiting Intelligent (TODO)
```python
# Dans nunoOcr - limiter les appels OpenAI par minute
if openai_calls_this_minute > 60:
    time.sleep(60)
```

## 📚 Fichiers Importants

| Fichier | Rôle |
|---------|------|
| `server_with_wound_analysis.py` | Serveur nunoOcr avec endpoints wound analysis |
| `django_microservice_integration.py` | Client Django pour appeler nunoOcr |
| `docker-compose.yml` | Configuration Docker (exemple) |
| `API_KEY_SETUP.md` | Guide d'installation API Keys Django |

## ✅ Checklist de Déploiement

- [ ] Service nunoOcr configuré avec `OPENAI_API_KEY`
- [ ] `server_with_wound_analysis.py` déployé dans nunoOcr
- [ ] Service nunoOcr accessible sur port 8765 (ou autre)
- [ ] Django configuré avec `NUNOOCR_SERVICE_URL`
- [ ] Django utilise `NunoOcrServiceClient`
- [ ] Modèle `APIKey` créé dans Django
- [ ] Decorator `@require_api_key` implémenté
- [ ] Tests de connexion réussis
- [ ] Crédits OpenAI ajoutés ($5-10 minimum)
- [ ] Health check fonctionne: `/health`
- [ ] Analyse de plaie testée end-to-end

---

**Architecture**: Microservice
**Version**: 2.0.0
**Date**: 2025-01-07
**Production Ready**: ✅
