# 🔐 Sécurité Service-to-Service

## 🎯 Problème

Vous avez raison! nunoOcr est sur un serveur différent de Django:
- **Django (inur)**: Serveur A
- **nunoOcr**: Serveur B (`46.224.6.193`)

C'est un **appel public sur Internet** qu'il faut sécuriser!

## 🏗️ Architecture de Sécurité Complète

```
Client (Mobile/Web)
  ↓ Authorization: Bearer nuno_user_abc123 (API Key User - Niveau 1)

Django (Serveur A)
  ↓ Vérifie API Key user + quota ✓
  ↓ Authorization: Bearer nuno_service_xyz789 (API Key Service - Niveau 2) ← NOUVEAU!

nunoOcr (Serveur B - 46.224.6.193)
  ↓ Vérifie API Key service ✓
  ↓ Vérifie IP whitelist (optionnel) ✓
  ↓ Utilise OPENAI_API_KEY (Niveau 3)

OpenAI API
  ↓ Retourne résultat
```

## 🔑 Les TROIS Clés

| Clé | Où | But | Format |
|-----|----|----|--------|
| **User API Key** | Django DB | Authentifier clients → Django | `nuno_user_abc123...` |
| **Service API Key** | Config Django + nunoOcr | Authentifier Django → nunoOcr | `nuno_service_xyz789...` |
| **OpenAI API Key** | nunoOcr config | Authentifier nunoOcr → OpenAI | `sk-proj-xxxxx` |

## ⚙️ Configuration

### 1️⃣ Générer la Service API Key

```bash
# Sur votre machine locale
python3 -c "import secrets; print(f'nuno_service_{secrets.token_urlsafe(40)}')"

# Output (exemple):
nuno_service_8kJ2mP9xQ4nL7vR3wS6tY1dF5hK0zB8cN4vM2pQ9xW7sT3yL
```

**Copiez cette clé** - vous en aurez besoin 2 fois!

### 2️⃣ Configurer nunoOcr (Serveur B)

**Dans Dokploy → nunoOcr → Environment Variables**:

```bash
# Clé de service pour protéger les endpoints
SERVICE_API_KEY=nuno_service_8kJ2mP9xQ4nL7vR3wS6tY1dF5hK0zB8cN4vM2pQ9xW7sT3yL

# Whitelist IP (optionnel mais recommandé)
ALLOWED_IPS=123.45.67.89,98.76.54.32  # IP de votre serveur Django

# Configuration existante
OPENAI_API_KEY=sk-proj-xxxxx
VISION_PROVIDER=openai
MODEL_NAME=deepseek-ai/DeepSeek-OCR
HOST=0.0.0.0
PORT=8000
```

**Redémarrer le service**:
```bash
ssh root@46.224.6.193
cd /etc/dokploy/compose/nunoocropefitoocom-nunoocr-ecwdho
docker compose down
docker compose up -d
```

### 3️⃣ Configurer Django (Serveur A)

**Dans `settings.py` ou variables d'environnement**:

```python
# settings.py
import os

# URL du service nunoOcr
NUNOOCR_SERVICE_URL = os.getenv(
    'NUNOOCR_SERVICE_URL',
    'http://46.224.6.193:8765'
)

# Service API Key (LA MÊME que dans nunoOcr!)
NUNOOCR_SERVICE_API_KEY = os.getenv('NUNOOCR_SERVICE_API_KEY')
```

**Ou dans `.env` / Dokploy variables**:
```bash
NUNOOCR_SERVICE_URL=http://46.224.6.193:8765
NUNOOCR_SERVICE_API_KEY=nuno_service_8kJ2mP9xQ4nL7vR3wS6tY1dF5hK0zB8cN4vM2pQ9xW7sT3yL
```

## 🧪 Tests

### Test 1: Health Check (public - pas d'auth)

```bash
curl http://46.224.6.193:8765/health

# Réponse:
{
  "status": "ok",
  "ocr_ready": true,
  "vision_provider": "openai",
  "vision_configured": true,
  "security": {
    "service_api_key_required": true,
    "ip_whitelist_enabled": true,
    "allowed_ips_count": 2
  }
}
```

### Test 2: Sans Service API Key (doit échouer)

```bash
curl -X POST http://46.224.6.193:8765/v1/analyze-wound \
     -F "wound_image=@wound.jpg"

# Réponse: 401 Unauthorized
{
  "detail": {
    "error": "Authorization required",
    "message": "Service API Key required. Set 'Authorization: Bearer YOUR_SERVICE_KEY'"
  }
}
```

### Test 3: Avec Mauvaise Service API Key (doit échouer)

```bash
curl -X POST http://46.224.6.193:8765/v1/analyze-wound \
     -H "Authorization: Bearer nuno_service_WRONG" \
     -F "wound_image=@wound.jpg"

# Réponse: 401 Unauthorized
{
  "detail": {
    "error": "Invalid service API key",
    "message": "The provided service API key is incorrect"
  }
}
```

### Test 4: Avec Bonne Service API Key (doit réussir)

```bash
curl -X POST http://46.224.6.193:8765/v1/analyze-wound \
     -H "Authorization: Bearer nuno_service_8kJ2mP9xQ4nL7vR3wS6tY1dF5hK0zB8cN4vM2pQ9xW7sT3yL" \
     -F "wound_image=@wound.jpg"

# Réponse: 200 OK
{
  "success": true,
  "data": {
    "type_plaie": "...",
    ...
  }
}
```

### Test 5: Depuis Django (doit réussir)

Le client Django envoie automatiquement la clé:

```python
from .nunoocr_client import NunoOcrServiceClient

client = NunoOcrServiceClient(
    base_url='http://46.224.6.193:8765',
    service_api_key='nuno_service_8kJ2mP9xQ4nL7vR3wS6tY1dF5hK0zB8cN4vM2pQ9xW7sT3yL'
)

result = client.analyze_wound(wound_image)
# ✓ Fonctionne!
```

### Test 6: Depuis IP Non-Whitelistée (doit échouer si whitelist activée)

Si vous avez configuré `ALLOWED_IPS`:

```bash
# Depuis une autre machine
curl -X POST http://46.224.6.193:8765/v1/analyze-wound \
     -H "Authorization: Bearer nuno_service_8kJ2mP9xQ4nL7vR3wS6tY1dF5hK0zB8cN4vM2pQ9xW7sT3yL" \
     -F "wound_image=@wound.jpg"

# Réponse: 403 Forbidden
{
  "detail": {
    "error": "IP not allowed",
    "message": "Your IP (1.2.3.4) is not authorized to access this service"
  }
}
```

## 🛡️ Niveaux de Sécurité

Vous pouvez choisir votre niveau:

### Niveau 1: Aucune Protection (NON RECOMMANDÉ!)
```bash
# nunoOcr - NE PAS configurer SERVICE_API_KEY
# Résultat: N'importe qui peut appeler nunoOcr
```

### Niveau 2: Service API Key Uniquement (RECOMMANDÉ MINIMUM)
```bash
# nunoOcr
SERVICE_API_KEY=nuno_service_xxxxx

# Résultat: Seuls ceux avec la clé peuvent appeler
```

### Niveau 3: Service API Key + IP Whitelist (RECOMMANDÉ)
```bash
# nunoOcr
SERVICE_API_KEY=nuno_service_xxxxx
ALLOWED_IPS=123.45.67.89  # IP de Django

# Résultat: Seul Django avec la bonne clé peut appeler
```

### Niveau 4: Service API Key + IP + Rate Limiting (MAXIMUM)
```bash
# nunoOcr
SERVICE_API_KEY=nuno_service_xxxxx
ALLOWED_IPS=123.45.67.89
# + Ajouter nginx rate limiting

# Résultat: Protection maximale
```

## 📊 Flow Complet Sécurisé

```
┌─────────────────────────────────────────────────────────────┐
│ CLIENT (Mobile App)                                          │
│ Authorization: Bearer nuno_user_abc123                       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ DJANGO (Serveur A - inur.opefitoo.com)                      │
│                                                              │
│ 1. Decorator @require_api_key                               │
│    - Vérifie nuno_user_abc123 en DB                         │
│    - Vérifie quota (10/jour)                                │
│    - Si OK, continue                                         │
│                                                              │
│ 2. NunoOcrServiceClient()                                   │
│    - Ajoute header:                                          │
│      Authorization: Bearer nuno_service_xyz789              │
│                                                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          │ POST http://46.224.6.193:8765/v1/analyze-wound
                          │ Authorization: Bearer nuno_service_xyz789
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ NUNOOCR (Serveur B - 46.224.6.193)                          │
│                                                              │
│ 1. verify_service_api_key()                                 │
│    - Vérifie nuno_service_xyz789                            │
│    - Si incorrect → 401 Unauthorized                         │
│                                                              │
│ 2. verify_ip_whitelist()                                    │
│    - Vérifie IP de Django                                   │
│    - Si non-whitelistée → 403 Forbidden                     │
│                                                              │
│ 3. Si OK, appelle OpenAI                                    │
│    - Utilise OPENAI_API_KEY                                 │
│                                                              │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          │ POST https://api.openai.com/v1/chat/completions
                          │ Authorization: Bearer sk-proj-xxxxx
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│ OPENAI API                                                   │
│ - Vérifie sk-proj-xxxxx                                     │
│ - Analyse l'image                                            │
│ - Retourne JSON                                              │
└─────────────────────────────────────────────────────────────┘
```

## ⚠️ Sécurité des Clés

### DO ✅

1. **Générer clé aléatoire forte**
   ```bash
   python3 -c "import secrets; print(f'nuno_service_{secrets.token_urlsafe(40)}')"
   ```

2. **Stocker dans variables d'environnement**
   ```bash
   # Django .env
   NUNOOCR_SERVICE_API_KEY=nuno_service_xxxxx

   # nunoOcr Dokploy env vars
   SERVICE_API_KEY=nuno_service_xxxxx
   ```

3. **Ne JAMAIS committer dans Git**
   ```bash
   # .gitignore
   .env
   .env.local
   ```

4. **Rotation régulière**
   - Changer la clé tous les 3-6 mois
   - Ou si suspicion de compromission

### DON'T ❌

1. **Ne pas utiliser clé simple**
   ```bash
   SERVICE_API_KEY=123456  # ❌ TROP SIMPLE
   ```

2. **Ne pas mettre dans le code**
   ```python
   # ❌ MAUVAIS
   SERVICE_API_KEY = "nuno_service_abc123"
   ```

3. **Ne pas réutiliser entre environnements**
   ```bash
   # Production: nuno_service_prod_xxxxx
   # Staging: nuno_service_staging_yyyyy
   # Dev: nuno_service_dev_zzzzz
   ```

## 🔄 Migration

### Étape 1: Ajouter Sécurité Sans Casser l'Existant

```bash
# nunoOcr - Activer mais pas forcer
SERVICE_API_KEY=nuno_service_xxxxx

# Le code vérifie mais log seulement si pas configurée
# → Backward compatible
```

### Étape 2: Déployer Client Django avec Clé

```python
# Django settings
NUNOOCR_SERVICE_API_KEY = 'nuno_service_xxxxx'
```

### Étape 3: Tester

```bash
# Vérifier que Django → nunoOcr fonctionne
curl https://inur.opefitoo.com/api/analyze-wound/ \
     -H "Authorization: Bearer nuno_user_abc123" \
     -F "wound_image=@wound.jpg"
```

### Étape 4: Forcer la Clé (optionnel)

Si vous voulez forcer absolument (empêcher accès sans clé):

```python
# Dans server_with_wound_analysis.py
# Modifier ligne 114:
if not SERVICE_API_KEY:
    raise HTTPException(
        status_code=500,
        detail="Service not configured - SERVICE_API_KEY required"
    )
```

## 📝 Checklist Déploiement

- [ ] Service API Key générée (format: `nuno_service_xxxxx`)
- [ ] `SERVICE_API_KEY` configurée dans nunoOcr (Dokploy env vars)
- [ ] `NUNOOCR_SERVICE_API_KEY` configurée dans Django (settings.py / .env)
- [ ] Les deux clés sont **identiques**
- [ ] Code mis à jour: `server_with_wound_analysis.py`
- [ ] Client Django mis à jour: `django_microservice_integration.py`
- [ ] Service nunoOcr redémarré
- [ ] Test sans clé (doit échouer - 401)
- [ ] Test avec mauvaise clé (doit échouer - 401)
- [ ] Test avec bonne clé (doit réussir - 200)
- [ ] Test end-to-end: Client → Django → nunoOcr → OpenAI
- [ ] (Optionnel) IP whitelist configurée `ALLOWED_IPS`
- [ ] (Optionnel) Test depuis IP non-whitelistée (doit échouer - 403)
- [ ] Clés sauvegardées de manière sécurisée
- [ ] `.env` dans `.gitignore`

## 🎉 Résultat

Vous avez maintenant:
- ✅ Authentification User → Django (API Key user)
- ✅ Authentification Django → nunoOcr (API Key service)
- ✅ Authentification nunoOcr → OpenAI (Clé OpenAI)
- ✅ (Optionnel) IP whitelist
- ✅ Impossible d'appeler nunoOcr sans la clé service
- ✅ Logs de toutes les tentatives

**Triple sécurité!** 🔐🔐🔐

---

**Version**: 2.0.0
**Date**: 2025-01-07
