# 🔑 Comprendre les Deux Types de Clés API

## ⚠️ NE PAS CONFONDRE!

Il y a **DEUX** types de clés API différentes dans votre architecture:

---

## 1️⃣ OPENAI_API_KEY (Clé OpenAI)

### 📌 C'est Quoi?
La clé secrète fournie par OpenAI pour utiliser leur API GPT-4 Vision.

### 🎯 Où l'Utiliser?
**Dans votre app Django `inur`** (celle qui appelle le service d'analyse de plaies)

### 💰 Qui Paye?
**Vous** - chaque appel à GPT-4 Vision est facturé sur votre compte OpenAI

### 📍 Où la Configurer?

#### Option A: Dans le .env de votre app Django
```bash
# Dans votre projet inur.django/.env
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
VISION_PROVIDER=openai
```

#### Option B: Dans les variables d'environnement Dokploy
Si votre app Django tourne sur Dokploy, ajoutez dans l'interface Dokploy:
```
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
```

### 🔧 Comment l'Utiliser dans Django?
```python
# Dans votre view Django (inur/views.py)
from nunoocr_client import DjangoOCRService
import os

def analyze_wound_view(request):
    ocr = DjangoOCRService(
        vision_api_key=os.getenv('OPENAI_API_KEY'),  # ← Votre clé OpenAI
        vision_provider='openai'
    )
    result = ocr.analyze_wound_from_uploaded_file(request.FILES['wound_image'])
```

### ✅ Status Actuel
Vous l'avez déjà créée: `sk-proj-rHu_SrM8g...` (visible dans votre screenshot billing)

**⚠️ PROBLÈME ACTUEL**: Vous n'avez pas ajouté de crédits! Vous devez:
1. Aller sur https://platform.openai.com/settings/organization/billing/overview
2. Ajouter une carte de crédit
3. Ajouter $5-10 de crédit

---

## 2️⃣ APIKey Model (Système d'Authentification)

### 📌 C'est Quoi?
Un système que **VOUS créez** dans Django pour authentifier **VOS utilisateurs** qui utilisent votre API.

### 🎯 Pourquoi Faire?
**PROTÉGER votre endpoint** pour éviter que n'importe qui appelle votre API et épuise vos crédits OpenAI!

### 💡 Exemple de Clé
```
nuno_abc123def456ghi789jkl012mno345pqr678stu901vwx234yz
```

### 🏗️ Architecture Complète

```
┌─────────────────────────────────────────────────────────────┐
│ CLIENT (Mobile App / Web App / Postman)                      │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ Header: Authorization: Bearer nuno_xxxxx
                        │         (APIKey - VOTRE système)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ DJANGO APP (inur.opefitoo.com)                               │
│                                                               │
│  1. Decorator @require_api_key vérifie la clé               │
│  2. Vérifie le quota (10/jour max)                          │
│  3. Si OK, appelle nunoocr_client                           │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        │ vision_api_key=OPENAI_API_KEY
                        │ (Clé OpenAI - leur système)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│ OPENAI API (api.openai.com)                                  │
│                                                               │
│  1. Vérifie OPENAI_API_KEY                                  │
│  2. Analyse l'image avec GPT-4 Vision                       │
│  3. Facture sur votre compte OpenAI                         │
└─────────────────────────────────────────────────────────────┘
```

### 📍 Où le Configurer?

**Dans votre app Django `inur`** (PAS dans le service nunoOcr!)

#### Étape 1: Ajouter le Modèle
```python
# inur/models.py
from django.db import models
from django.contrib.auth.models import User
import secrets

class APIKey(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    key = models.CharField(max_length=64, unique=True)
    name = models.CharField(max_length=100)
    daily_limit = models.IntegerField(default=10)
    calls_today = models.IntegerField(default=0)
    is_active = models.BooleanField(default=True)

    @staticmethod
    def generate_key():
        return f"nuno_{secrets.token_urlsafe(32)}"
```

#### Étape 2: Créer la Migration
```bash
cd /path/to/inur.django
python manage.py makemigrations
python manage.py migrate
```

#### Étape 3: Créer des Clés pour vos Utilisateurs
```python
# Django shell
python manage.py shell

from django.contrib.auth.models import User
from inur.models import APIKey

# Créer une clé pour l'utilisateur "mehdi"
user = User.objects.get(username='mehdi')
api_key = APIKey.objects.create(
    user=user,
    key=APIKey.generate_key(),
    name="Mobile App Production",
    daily_limit=50
)

print(f"Clé créée: {api_key.key}")
# Output: nuno_abc123def456...
```

#### Étape 4: Protéger votre Endpoint
```python
# inur/views.py
from django.http import JsonResponse
from .decorators import require_api_key

@require_api_key
def analyze_wound_api(request, api_key):
    """
    Cette view est maintenant protégée!
    Seuls les utilisateurs avec une APIKey valide peuvent l'appeler
    """
    # Vérifier quota (fait automatiquement par le decorator)

    # Analyser la plaie
    ocr = DjangoOCRService(
        vision_api_key=os.getenv('OPENAI_API_KEY')  # ← Clé OpenAI
    )
    result = ocr.analyze_wound_from_uploaded_file(request.FILES['wound_image'])

    return JsonResponse({
        'success': True,
        'data': result,
        'remaining_calls_today': api_key.daily_limit - api_key.calls_today
    })
```

---

## 🔐 Résumé: Les Deux Clés

| Aspect | OPENAI_API_KEY | APIKey Model |
|--------|----------------|--------------|
| **Type** | Clé externe (OpenAI) | Système interne (Django) |
| **Format** | `sk-proj-xxxxx` | `nuno_xxxxx` |
| **Où** | Variables d'env Django | Base de données Django |
| **But** | Authentifier VOUS auprès d'OpenAI | Authentifier VOS USERS auprès de vous |
| **Quota** | Limite OpenAI (rate limit leur côté) | Limite que VOUS définissez (10/jour) |
| **Coût** | Facturé par OpenAI ($0.01-0.03/image) | Gratuit (votre système) |
| **Où configurer** | `.env` Django ou Dokploy env vars | Django models + migrations |
| **Qui la voit** | Seulement votre backend Django | Vos utilisateurs (app mobile/web) |

---

## 🎯 Flow Complet d'un Appel

### 1. Client Envoie une Requête
```bash
curl -X POST https://inur.opefitoo.com/api/analyze-wound/ \
     -H "Authorization: Bearer nuno_abc123def456..." \  # ← APIKey (votre système)
     -F "wound_image=@wound.jpg"
```

### 2. Django Vérifie l'APIKey
```python
# Decorator @require_api_key
api_key = APIKey.objects.get(key='nuno_abc123def456...')
if api_key.calls_today >= api_key.daily_limit:
    return 429 "Quota dépassé"
```

### 3. Django Appelle OpenAI
```python
# nunoocr_client utilise OPENAI_API_KEY
headers = {
    'Authorization': f'Bearer {vision_api_key}'  # ← sk-proj-xxxxx (OpenAI)
}
response = requests.post('https://api.openai.com/v1/chat/completions', ...)
```

### 4. OpenAI Répond
```json
{
  "type_plaie": "ulcère de pression",
  "localisation": "cheville gauche",
  ...
}
```

### 5. Django Incrémente le Quota
```python
api_key.calls_today += 1
api_key.save()
```

---

## ✅ Actions à Faire MAINTENANT

### 1. Ajouter des Crédits OpenAI ⚠️ URGENT
- [ ] Aller sur https://platform.openai.com/settings/organization/billing/overview
- [ ] Ajouter carte de crédit
- [ ] Ajouter $5-10 de crédit

### 2. Configurer OPENAI_API_KEY dans Django
```bash
# Dans votre app Django inur
# Fichier .env ou variables Dokploy
OPENAI_API_KEY=sk-proj-rHu_SrM8g...  # Votre clé OpenAI
VISION_PROVIDER=openai
```

### 3. Implémenter le Système APIKey
```bash
# Dans votre app Django inur
cd /path/to/inur.django

# Copier le modèle APIKey depuis django_api_key_auth.py
# Ajouter dans inur/models.py

# Créer la migration
python manage.py makemigrations
python manage.py migrate

# Créer une première clé de test
python manage.py shell
# ... (voir code ci-dessus)
```

### 4. Protéger votre Endpoint
```python
# Ajouter le decorator @require_api_key
# Voir INTEGRATION_CHECKLIST.md pour le code complet
```

---

## 🆘 Où Demander de l'Aide?

- **OPENAI_API_KEY**: Documentation OpenAI - https://platform.openai.com/docs/api-reference/authentication
- **APIKey Model**: `API_KEY_SETUP.md` dans ce repo
- **Intégration**: `INTEGRATION_CHECKLIST.md` dans ce repo

---

**TL;DR**:
- `OPENAI_API_KEY` = Vous payez OpenAI pour l'analyse → Configurez dans `.env` Django
- `APIKey Model` = Vos users vous donnent leur clé → Créez dans Django models

Les deux sont **NÉCESSAIRES** et **COMPLÉMENTAIRES**! 🔐
