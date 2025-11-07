# ✅ Checklist d'Intégration - API Wound Analysis

## 📦 Fichiers Disponibles

Tous les fichiers nécessaires sont maintenant dans le repo:

- ✅ `nunoocr_client.py` - Client Python avec support GPT-4 Vision/Claude
- ✅ `django_api_key_auth.py` - Système d'authentification par API Key
- ✅ `API_KEY_SETUP.md` - Guide complet d'installation
- ✅ `API_SECURITY.md` - Guide de sécurisation
- ✅ `django_rate_limiting_example.py` - Exemples de rate limiting
- ✅ `DUAL_ENDPOINT_README.md` - Architecture et coûts
- ✅ `wound_analysis_example.py` - Exemples d'utilisation

## 🔑 Configuration OpenAI (REQUIS)

### 1. Ajouter des Crédits OpenAI

Votre compte OpenAI est actuellement en "Free Trial $0.00". Vous devez:

1. Aller sur https://platform.openai.com/settings/organization/billing/overview
2. Cliquer "Add payment method"
3. Ajouter une carte de crédit
4. Ajouter au minimum $5-10 de crédit

**Coût estimé**: ~$0.01-0.03 par image de plaie analysée

### 2. Variables d'Environnement

Dans votre environnement Docker/Dokploy, assurez-vous d'avoir:

```bash
OPENAI_API_KEY=sk-proj-xxxxx  # Déjà configuré ✅
VISION_PROVIDER=openai         # openai ou anthropic
```

## 🔧 Intégration Django

### Étape 1: Ajouter le Modèle APIKey

Dans votre app Django `inur.django`, ajoutez le modèle APIKey:

**Fichier**: `inur/models.py` (ou créez `inur/api_models.py`)

```python
# Copier le contenu de django_api_key_auth.py
# Depuis la ligne 108 à 175 (classe APIKey)
```

### Étape 2: Migration

```bash
python manage.py makemigrations
python manage.py migrate
```

### Étape 3: Créer le Decorator

**Fichier**: `inur/decorators.py`

```python
# Copier le decorator require_api_key
# Depuis django_api_key_auth.py lignes 181-233
```

### Étape 4: Créer la View API

**Fichier**: `inur/views.py` ou `inur/api_views.py`

```python
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.http import JsonResponse
from .decorators import require_api_key
from nunoocr_client import DjangoOCRService

@csrf_exempt
@require_POST
@require_api_key
def analyze_wound_api(request, api_key):
    """
    Endpoint protégé pour l'analyse de plaies

    Usage:
        curl -X POST https://inur.opefitoo.com/api/analyze-wound/ \
             -H "Authorization: Bearer nuno_xxxxx" \
             -F "wound_image=@wound.jpg"
    """
    if 'wound_image' not in request.FILES:
        return JsonResponse({'error': 'Image requise'}, status=400)

    file = request.FILES['wound_image']

    # Validation taille (5MB max)
    if file.size > 5 * 1024 * 1024:
        return JsonResponse({'error': 'Image trop grande (max 5MB)'}, status=400)

    try:
        # Initialiser le service avec vision API
        ocr = DjangoOCRService(
            vision_api_key=settings.OPENAI_API_KEY,  # ou settings.VISION_API_KEY
            vision_provider='openai'
        )

        # Analyser la plaie
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
```

### Étape 5: Ajouter l'URL

**Fichier**: `inur/urls.py`

```python
from django.urls import path
from . import api_views  # ou views

urlpatterns = [
    # ... vos URLs existantes ...
    path('api/analyze-wound/', api_views.analyze_wound_api, name='analyze_wound_api'),
]
```

### Étape 6: Admin Interface

**Fichier**: `inur/admin.py`

```python
from django.contrib import admin
from .models import APIKey  # ou .api_models

@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = ['name', 'user', 'is_active', 'calls_today', 'daily_limit', 'created_at']
    readonly_fields = ['key', 'created_at', 'last_used', 'total_calls']

    def save_model(self, request, obj, form, change):
        if not change:  # Nouvelle clé
            obj.key = APIKey.generate_key()
        super().save_model(request, obj, form, change)
```

## 🧪 Tester l'API

### 1. Créer une API Key

Dans Django Admin:
1. Aller dans "API Keys"
2. Cliquer "Add API Key"
3. Sélectionner un utilisateur
4. Nom: "Test Key"
5. Daily limit: 10
6. Sauvegarder
7. **Copier la clé générée** (commence par `nuno_`)

Ou via shell:
```bash
python manage.py shell

from django.contrib.auth.models import User
from inur.models import APIKey

user = User.objects.get(username='votre_username')
api_key = APIKey.objects.create(
    user=user,
    key=APIKey.generate_key(),
    name="Test Production",
    daily_limit=50
)

print(f"API Key: {api_key.key}")
```

### 2. Tester avec curl

```bash
# Remplacer YOUR_API_KEY par la clé générée
curl -X POST https://inur.opefitoo.com/api/analyze-wound/ \
     -H "Authorization: Bearer nuno_abc123def456..." \
     -F "wound_image=@wound.jpg"
```

**Réponse attendue:**
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
        "confiance": "élevée"
    },
    "api_key_name": "Test Key",
    "remaining_calls_today": 9
}
```

### 3. Tester avec Python

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

## 🔒 Sécurité - Checklist

- [ ] OpenAI API Key configurée dans l'environnement
- [ ] Modèle APIKey créé et migré
- [ ] Decorator `@require_api_key` implémenté
- [ ] Rate limiting activé (quotas quotidiens)
- [ ] Validation des fichiers (max 5MB)
- [ ] HTTPS obligatoire en production
- [ ] API Keys créées pour utilisateurs autorisés
- [ ] Admin interface configurée
- [ ] Tests effectués (curl/Postman)

## 📊 Monitoring

### Vérifier l'Usage

```python
from inur.models import APIKey

# Usage d'une clé spécifique
key = APIKey.objects.get(name="Production")
print(f"Appels aujourd'hui: {key.calls_today}/{key.daily_limit}")
print(f"Total: {key.total_calls}")

# Toutes les clés actives
for key in APIKey.objects.filter(is_active=True):
    print(f"{key.name}: {key.calls_today}/{key.daily_limit}")
```

## 💰 Coûts Estimés

### GPT-4o Vision (recommandé)
- **Prix**: ~$0.01-0.03 par image
- **Quota 10/jour**: ~$0.10-0.30/jour max
- **Budget mensuel**: ~$3-9/mois (si 10 images/jour)

### Conseil
Commencez avec un quota de 10/jour par utilisateur. Ajustez selon l'usage réel.

## 🆘 Aide

Si vous rencontrez des problèmes:

1. **Erreur 401 "API Key invalide"**
   - Vérifiez le format: `Authorization: Bearer nuno_xxxxx`
   - Vérifiez que la clé est active dans l'admin

2. **Erreur 429 "Quota dépassé"**
   - L'utilisateur a atteint sa limite quotidienne
   - Augmentez le `daily_limit` ou attendez minuit

3. **Erreur 500 "Analyse échouée"**
   - Vérifiez que `OPENAI_API_KEY` est configurée
   - Vérifiez que vous avez des crédits OpenAI
   - Vérifiez les logs Django pour plus de détails

4. **OpenAI 429 "Too Many Requests"**
   - Vous n'avez pas de crédits OpenAI
   - Ajoutez une carte de crédit et $5-10 de crédit

## 📚 Documentation Complète

- `API_KEY_SETUP.md` - Guide détaillé d'installation
- `API_SECURITY.md` - Guide de sécurisation et rate limiting
- `DUAL_ENDPOINT_README.md` - Architecture et coûts
- `django_api_key_auth.py` - Code complet de référence

---

**Status**: ✅ Prêt pour l'intégration
**Version**: 1.0.0
**Date**: 2025-01-07
