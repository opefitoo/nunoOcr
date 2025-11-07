# Configuration Double Endpoint: DeepSeek-OCR + Vision API

Ce système utilise **deux services différents** pour maximiser la qualité:

1. **DeepSeek-OCR** (self-hosted) → Extraction de texte des ordonnances
2. **GPT-4 Vision / Claude Vision** (API cloud) → Analyse médicale des plaies

## 🎯 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Application Django                     │
└─────────────────┬──────────────────┬───────────────────┘
                  │                  │
    ┌─────────────▼──────┐  ┌───────▼──────────────┐
    │  DeepSeek-OCR      │  │  Vision API          │
    │  (Self-hosted)     │  │  (Cloud)             │
    │                    │  │                      │
    │  • Ordonnances     │  │  • Analyse plaies    │
    │  • Extraction OCR  │  │  • GPT-4 Vision      │
    │  • Text seulement  │  │  • Claude Vision     │
    └────────────────────┘  └──────────────────────┘
```

## ⚙️ Configuration

### 1. Variables d'Environnement

Créez un fichier `.env.vision` (copier depuis `.env.vision.example`):

```bash
# Service OCR pour ordonnances
OCR_SERVICE_URL=https://nunoocrapi.opefitoo.com
OCR_SERVICE_API_KEY=

# Provider de vision (openai ou anthropic)
VISION_PROVIDER=openai

# Clés API
OPENAI_API_KEY=sk-proj-...
# OU
ANTHROPIC_API_KEY=sk-ant-...
```

### 2. Django Settings

Dans votre `settings.py`:

```python
import environ

env = environ.Env()

# Service OCR (DeepSeek) pour ordonnances
OCR_SERVICE_URL = env('OCR_SERVICE_URL', default='http://localhost:8765')
OCR_SERVICE_API_KEY = env('OCR_SERVICE_API_KEY', default='')

# Vision API pour analyse de plaies
VISION_PROVIDER = env('VISION_PROVIDER', default='openai')  # 'openai' ou 'anthropic'

# Clé API selon le provider
if VISION_PROVIDER == 'openai':
    VISION_API_KEY = env('OPENAI_API_KEY')
else:
    VISION_API_KEY = env('ANTHROPIC_API_KEY')
```

## 💻 Utilisation

### Python Standalone

```python
from nunoocr_client import DeepSeekOCRClient

# Configuration pour les deux services
client = DeepSeekOCRClient(
    base_url="https://nunoocrapi.opefitoo.com",  # DeepSeek pour ordonnances
    vision_api_key="sk-...",                      # GPT-4V pour plaies
    vision_provider="openai"
)

# Extraire une ordonnance (utilise DeepSeek-OCR)
with open('ordonnance.pdf', 'rb') as f:
    prescription = client.extract_prescription_data(f)
    print(prescription['medications'])

# Analyser une plaie (utilise GPT-4 Vision)
with open('plaie.jpg', 'rb') as f:
    wound = client.analyze_wound(f)
    print(wound['type_plaie'])
    print(wound['etat_general'])
```

### Django Integration

```python
from nunoocr_client import DjangoOCRService

# Le service se configure automatiquement depuis settings.py
ocr_service = DjangoOCRService()

# Ordonnances → DeepSeek-OCR
prescription_data = ocr_service.extract_from_uploaded_file(
    request.FILES['prescription'],
    extract_structured=True
)

# Plaies → GPT-4 Vision
wound_data = ocr_service.analyze_wound_from_uploaded_file(
    request.FILES['wound_image'],
    return_structured=True
)
```

## 📝 Exemple Vue Django

```python
from django.shortcuts import render
from django.http import JsonResponse
from nunoocr_client import DjangoOCRService

def analyze_document_view(request):
    """
    Endpoint unique qui route vers le bon service
    selon le type de document
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST requis'}, status=405)

    ocr = DjangoOCRService()

    document_type = request.POST.get('type')  # 'prescription' ou 'wound'
    file = request.FILES.get('document')

    if not file:
        return JsonResponse({'error': 'Aucun fichier fourni'}, status=400)

    try:
        if document_type == 'prescription':
            # Utilise DeepSeek-OCR (self-hosted)
            result = ocr.extract_from_uploaded_file(
                file,
                extract_structured=True
            )
            return JsonResponse({
                'success': True,
                'type': 'prescription',
                'data': result,
                'service': 'deepseek-ocr'
            })

        elif document_type == 'wound':
            # Utilise GPT-4 Vision (cloud API)
            result = ocr.analyze_wound_from_uploaded_file(
                file,
                return_structured=True
            )
            return JsonResponse({
                'success': True,
                'type': 'wound',
                'data': result,
                'service': result.get('_metadata', {}).get('provider', 'vision-api')
            })

        else:
            return JsonResponse({
                'error': 'Type invalide',
                'valid_types': ['prescription', 'wound']
            }, status=400)

    except Exception as e:
        return JsonResponse({
            'error': 'Échec de l\'analyse',
            'detail': str(e)
        }, status=500)
```

## 🧪 Tests

### Test Analyse de Plaie avec GPT-4 Vision

```bash
# Définir la clé API
export OPENAI_API_KEY="sk-proj-..."

# Tester
python test_wound_vision.py wound.jpg
```

### Test avec Claude Vision

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python test_wound_vision.py wound.jpg --provider anthropic
```

### Test Ordonnance avec DeepSeek

```bash
python test_ocr.py prescription.pdf https://nunoocrapi.opefitoo.com
```

## 💰 Coûts

### DeepSeek-OCR (Self-hosted)
- **Coût**: Serveur uniquement (CX53: ~15€/mois)
- **Usage**: Illimité
- **Latence**: 30-60s par page
- **Usage recommandé**: Ordonnances (texte structuré)

### GPT-4 Vision (OpenAI)
- **Coût**: ~$0.01-0.03 par image (selon détail)
- **Latence**: 5-15s par image
- **Qualité**: Excellente pour analyse médicale
- **Model**: `gpt-4o` (recommandé) ou `gpt-4o-mini` (économique)

### Claude Vision (Anthropic)
- **Coût**: ~$0.015-0.075 par image
- **Latence**: 5-15s par image
- **Qualité**: Excellente, très détaillée
- **Model**: `claude-3-5-sonnet-20241022`

## 🔒 Sécurité

### Données Sensibles

1. **Ordonnances** (DeepSeek self-hosted):
   - ✅ Restent sur votre infrastructure
   - ✅ Conformité RGPD totale
   - ✅ Aucune donnée envoyée à des tiers

2. **Images de plaies** (Vision API cloud):
   - ⚠️  Envoyées à OpenAI/Anthropic
   - ⚠️  Lire les politiques de confidentialité:
     - [OpenAI Data Policy](https://openai.com/policies/privacy-policy)
     - [Anthropic Privacy](https://www.anthropic.com/legal/privacy)
   - ✅ Pas de stockage selon les politiques (API calls)
   - ✅ Anonymisation recommandée

### Recommandations

1. **Anonymiser les images de plaies** avant envoi:
   - Retirer métadonnées EXIF
   - Masquer éventuels tatouages/marques distinctives
   - Ne pas inclure visage du patient

2. **Informer les patients**:
   - Usage d'API cloud pour analyse
   - Consentement explicite

3. **Alternative self-hosted**:
   - Pour conformité stricte, envisager:
     - Llama 3.2 Vision (90B)
     - Qwen-VL
     - BiomedCLIP
   - Nécessite GPU puissant (A100/H100)

## 📊 Comparaison des Services

| Critère | DeepSeek-OCR | GPT-4 Vision | Claude Vision |
|---------|--------------|--------------|---------------|
| **Type** | Self-hosted | Cloud API | Cloud API |
| **Usage** | OCR texte | Vision générale | Vision détaillée |
| **Coût** | Fixe (~15€/mois) | Variable ($0.01-0.03/img) | Variable ($0.015-0.075/img) |
| **Latence** | 30-60s | 5-15s | 5-15s |
| **RGPD** | ✅ 100% | ⚠️  Cloud | ⚠️  Cloud |
| **Qualité OCR** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Analyse médicale** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🚀 Déploiement Production

### Configuration Recommandée

```python
# settings.py (production)

# Pour ordonnances: service self-hosted
OCR_SERVICE_URL = env('OCR_SERVICE_URL')
OCR_SERVICE_API_KEY = env('OCR_SERVICE_API_KEY')

# Pour plaies: GPT-4 Vision (meilleur rapport qualité/prix)
VISION_PROVIDER = 'openai'
VISION_API_KEY = env('OPENAI_API_KEY')

# Monitoring
SENTRY_DSN = env('SENTRY_DSN')  # Pour tracker erreurs API
```

### Rate Limiting

```python
from django.core.cache import cache
from django.utils.decorators import method_decorator
from django.views.decorators.cache import cache_page

# Limiter les appels Vision API (coûteux)
@method_decorator(cache_page(60 * 5), name='dispatch')  # Cache 5 min
class WoundAnalysisView(View):
    def post(self, request):
        # Vérifier rate limit par utilisateur
        user_id = request.user.id
        cache_key = f'wound_analysis_{user_id}'

        if cache.get(cache_key):
            return JsonResponse({
                'error': 'Trop de requêtes, attendez 1 minute'
            }, status=429)

        cache.set(cache_key, True, 60)  # 1 min cooldown

        # Faire l'analyse...
```

## 📚 Documentation Complète

- [WOUND_ANALYSIS_README.md](WOUND_ANALYSIS_README.md) - Guide complet analyse de plaies
- [README.md](README.md) - Configuration générale
- [QUICKSTART.md](QUICKSTART.md) - Démarrage rapide

## 🆘 Support

### Problèmes Courants

**Vision API ne fonctionne pas:**
```bash
# Vérifier la clé API
python test_wound_vision.py wound.jpg

# Si erreur 401: clé invalide
# Si erreur 429: quota dépassé
# Si erreur 500: format image incompatible
```

**DeepSeek OCR ne répond pas:**
```bash
# Vérifier le service
curl https://nunoocrapi.opefitoo.com/health
```

---

**Version**: 2.0.0
**Date**: 2025-01-07
**Architecture**: Dual-Endpoint (Self-hosted + Cloud API)
