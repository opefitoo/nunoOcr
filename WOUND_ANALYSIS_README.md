# Analyse de Plaies - DeepSeek-OCR

Service d'analyse de plaies médicales utilisant DeepSeek-OCR avec support complet du français pour:
- Analyse d'images de plaies individuelles
- Suivi de progression avec images multiples
- Intégration Django complète

## 🚀 Fonctionnalités

### 1. Analyse de Plaie Individuelle

Analyse détaillée d'une image de plaie avec extraction de:
- Type de plaie (incision chirurgicale, lacération, ulcère, etc.)
- Localisation anatomique
- Dimensions (longueur × largeur en cm)
- Stade de cicatrisation
- Méthode de fermeture (points, agrafes, adhésif)
- Signes d'infection
- État général et recommandations

### 2. Analyse de Progression

Comparaison de plusieurs images prises à différentes dates pour évaluer:
- Évolution globale (amélioration/stable/détérioration)
- Changements de dimensions
- Progression de la cicatrisation
- Évolution des signes d'infection
- Recommandations médicales adaptées
- Planification du prochain contrôle

## 📋 Structure des Données

### Analyse Individuelle (JSON)

```json
{
  "type_plaie": "incision chirurgicale",
  "localisation": "cheville gauche",
  "dimensions": {
    "longueur_cm": 3.5,
    "largeur_cm": 0.5
  },
  "stade_cicatrisation": "en cours de cicatrisation",
  "methode_fermeture": "points de suture",
  "nombre_points": 8,
  "signes_infection": [],
  "complications": [],
  "etat_general": "Plaie propre en voie de cicatrisation normale",
  "confiance": "élevée",
  "notes": "Points de suture intacts, pas de signes d'infection"
}
```

### Analyse de Progression (JSON)

```json
{
  "periode_analyse": "14 jours",
  "nombre_evaluations": 3,
  "evolution_globale": "amélioration significative",
  "ameliorations": [
    "Réduction importante de la taille de la plaie",
    "Disparition des signes inflammatoires",
    "Progression normale de la cicatrisation"
  ],
  "preoccupations": [],
  "changement_dimensions": {
    "evolution": "réduction",
    "pourcentage": -35.2
  },
  "cicatrisation_progression": "Évolution favorable avec réépithélialisation progressive",
  "infection_evolution": "aucune infection",
  "recommandations": [
    "Continuer les soins actuels",
    "Maintenir la plaie propre et sèche",
    "Ablation des points prévue dans 3-5 jours"
  ],
  "prochain_controle": "3-5 jours",
  "notes_progression": "Cicatrisation conforme aux attentes..."
}
```

## 🔧 Installation

```bash
# Le service utilise la même infrastructure DeepSeek-OCR
# Aucune installation supplémentaire requise

# Assurez-vous que le service est démarré
docker compose up -d
```

## 💻 Utilisation

### Python Standalone

#### Analyse d'une plaie

```python
from nunoocr_client import DeepSeekOCRClient

client = DeepSeekOCRClient(base_url="http://localhost:8765")

# Analyser une plaie
with open('plaie.jpg', 'rb') as f:
    analyse = client.analyze_wound(f, return_structured=True)

print(f"Type: {analyse['type_plaie']}")
print(f"Localisation: {analyse['localisation']}")
print(f"État: {analyse['etat_general']}")
```

#### Analyse de progression

```python
from nunoocr_client import DeepSeekOCRClient

client = DeepSeekOCRClient(base_url="http://localhost:8765")

# Préparer les images avec dates
images = [
    {
        'file_obj': open('plaie_jour1.jpg', 'rb'),
        'date': '2025-01-01',
        'notes': 'Plaie initiale post-opératoire'
    },
    {
        'file_obj': open('plaie_jour7.jpg', 'rb'),
        'date': '2025-01-07',
        'notes': 'Premier contrôle'
    },
    {
        'file_obj': open('plaie_jour14.jpg', 'rb'),
        'date': '2025-01-14',
        'notes': 'Contrôle avant ablation des points'
    }
]

# Analyser la progression
progression = client.compare_wound_progress(images, return_structured=True)

print(f"Évolution: {progression['evolution_globale']}")
print(f"Améliorations: {progression['ameliorations']}")
print(f"Recommandations: {progression['recommandations']}")
print(f"Prochain contrôle: {progression['prochain_controle']}")
```

### Intégration Django

#### Configuration (settings.py)

```python
# Service OCR
OCR_SERVICE_URL = env('OCR_SERVICE_URL', default='http://localhost:8765')
OCR_SERVICE_API_KEY = env('OCR_SERVICE_API_KEY', default='')
```

#### Modèles (models.py)

```python
from django.db import models

class EvaluationPlaie(models.Model):
    """Évaluation d'une plaie"""
    patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='plaies/%Y/%m/')

    # Résultats d'analyse
    type_plaie = models.CharField(max_length=100)
    localisation = models.CharField(max_length=200)
    longueur_cm = models.FloatField(null=True)
    largeur_cm = models.FloatField(null=True)
    stade_cicatrisation = models.CharField(max_length=50)
    etat_general = models.TextField()

    analyzed_at = models.DateTimeField(auto_now_add=True)
    analyse_brute = models.JSONField()

    class Meta:
        ordering = ['-analyzed_at']


class ProgressionPlaie(models.Model):
    """Analyse de progression"""
    patient = models.ForeignKey('Patient', on_delete=models.CASCADE)
    evaluations = models.ManyToManyField(EvaluationPlaie)

    evolution_globale = models.CharField(max_length=100)
    ameliorations = models.JSONField(default=list)
    preoccupations = models.JSONField(default=list)
    recommandations = models.JSONField(default=list)
    prochain_controle = models.CharField(max_length=50)

    created_at = models.DateTimeField(auto_now_add=True)
    analyse_brute = models.JSONField()
```

#### Vues (views.py)

```python
from django.http import JsonResponse
from nunoocr_client import DjangoOCRService

def analyser_plaie_view(request, patient_id):
    """Analyser une nouvelle image de plaie"""
    ocr = DjangoOCRService()

    if not ocr.is_available():
        return JsonResponse({'error': 'Service indisponible'}, status=503)

    # Analyser l'image
    analyse = ocr.analyze_wound_from_uploaded_file(
        request.FILES['image_plaie'],
        return_structured=True
    )

    # Sauvegarder l'évaluation
    evaluation = EvaluationPlaie.objects.create(
        patient_id=patient_id,
        image=request.FILES['image_plaie'],
        type_plaie=analyse['type_plaie'],
        localisation=analyse['localisation'],
        # ... autres champs
        analyse_brute=analyse
    )

    return JsonResponse({'success': True, 'analyse': analyse})


def analyser_progression_view(request, patient_id):
    """Analyser la progression des plaies d'un patient"""
    ocr = DjangoOCRService()

    # Récupérer toutes les évaluations du patient
    evaluations = EvaluationPlaie.objects.filter(
        patient_id=patient_id
    ).order_by('analyzed_at')

    if evaluations.count() < 2:
        return JsonResponse({
            'error': 'Au moins 2 évaluations requises'
        }, status=400)

    # Analyser la progression
    progression = ocr.compare_wound_progress_from_model(
        evaluations,
        return_structured=True
    )

    # Sauvegarder le rapport
    rapport = ProgressionPlaie.objects.create(
        patient_id=patient_id,
        evolution_globale=progression['evolution_globale'],
        ameliorations=progression['ameliorations'],
        # ... autres champs
        analyse_brute=progression
    )
    rapport.evaluations.set(evaluations)

    return JsonResponse({'success': True, 'progression': progression})
```

## 📝 Scripts d'Exemple

### 1. Test avec une image réelle

```bash
# Test local
python test_real_wound.py plaie.jpg

# Test avec service distant
python test_real_wound.py plaie.jpg https://nunoocrapi.opefitoo.com
```

### 2. Analyse de progression

```bash
# Analyser plusieurs images avec dates
python wound_progression_example.py \
    plaie_jour1.jpg:2025-01-01 \
    plaie_jour7.jpg:2025-01-07 \
    plaie_jour14.jpg:2025-01-14

# Afficher l'exemple Django
python wound_progression_example.py --django-example
```

### 3. Exemples complets

```bash
# Voir tous les exemples d'utilisation
python wound_analysis_example.py --django-example
```

## 🔍 Cas d'Usage

### 1. Suivi Post-Opératoire

Documenter la cicatrisation après une intervention chirurgicale:
- Jour 0: Plaie post-opératoire immédiate
- Jour 7: Premier contrôle
- Jour 14: Contrôle avant ablation des points
- Jour 21: Contrôle final

### 2. Traitement d'Ulcères

Suivre l'évolution d'ulcères chroniques:
- Semaine 1: État initial
- Semaine 4: Après 1 mois de traitement
- Semaine 8: Évaluation à mi-parcours
- Semaine 12: Évaluation finale

### 3. Soins à Domicile

Permettre aux patients de documenter leurs plaies:
- Photos prises par le patient/famille
- Analyse automatique
- Alertes en cas de détérioration
- Consultation à distance

### 4. Dossier Médical Électronique

Intégration dans les systèmes existants:
- Documentation automatique
- Rapports structurés
- Traçabilité complète
- Export pour assurance/administration

## ⚙️ Configuration Avancée

### Timeout pour Analyses Longues

```python
# Pour les analyses de progression complexes
client = DeepSeekOCRClient(
    base_url="http://localhost:8765",
    timeout=300  # 5 minutes
)
```

### Analyse Non-Structurée

```python
# Obtenir une description textuelle au lieu de JSON
analyse = client.analyze_wound(
    file_obj,
    return_structured=False
)
print(analyse['analysis'])  # Texte descriptif
```

## 🧪 Tests

```bash
# Test du service de base
python test_wound_analysis.py https://nunoocrapi.opefitoo.com wound.jpg

# Test avec image réelle
python test_real_wound.py wound.jpg

# Test de progression
python wound_progression_example.py \
    wound1.jpg:2025-01-01 \
    wound2.jpg:2025-01-08
```

## 📊 Performance

- **Analyse individuelle**: 30-60 secondes par image
- **Analyse de progression**: 1-3 minutes pour 2-5 images
- **Mémoire requise**: 8-12 GB RAM (serveur)
- **Précision**: Élevée pour plaies clairement visibles

## 🔒 Considérations Médicales

⚠️ **Important**:
- Ce service est un **outil d'aide à la documentation**, pas un diagnostic médical
- Les résultats doivent être validés par un professionnel de santé
- Ne remplace pas l'examen clinique
- Respecter les règlements RGPD pour les données de santé
- Stocker les images de manière sécurisée et conforme

## 📚 Ressources

### Fichiers

- `nunoocr_client.py`: Client Python avec méthodes d'analyse
- `test_real_wound.py`: Test avec images réelles
- `wound_analysis_example.py`: Exemples d'utilisation
- `wound_progression_example.py`: Exemples de progression
- `WOUND_ANALYSIS_README.md`: Cette documentation

### Documentation Complémentaire

- [README principal](README.md): Configuration générale du service
- [Guide Django](django_integration_example.py): Intégration complète
- [API Reference](nunoocr_client.py): Documentation du code

## 🆘 Support

### Problèmes Courants

**Le service ne répond pas:**
```bash
# Vérifier l'état du service
docker compose ps

# Voir les logs
docker compose logs -f deepseek-ocr
```

**Timeout sur les analyses:**
```python
# Augmenter le timeout
client = DeepSeekOCRClient(timeout=600)  # 10 minutes
```

**Erreur de parsing JSON:**
- Le modèle peut parfois retourner du texte supplémentaire
- Utilisez `return_structured=False` pour obtenir le texte brut
- Vérifiez les logs pour voir la réponse complète

## 📞 Contact

Pour questions et support:
- Issues GitHub: [votre-repo]/issues
- Documentation: [votre-docs]
- Email: [votre-email]

---

**Version**: 1.0.0
**Date**: 2025-01-07
**Langue**: Français (FR)
**Modèle**: DeepSeek-OCR via transformers
