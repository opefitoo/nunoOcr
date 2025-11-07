#!/usr/bin/env python3
"""
Exemple d'analyse de progression de plaies - Comparaison d'images multiples avec dates

Cet exemple montre comment utiliser le service d'analyse de progression pour:
1. Comparer plusieurs images de plaies prises à différentes dates
2. Évaluer l'évolution de la cicatrisation
3. Obtenir des recommandations médicales basées sur la progression
4. Intégrer dans Django pour le suivi des patients

Usage:
    # Analyse de progression avec plusieurs images
    python wound_progression_example.py image1.jpg:2025-01-01 image2.jpg:2025-01-07 image3.jpg:2025-01-14

    # Avec URL de service personnalisée
    python wound_progression_example.py --url http://localhost:8765 img1.jpg:2025-01-01 img2.jpg:2025-01-07

    # Afficher uniquement l'exemple Django
    python wound_progression_example.py --django-example
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
from nunoocr_client import DeepSeekOCRClient, DjangoOCRService, DeepSeekOCRError
import json


def parse_image_arg(arg: str):
    """Parse image:date argument"""
    parts = arg.split(':')
    if len(parts) != 2:
        raise ValueError(f"Format invalide: {arg}. Utilisez image.jpg:YYYY-MM-DD")

    image_path, date = parts
    if not Path(image_path).exists():
        raise ValueError(f"Image introuvable: {image_path}")

    # Validate date format
    try:
        datetime.strptime(date, '%Y-%m-%d')
    except ValueError:
        raise ValueError(f"Format de date invalide: {date}. Utilisez YYYY-MM-DD")

    return image_path, date


def example_progression_analysis(image_data_list, service_url: str = "http://localhost:8765"):
    """
    Exemple 1: Analyse de progression avec images multiples
    """
    print("=" * 80)
    print("Exemple 1: Analyse de Progression de Plaie")
    print("=" * 80)

    client = DeepSeekOCRClient(base_url=service_url, timeout=300)

    # Check service availability
    if not client.health_check():
        print(f"⚠️  Attention: Le service OCR à {service_url} ne répond pas")
        print("   Assurez-vous que le service est démarré avec: docker compose up")
        return False

    print(f"✅ Connecté au service OCR à {service_url}\n")

    try:
        # Prepare wound images list
        wound_images = []
        print("Images à analyser:")
        for image_path, date in image_data_list:
            print(f"  - {Path(image_path).name} ({date})")
            with open(image_path, 'rb') as f:
                # Read file into memory since we can't reuse file handles
                file_content = f.read()
                import io
                wound_images.append({
                    'file_obj': io.BytesIO(file_content),
                    'date': date,
                    'file_type': 'image'
                })

        print(f"\nAnalyse de {len(wound_images)} images...")
        print("Cela peut prendre 1-3 minutes selon le nombre d'images...\n")

        # Get progression analysis
        import time
        start_time = time.time()
        result = client.compare_wound_progress(wound_images, return_structured=True)
        elapsed = time.time() - start_time

        print(f"✅ Analyse terminée en {elapsed:.1f} secondes!\n")
        print("=" * 80)
        print("RÉSULTATS DE L'ANALYSE DE PROGRESSION")
        print("=" * 80)

        if result.get('structured', True):
            # Display structured progression results
            print(f"\n📅 PÉRIODE D'ANALYSE: {result.get('periode_analyse', 'N/A')}")
            print(f"📊 NOMBRE D'ÉVALUATIONS: {result.get('nombre_evaluations', 'N/A')}")

            evolution = result.get('evolution_globale', 'N/A')
            print(f"\n🔍 ÉVOLUTION GLOBALE: {evolution}")

            # Color code based on evolution
            if 'amélioration' in evolution.lower():
                print("   ✅ Tendance positive détectée")
            elif 'détérioration' in evolution.lower():
                print("   ⚠️  Attention: détérioration détectée")
            elif 'stable' in evolution.lower():
                print("   ➡️  État stable")

            # Improvements
            ameliorations = result.get('ameliorations', [])
            if ameliorations:
                print(f"\n✅ AMÉLIORATIONS OBSERVÉES:")
                for item in ameliorations:
                    print(f"   • {item}")

            # Concerns
            preoccupations = result.get('preoccupations', [])
            if preoccupations:
                print(f"\n⚠️  PRÉOCCUPATIONS:")
                for item in preoccupations:
                    print(f"   • {item}")
            else:
                print(f"\n✅ AUCUNE PRÉOCCUPATION MAJEURE")

            # Dimension changes
            changement_dim = result.get('changement_dimensions', {})
            if changement_dim:
                evolution_dim = changement_dim.get('evolution', 'N/A')
                pourcentage = changement_dim.get('pourcentage')
                print(f"\n📏 CHANGEMENT DE DIMENSIONS: {evolution_dim}", end="")
                if pourcentage is not None:
                    print(f" ({pourcentage:+.1f}%)")
                else:
                    print()

            # Healing progression
            cicatrisation = result.get('cicatrisation_progression', 'N/A')
            print(f"\n🏥 PROGRESSION DE CICATRISATION:")
            print(f"   {cicatrisation}")

            # Infection evolution
            infection_evo = result.get('infection_evolution', 'N/A')
            print(f"\n🦠 ÉVOLUTION DE L'INFECTION: {infection_evo}")

            # Recommendations
            recommandations = result.get('recommandations', [])
            if recommandations:
                print(f"\n💊 RECOMMANDATIONS MÉDICALES:")
                for i, rec in enumerate(recommandations, 1):
                    print(f"   {i}. {rec}")

            # Next checkup
            prochain = result.get('prochain_controle', 'N/A')
            print(f"\n📅 PROCHAIN CONTRÔLE RECOMMANDÉ: {prochain}")

            # Detailed notes
            notes = result.get('notes_progression', '')
            if notes:
                print(f"\n📝 NOTES DÉTAILLÉES:")
                print(f"   {notes}")

            print("\n" + "=" * 80)
            print("ANALYSES INDIVIDUELLES")
            print("=" * 80)

            # Show individual analyses
            analyses_ind = result.get('analyses_individuelles', [])
            for i, analysis in enumerate(analyses_ind, 1):
                print(f"\nImage {i} - {analysis['date']}:")
                an = analysis['analysis']
                print(f"  Type: {an.get('type_plaie', 'N/A')}")
                dims = an.get('dimensions', {})
                if dims:
                    print(f"  Dimensions: {dims.get('longueur_cm', 'N/A')} x {dims.get('largeur_cm', 'N/A')} cm")
                print(f"  Stade: {an.get('stade_cicatrisation', 'N/A')}")
                print(f"  État: {an.get('etat_general', 'N/A')}")

            print("\n" + "=" * 80)
            print("RÉPONSE JSON COMPLÈTE")
            print("=" * 80)
            # Remove large nested data for cleaner display
            display_data = {k: v for k, v in result.items()
                          if k not in ['_metadata', 'analyses_individuelles']}
            print(json.dumps(display_data, indent=2, ensure_ascii=False))

        else:
            print("\n📄 ANALYSE TEXTUELLE:")
            print(result.get('comparison', 'Aucune analyse disponible'))

        print("\n" + "=" * 80)

        # Metadata
        metadata = result.get('_metadata', {})
        print(f"Tokens utilisés: {metadata.get('tokens_used', 'N/A')}")
        print(f"Modèle: {metadata.get('model', 'N/A')}")
        print("=" * 80)

        return True

    except FileNotFoundError as e:
        print(f"❌ Erreur: {e}")
        return False
    except DeepSeekOCRError as e:
        print(f"❌ Erreur OCR: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return False


def example_django_integration():
    """
    Exemple 2: Intégration Django pour le suivi de progression
    """
    print("\n" + "=" * 80)
    print("Exemple 2: Intégration Django - Code d'Exemple")
    print("=" * 80)

    django_code = '''
# Dans votre Django models.py:

from django.db import models
from django.contrib.auth.models import User

class Patient(models.Model):
    """Modèle patient"""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    nom = models.CharField(max_length=100)
    prenom = models.CharField(max_length=100)
    date_naissance = models.DateField()
    numero_securite_sociale = models.CharField(max_length=13)

    def __str__(self):
        return f"{self.prenom} {self.nom}"


class EvaluationPlaie(models.Model):
    """Modèle pour stocker les évaluations de plaies"""
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='plaies/%Y/%m/')

    # Informations d'analyse
    type_plaie = models.CharField(max_length=100)
    localisation = models.CharField(max_length=200)
    longueur_cm = models.FloatField(null=True, blank=True)
    largeur_cm = models.FloatField(null=True, blank=True)
    stade_cicatrisation = models.CharField(max_length=50)
    methode_fermeture = models.CharField(max_length=50)
    nombre_points = models.IntegerField(null=True, blank=True)
    etat_general = models.TextField()
    notes = models.TextField(blank=True)

    # Métadonnées
    confiance = models.CharField(max_length=20)
    analyzed_at = models.DateTimeField(auto_now_add=True)
    analyse_brute = models.JSONField()

    class Meta:
        ordering = ['-analyzed_at']
        verbose_name = "Évaluation de plaie"
        verbose_name_plural = "Évaluations de plaies"

    def __str__(self):
        return f"Évaluation {self.patient} - {self.analyzed_at.strftime('%Y-%m-%d')}"


class ProgressionPlaie(models.Model):
    """Modèle pour stocker les analyses de progression"""
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    evaluations = models.ManyToManyField(EvaluationPlaie)

    # Résultats de progression
    periode_analyse = models.CharField(max_length=50)
    nombre_evaluations = models.IntegerField()
    evolution_globale = models.CharField(max_length=100)
    ameliorations = models.JSONField(default=list)
    preoccupations = models.JSONField(default=list)
    cicatrisation_progression = models.TextField()
    infection_evolution = models.CharField(max_length=50)
    recommandations = models.JSONField(default=list)
    prochain_controle = models.CharField(max_length=50)
    notes_progression = models.TextField()

    # Métadonnées
    created_at = models.DateTimeField(auto_now_add=True)
    analyse_brute = models.JSONField()

    class Meta:
        ordering = ['-created_at']
        verbose_name = "Progression de plaie"
        verbose_name_plural = "Progressions de plaies"

    def __str__(self):
        return f"Progression {self.patient} - {self.created_at.strftime('%Y-%m-%d')}"


# Dans votre Django views.py:

from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from nunoocr_client import DjangoOCRService, DeepSeekOCRError
from .models import Patient, EvaluationPlaie, ProgressionPlaie

@csrf_exempt
def analyser_plaie_view(request, patient_id):
    """
    Endpoint API pour analyser une image de plaie
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST requis'}, status=405)

    if 'image_plaie' not in request.FILES:
        return JsonResponse({'error': 'Aucune image fournie'}, status=400)

    patient = get_object_or_404(Patient, id=patient_id)
    ocr = DjangoOCRService()

    if not ocr.is_available():
        return JsonResponse({
            'error': 'Service OCR indisponible'
        }, status=503)

    try:
        # Analyser la plaie
        analyse = ocr.analyze_wound_from_uploaded_file(
            request.FILES['image_plaie'],
            return_structured=True
        )

        # Créer l'évaluation
        evaluation = EvaluationPlaie.objects.create(
            patient=patient,
            image=request.FILES['image_plaie'],
            type_plaie=analyse.get('type_plaie', ''),
            localisation=analyse.get('localisation', ''),
            longueur_cm=analyse.get('dimensions', {}).get('longueur_cm'),
            largeur_cm=analyse.get('dimensions', {}).get('largeur_cm'),
            stade_cicatrisation=analyse.get('stade_cicatrisation', ''),
            methode_fermeture=analyse.get('methode_fermeture', ''),
            nombre_points=analyse.get('nombre_points'),
            etat_general=analyse.get('etat_general', ''),
            notes=analyse.get('notes', ''),
            confiance=analyse.get('confiance', ''),
            analyse_brute=analyse
        )

        return JsonResponse({
            'success': True,
            'evaluation_id': evaluation.id,
            'analyse': analyse
        })

    except DeepSeekOCRError as e:
        return JsonResponse({
            'error': 'Échec de l\'analyse',
            'detail': str(e)
        }, status=500)


@csrf_exempt
def analyser_progression_view(request, patient_id):
    """
    Endpoint API pour analyser la progression d'une plaie
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST requis'}, status=405)

    patient = get_object_or_404(Patient, id=patient_id)
    ocr = DjangoOCRService()

    if not ocr.is_available():
        return JsonResponse({
            'error': 'Service OCR indisponible'
        }, status=503)

    try:
        # Récupérer toutes les évaluations du patient
        evaluations = EvaluationPlaie.objects.filter(
            patient=patient
        ).order_by('analyzed_at')

        if evaluations.count() < 2:
            return JsonResponse({
                'error': 'Au moins 2 évaluations requises pour l\'analyse de progression'
            }, status=400)

        # Analyser la progression
        progression = ocr.compare_wound_progress_from_model(
            evaluations,
            return_structured=True
        )

        # Créer le rapport de progression
        rapport = ProgressionPlaie.objects.create(
            patient=patient,
            periode_analyse=progression.get('periode_analyse', ''),
            nombre_evaluations=progression.get('nombre_evaluations', 0),
            evolution_globale=progression.get('evolution_globale', ''),
            ameliorations=progression.get('ameliorations', []),
            preoccupations=progression.get('preoccupations', []),
            cicatrisation_progression=progression.get('cicatrisation_progression', ''),
            infection_evolution=progression.get('infection_evolution', ''),
            recommandations=progression.get('recommandations', []),
            prochain_controle=progression.get('prochain_controle', ''),
            notes_progression=progression.get('notes_progression', ''),
            analyse_brute=progression
        )

        # Associer les évaluations
        rapport.evaluations.set(evaluations)

        return JsonResponse({
            'success': True,
            'progression_id': rapport.id,
            'progression': progression
        })

    except DeepSeekOCRError as e:
        return JsonResponse({
            'error': 'Échec de l\'analyse de progression',
            'detail': str(e)
        }, status=500)


def progression_dashboard_view(request, patient_id):
    """
    Vue pour afficher le tableau de bord de progression
    """
    patient = get_object_or_404(Patient, id=patient_id)
    evaluations = EvaluationPlaie.objects.filter(patient=patient)
    progressions = ProgressionPlaie.objects.filter(patient=patient)

    context = {
        'patient': patient,
        'evaluations': evaluations,
        'progressions': progressions,
    }

    return render(request, 'plaies/progression_dashboard.html', context)


# Dans votre Django urls.py:

from django.urls import path
from . import views

urlpatterns = [
    path('api/patients/<int:patient_id>/analyser-plaie/',
         views.analyser_plaie_view, name='analyser_plaie'),
    path('api/patients/<int:patient_id>/analyser-progression/',
         views.analyser_progression_view, name='analyser_progression'),
    path('patients/<int:patient_id>/progression/',
         views.progression_dashboard_view, name='progression_dashboard'),
]


# Template HTML exemple (plaies/progression_dashboard.html):

{% extends "base.html" %}

{% block content %}
<div class="container">
    <h1>Suivi de Progression - {{ patient.prenom }} {{ patient.nom }}</h1>

    <div class="evaluations-section">
        <h2>Évaluations ({{ evaluations.count }})</h2>
        <div class="timeline">
            {% for eval in evaluations %}
            <div class="evaluation-card">
                <img src="{{ eval.image.url }}" alt="Plaie {{ eval.analyzed_at|date:'d/m/Y' }}">
                <div class="eval-info">
                    <h3>{{ eval.analyzed_at|date:"d/m/Y H:i" }}</h3>
                    <p><strong>Type:</strong> {{ eval.type_plaie }}</p>
                    <p><strong>Localisation:</strong> {{ eval.localisation }}</p>
                    <p><strong>Dimensions:</strong> {{ eval.longueur_cm }} x {{ eval.largeur_cm }} cm</p>
                    <p><strong>Stade:</strong> {{ eval.stade_cicatrisation }}</p>
                    <p><strong>État:</strong> {{ eval.etat_general }}</p>
                </div>
            </div>
            {% endfor %}
        </div>
    </div>

    <div class="progression-section">
        <h2>Analyses de Progression</h2>
        {% for prog in progressions %}
        <div class="progression-card">
            <h3>{{ prog.created_at|date:"d/m/Y H:i" }}</h3>
            <p><strong>Période:</strong> {{ prog.periode_analyse }}</p>
            <p><strong>Évolution:</strong> {{ prog.evolution_globale }}</p>

            {% if prog.ameliorations %}
            <div class="ameliorations">
                <h4>✅ Améliorations</h4>
                <ul>
                    {% for item in prog.ameliorations %}
                    <li>{{ item }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}

            {% if prog.preoccupations %}
            <div class="preoccupations">
                <h4>⚠️ Préoccupations</h4>
                <ul>
                    {% for item in prog.preoccupations %}
                    <li>{{ item }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}

            <div class="recommandations">
                <h4>💊 Recommandations</h4>
                <ul>
                    {% for rec in prog.recommandations %}
                    <li>{{ rec }}</li>
                    {% endfor %}
                </ul>
            </div>

            <p><strong>Prochain contrôle:</strong> {{ prog.prochain_controle }}</p>
        </div>
        {% endfor %}
    </div>

    <div class="actions">
        <form id="upload-form" method="post" enctype="multipart/form-data"
              action="{% url 'analyser_plaie' patient.id %}">
            {% csrf_token %}
            <input type="file" name="image_plaie" accept="image/*" required>
            <button type="submit">Ajouter une nouvelle évaluation</button>
        </form>

        {% if evaluations.count >= 2 %}
        <form method="post" action="{% url 'analyser_progression' patient.id %}">
            {% csrf_token %}
            <button type="submit">Générer une analyse de progression</button>
        </form>
        {% endif %}
    </div>
</div>
{% endblock %}
'''

    print(django_code)
    print("=" * 80)
    print("\nVous pouvez copier ce code pour intégrer le suivi de progression dans Django.")


def main():
    parser = argparse.ArgumentParser(
        description='Exemples d\'analyse de progression de plaies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Exemples:
  %(prog)s wound_day1.jpg:2025-01-01 wound_day7.jpg:2025-01-07 wound_day14.jpg:2025-01-14
  %(prog)s --url https://nunoocrapi.opefitoo.com img1.jpg:2025-01-01 img2.jpg:2025-01-08
  %(prog)s --django-example
        '''
    )

    parser.add_argument('images', nargs='*',
                       help='Images avec dates (format: image.jpg:YYYY-MM-DD)')
    parser.add_argument('--url', default='http://localhost:8765',
                       help='URL du service OCR (défaut: http://localhost:8765)')
    parser.add_argument('--django-example', action='store_true',
                       help='Afficher l\'exemple d\'intégration Django')

    args = parser.parse_args()

    # Show Django example if requested
    if args.django_example:
        example_django_integration()
        return 0

    # Require at least 2 images
    if len(args.images) < 2:
        parser.print_help()
        print("\nErreur: Au moins 2 images avec dates sont requises")
        print("Format: image.jpg:YYYY-MM-DD")
        print("\nExemple:")
        print("  python wound_progression_example.py wound1.jpg:2025-01-01 wound2.jpg:2025-01-07")
        return 1

    # Parse image arguments
    try:
        image_data_list = [parse_image_arg(arg) for arg in args.images]
    except ValueError as e:
        print(f"❌ Erreur: {e}")
        return 1

    print("\n🏥 Analyse de Progression de Plaies - DeepSeek-OCR")
    print(f"Service URL: {args.url}")
    print(f"Nombre d'images: {len(image_data_list)}\n")

    # Run progression analysis
    success = example_progression_analysis(image_data_list, args.url)

    if success:
        example_django_integration()

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
