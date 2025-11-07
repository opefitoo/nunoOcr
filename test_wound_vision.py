#!/usr/bin/env python3
"""
Test wound analysis with GPT-4 Vision or Claude Vision

This script tests the wound analysis using actual vision models (GPT-4V/Claude)
instead of the DeepSeek-OCR text extraction model.

Usage:
    export OPENAI_API_KEY="sk-..."
    python test_wound_vision.py wound.jpg

    # Or with Claude
    export ANTHROPIC_API_KEY="sk-ant-..."
    python test_wound_vision.py wound.jpg --provider anthropic
"""

import sys
import os
import argparse
from pathlib import Path
from nunoocr_client import DeepSeekOCRClient, DeepSeekOCRError
import json


def test_wound_vision(image_path: str, provider: str = 'openai'):
    """Test wound analysis with vision API"""

    # Get API key from environment
    if provider == 'openai':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ Erreur: OPENAI_API_KEY non définie")
            print("   Définissez-la avec: export OPENAI_API_KEY='sk-...'")
            return False
    elif provider == 'anthropic':
        api_key = os.getenv('ANTHROPIC_API_KEY')
        if not api_key:
            print("❌ Erreur: ANTHROPIC_API_KEY non définie")
            print("   Définissez-la avec: export ANTHROPIC_API_KEY='sk-ant-...'")
            return False
    else:
        print(f"❌ Provider inconnu: {provider}")
        return False

    print("=" * 80)
    print(f"Test d'Analyse de Plaie avec {provider.upper()}")
    print("=" * 80)
    print(f"Image: {Path(image_path).name}")
    print(f"Provider: {provider}")
    print(f"Taille: {Path(image_path).stat().st_size / 1024:.1f} KB\n")

    try:
        # Initialize client with vision API
        client = DeepSeekOCRClient(
            vision_api_key=api_key,
            vision_provider=provider,
            timeout=120
        )

        print("Analyse en cours...")
        print("Cela peut prendre 10-30 secondes...\n")

        import time
        start_time = time.time()

        with open(image_path, 'rb') as f:
            result = client.analyze_wound(f, return_structured=True)

        elapsed = time.time() - start_time

        print(f"✅ Analyse terminée en {elapsed:.1f} secondes!\n")
        print("=" * 80)
        print("RÉSULTATS DE L'ANALYSE")
        print("=" * 80)

        if result.get('structured', True):
            # Display structured results
            print(f"\n🔍 TYPE DE PLAIE: {result.get('type_plaie', 'N/A')}")
            print(f"📍 LOCALISATION: {result.get('localisation', 'N/A')}")

            dimensions = result.get('dimensions', {})
            if dimensions:
                length = dimensions.get('longueur_cm', 'N/A')
                width = dimensions.get('largeur_cm', 'N/A')
                print(f"📏 DIMENSIONS: {length} x {width} cm")

            print(f"\n🏥 STADE DE CICATRISATION: {result.get('stade_cicatrisation', 'N/A')}")
            print(f"🔗 MÉTHODE DE FERMETURE: {result.get('methode_fermeture', 'N/A')}")

            closure_count = result.get('nombre_points')
            if closure_count:
                print(f"🔢 NOMBRE DE POINTS: {closure_count}")

            # Infection signs
            infections = result.get('signes_infection', [])
            print(f"\n⚠️  SIGNES D'INFECTION: ", end="")
            if infections:
                print(", ".join(infections))
            else:
                print("Aucun détecté")

            # Complications
            complications = result.get('complications', [])
            print(f"⚠️  COMPLICATIONS: ", end="")
            if complications:
                print(", ".join(complications))
            else:
                print("Aucune détectée")

            print(f"\n📊 ÉTAT GÉNÉRAL: {result.get('etat_general', 'N/A')}")
            print(f"🎯 CONFIANCE: {result.get('confiance', 'N/A')}")

            notes = result.get('notes', '')
            if notes:
                print(f"\n📝 NOTES SUPPLÉMENTAIRES:")
                print(f"   {notes}")

            print("\n" + "=" * 80)
            print("RÉPONSE JSON COMPLÈTE")
            print("=" * 80)
            display_data = {k: v for k, v in result.items() if k != '_metadata'}
            print(json.dumps(display_data, indent=2, ensure_ascii=False))

        else:
            print("\n📄 ANALYSE TEXTUELLE:")
            print(result.get('analysis', 'Aucune analyse disponible'))

        print("\n" + "=" * 80)

        # Metadata
        metadata = result.get('_metadata', {})
        print(f"Provider: {metadata.get('provider', 'N/A')}")
        print(f"Modèle: {metadata.get('model', 'N/A')}")
        print(f"Tokens utilisés: {metadata.get('tokens_used', 'N/A')}")
        print(f"Temps: {elapsed:.1f}s")
        print("=" * 80)

        return True

    except FileNotFoundError:
        print(f"❌ Erreur: Image introuvable: {image_path}")
        return False
    except DeepSeekOCRError as e:
        print(f"❌ Erreur d'analyse: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test d\'analyse de plaie avec GPT-4 Vision ou Claude',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Exemples:
  # Avec OpenAI GPT-4 Vision
  export OPENAI_API_KEY="sk-..."
  %(prog)s wound.jpg

  # Avec Claude Vision
  export ANTHROPIC_API_KEY="sk-ant-..."
  %(prog)s wound.jpg --provider anthropic

  # Analyse non-structurée (texte libre)
  %(prog)s wound.jpg --unstructured
        '''
    )

    parser.add_argument('image', help='Chemin vers l\'image de plaie')
    parser.add_argument('--provider', choices=['openai', 'anthropic'],
                       default='openai',
                       help='Provider API (défaut: openai)')
    parser.add_argument('--unstructured', action='store_true',
                       help='Obtenir une analyse textuelle au lieu de JSON')

    args = parser.parse_args()

    # Validate image path
    if not Path(args.image).exists():
        print(f"❌ Erreur: Image introuvable: {args.image}")
        return 1

    print("\n🏥 Test d'Analyse de Plaie avec Vision API")
    print(f"Provider: {args.provider.upper()}\n")

    success = test_wound_vision(args.image, args.provider)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
