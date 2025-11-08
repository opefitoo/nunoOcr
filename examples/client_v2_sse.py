#!/usr/bin/env python3
"""
Test client for nunoOcr v2 API with Server-Sent Events (SSE)

This demonstrates how to consume the SSE endpoint from Python.
"""

import sys
import requests
import json
from pathlib import Path


def analyze_wound_sse(
    image_path: str,
    api_url: str = "http://46.224.6.193:8765/v2/analyze-wound",
    service_api_key: str = None
):
    """
    Analyze wound using the v2 SSE endpoint with real-time progress updates.

    Args:
        image_path: Path to wound image
        api_url: URL of the v2 API endpoint
        service_api_key: Service API key (optional if not configured)
    """
    print("=" * 80)
    print("nunoOcr v2 - Analyse de Plaie en Temps Réel (SSE)")
    print("=" * 80)
    print(f"Image: {Path(image_path).name}")
    print(f"URL: {api_url}\n")

    # Prepare headers
    headers = {}
    if service_api_key:
        headers['Authorization'] = f'Bearer {service_api_key}'

    # Prepare files
    files = {
        'wound_image': open(image_path, 'rb')
    }

    try:
        # Make request with streaming
        print("🚀 Envoi de la requête...\n")
        response = requests.post(
            api_url,
            headers=headers,
            files=files,
            stream=True  # Enable streaming
        )

        if response.status_code != 200:
            print(f"❌ Erreur HTTP {response.status_code}")
            print(response.text)
            return

        print("✅ Connexion établie, lecture du stream SSE...\n")

        # Process SSE stream
        buffer = ""
        final_result = None

        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                buffer += chunk

                # Split by double newline (SSE message separator)
                while '\n\n' in buffer:
                    message, buffer = buffer.split('\n\n', 1)

                    if not message.strip():
                        continue

                    # Parse SSE message
                    lines = message.strip().split('\n')
                    event_type = None
                    data = None

                    for line in lines:
                        if line.startswith('event: '):
                            event_type = line[7:]
                        elif line.startswith('data: '):
                            data = json.loads(line[6:])

                    if not event_type or not data:
                        continue

                    # Handle different event types
                    if event_type == 'progress':
                        percent = data.get('percent', 0)
                        message_text = data.get('message', '')
                        print(f"📊 [{percent}%] {message_text}")

                    elif event_type == 'result':
                        print("\n" + "=" * 80)
                        print("🎉 RÉSULTAT DE L'ANALYSE")
                        print("=" * 80)
                        final_result = data.get('data', {})

                        # Display structured result
                        if isinstance(final_result, dict):
                            print(f"\n🔍 TYPE DE PLAIE: {final_result.get('type_plaie', 'N/A')}")
                            print(f"📍 LOCALISATION: {final_result.get('localisation', 'N/A')}")

                            dimensions = final_result.get('dimensions', {})
                            if dimensions:
                                length = dimensions.get('longueur_cm', 'N/A')
                                width = dimensions.get('largeur_cm', 'N/A')
                                print(f"📏 DIMENSIONS: {length} x {width} cm")

                            print(f"\n🏥 STADE: {final_result.get('stade_cicatrisation', 'N/A')}")
                            print(f"🔗 FERMETURE: {final_result.get('methode_fermeture', 'N/A')}")

                            infections = final_result.get('signes_infection', [])
                            print(f"\n⚠️  SIGNES D'INFECTION: ", end="")
                            print(", ".join(infections) if infections else "Aucun détecté")

                            complications = final_result.get('complications', [])
                            print(f"⚠️  COMPLICATIONS: ", end="")
                            print(", ".join(complications) if complications else "Aucune détectée")

                            print(f"\n📊 ÉTAT: {final_result.get('etat_general', 'N/A')}")
                            print(f"🎯 CONFIANCE: {final_result.get('confiance', 'N/A')}")

                            notes = final_result.get('notes', '')
                            if notes:
                                print(f"\n📝 NOTES: {notes}")

                    elif event_type == 'complete':
                        print("\n✅ Analyse terminée!")

                    elif event_type == 'error':
                        print(f"\n❌ ERREUR: {data.get('error', 'Unknown error')}")
                        return False

        print("\n" + "=" * 80)
        print("JSON COMPLET:")
        print("=" * 80)
        print(json.dumps(final_result, indent=2, ensure_ascii=False))
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_v2_sse_client.py <image_path> [service_api_key]")
        print("\nExemple:")
        print("  python test_v2_sse_client.py wound.jpg")
        print("  python test_v2_sse_client.py wound.jpg nuno_service_xxxxx")
        sys.exit(1)

    image_path = sys.argv[1]
    service_api_key = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(image_path).exists():
        print(f"❌ Image introuvable: {image_path}")
        sys.exit(1)

    success = analyze_wound_sse(image_path, service_api_key=service_api_key)
    sys.exit(0 if success else 1)
