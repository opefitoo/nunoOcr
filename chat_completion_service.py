#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chat Completion Service for DeepSeek LLM

A generic chat completion service that can be used for various text processing tasks
including report cleanup, summarization, and text improvement.

This service provides an OpenAI-compatible API for text-based tasks (no images required).
"""
import os
import logging
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-V2.5")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8766"))  # Different port from OCR service

# Initialize FastAPI app
app = FastAPI(
    title="DeepSeek Chat Completion API",
    description="OpenAI-compatible Chat Completion API for text processing tasks",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
llm = None


# =============================================================================
# Request/Response Models
# =============================================================================

class ChatMessage(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.1
    max_tokens: Optional[int] = 2000
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = False


class ReportCleanupRequest(BaseModel):
    """Specialized request for cleaning up nursing reports."""
    text: str
    preserve_content: bool = True
    language: str = "fr"


class ReportCleanupResponse(BaseModel):
    """Response for report cleanup."""
    original_text: str
    cleaned_text: str
    prompt_used: str
    changes_made: bool
    model: str


class TextSummaryRequest(BaseModel):
    """Request for text summarization."""
    text: str
    max_length: int = 100
    language: str = "fr"


class EventReport(BaseModel):
    """A single event report with date."""
    date: str
    report: str


class PatientMedicalSummaryRequest(BaseModel):
    """Request for generating a comprehensive medical summary from patient event reports."""
    patient_id: int
    patient_name: str
    events: List[EventReport]
    max_summary_length: int = 4000
    language: str = "fr"


class KeyMedicalFacts(BaseModel):
    """Structured key medical facts extracted from reports."""
    allergies: List[str] = []
    chronic_conditions: List[str] = []
    current_treatments: List[str] = []
    hospitalizations: List[str] = []
    important_events: List[str] = []
    mobility_status: Optional[str] = None
    cognitive_status: Optional[str] = None
    skin_conditions: List[str] = []
    vital_signs_alerts: List[str] = []
    social_context: Optional[str] = None


class PatientMedicalSummaryResponse(BaseModel):
    """Response containing the generated medical summary."""
    patient_id: int
    patient_name: str
    summary: str
    key_facts: KeyMedicalFacts
    events_analyzed: int
    date_range: Dict[str, str]
    generated_at: str
    model: str


# =============================================================================
# Prompts
# =============================================================================

REPORT_CLEANUP_SYSTEM_PROMPT_FR = """Tu es un assistant spécialisé dans la correction de rapports de soins infirmiers en français.

Ton rôle:
1. Corriger les fautes d'orthographe et de grammaire
2. Corriger la ponctuation
3. Standardiser les abréviations médicales courantes (TA, FC, SpO2, etc.)
4. Garder un style professionnel et concis

Règles STRICTES:
- NE JAMAIS ajouter d'informations qui ne sont pas dans le texte original
- NE JAMAIS supprimer d'informations importantes
- NE JAMAIS changer le sens du texte
- Garder le sens exact du texte original
- Utiliser les abréviations médicales standard
- Retourner UNIQUEMENT le texte corrigé, sans explication ni commentaire"""

REPORT_CLEANUP_USER_PROMPT_FR = """Corrige ce rapport de soin infirmier. Retourne uniquement le texte corrigé:

{text}"""

SUMMARY_SYSTEM_PROMPT_FR = """Tu es un assistant médical. Résume les rapports de soins de manière concise et professionnelle.
Ne mentionne que les actions principales et observations importantes."""

SUMMARY_USER_PROMPT_FR = """Résume ce rapport de soin en maximum {max_length} caractères:

{text}"""


# Patient Medical Summary Prompts
MEDICAL_SUMMARY_SYSTEM_PROMPT_FR = """Tu es un assistant médical spécialisé dans l'analyse de dossiers de soins infirmiers.

Ton rôle est d'analyser une série de rapports de soins d'un patient et d'en extraire un résumé médical structuré.

Tu dois identifier et extraire:
1. **Allergies** détectées ou mentionnées
2. **Pathologies chroniques** (diabète, HTA, insuffisance cardiaque, etc.)
3. **Traitements en cours** (médicaments, pansements, soins réguliers)
4. **Hospitalisations** et événements médicaux majeurs
5. **Événements importants** (chutes, incidents, changements d'état)
6. **État de mobilité** (autonome, aide technique, fauteuil, alité)
7. **État cognitif** (orienté, confus, démence, etc.)
8. **État cutané** (escarres, plaies, ulcères)
9. **Alertes constantes vitales** (HTA non contrôlée, hypoglycémies, etc.)
10. **Contexte social** (vit seul, aidant naturel, isolement)

RÈGLES IMPORTANTES:
- Extraire UNIQUEMENT les informations présentes dans les rapports
- Prioriser les informations les plus récentes
- Ignorer les détails de routine (soins quotidiens normaux)
- Mettre en avant les changements d'état et les alertes
- Être concis et factuel
- Répondre en JSON valide"""


MEDICAL_SUMMARY_USER_PROMPT_FR = """Analyse ces {event_count} rapports de soins du patient "{patient_name}" (du {date_start} au {date_end}).

Génère un résumé médical structuré en JSON avec ce format exact:
{{
    "summary": "Résumé narratif de l'état du patient et de son évolution (max 2000 caractères)",
    "key_facts": {{
        "allergies": ["liste des allergies détectées"],
        "chronic_conditions": ["liste des pathologies chroniques"],
        "current_treatments": ["traitements en cours"],
        "hospitalizations": ["hospitalisations avec dates si disponibles"],
        "important_events": ["événements marquants: chutes, incidents, changements d'état"],
        "mobility_status": "description de l'état de mobilité ou null",
        "cognitive_status": "description de l'état cognitif ou null",
        "skin_conditions": ["problèmes cutanés: escarres, plaies"],
        "vital_signs_alerts": ["alertes constantes: HTA, hypoglycémie"],
        "social_context": "contexte social ou null"
    }}
}}

RAPPORTS DE SOINS:
{reports}

Réponds UNIQUEMENT avec le JSON, sans texte avant ou après."""


# =============================================================================
# API Endpoints
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    global llm

    logger.info(f"Loading model: {MODEL_NAME}")
    logger.info("This may take several minutes on first run...")

    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
        )

        logger.info(f"Model loaded successfully: {MODEL_NAME}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.warning("Server will start but inference will fail until model is loaded")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if llm is None:
        return {"status": "initializing", "model": MODEL_NAME}
    return {"status": "ok", "model": MODEL_NAME}


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": 1234567890,
                "owned_by": "deepseek-ai"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.
    """
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please wait and try again."
        )

    try:
        from vllm import SamplingParams

        # Build prompt from messages
        prompt_parts = []
        for msg in request.messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")

        prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )

        # Generate response
        logger.info(f"Generating response for prompt length: {len(prompt)}")
        outputs = llm.generate([prompt], sampling_params)

        # Extract generated text
        generated_text = outputs[0].outputs[0].text.strip()

        # Format OpenAI-compatible response
        response = {
            "id": f"chatcmpl-{os.urandom(12).hex()}",
            "object": "chat.completion",
            "created": int(__import__('time').time()),
            "model": MODEL_NAME,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_text
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(prompt.split()) + len(generated_text.split())
            }
        }

        return response

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/v1/report/cleanup", response_model=ReportCleanupResponse)
async def cleanup_report(request: ReportCleanupRequest):
    """
    Specialized endpoint for cleaning up nursing reports.
    Returns the original text, cleaned text, and the prompt used.
    """
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please wait and try again."
        )

    if not request.text or not request.text.strip():
        return ReportCleanupResponse(
            original_text=request.text,
            cleaned_text=request.text,
            prompt_used="",
            changes_made=False,
            model=MODEL_NAME
        )

    try:
        from vllm import SamplingParams

        # Build prompts
        system_prompt = REPORT_CLEANUP_SYSTEM_PROMPT_FR
        user_prompt = REPORT_CLEANUP_USER_PROMPT_FR.format(text=request.text)

        # Full prompt for transparency
        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

        # Sampling parameters - low temperature for consistency
        sampling_params = SamplingParams(
            temperature=0.1,
            max_tokens=len(request.text) * 2 + 100,
            top_p=1.0,
        )

        # Generate response
        outputs = llm.generate([full_prompt], sampling_params)
        cleaned_text = outputs[0].outputs[0].text.strip()

        # Remove quotes if model added them
        if cleaned_text.startswith('"') and cleaned_text.endswith('"'):
            cleaned_text = cleaned_text[1:-1]
        if cleaned_text.startswith("'") and cleaned_text.endswith("'"):
            cleaned_text = cleaned_text[1:-1]

        return ReportCleanupResponse(
            original_text=request.text,
            cleaned_text=cleaned_text,
            prompt_used=full_prompt,
            changes_made=(cleaned_text != request.text),
            model=MODEL_NAME
        )

    except Exception as e:
        logger.error(f"Report cleanup error: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@app.post("/v1/report/cleanup/preview")
async def cleanup_report_preview(request: ReportCleanupRequest):
    """
    Preview endpoint - returns the prompt that would be used without executing.
    Useful for debugging and transparency.
    """
    system_prompt = REPORT_CLEANUP_SYSTEM_PROMPT_FR
    user_prompt = REPORT_CLEANUP_USER_PROMPT_FR.format(text=request.text)

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "full_prompt": f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:",
        "original_text": request.text,
        "model": MODEL_NAME
    }


@app.post("/v1/report/summarize")
async def summarize_report(request: TextSummaryRequest):
    """
    Summarize a nursing report.
    """
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please wait and try again."
        )

    if not request.text or not request.text.strip():
        return {"summary": "", "original_length": 0, "summary_length": 0}

    try:
        from vllm import SamplingParams

        system_prompt = SUMMARY_SYSTEM_PROMPT_FR
        user_prompt = SUMMARY_USER_PROMPT_FR.format(
            text=request.text,
            max_length=request.max_length
        )

        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

        sampling_params = SamplingParams(
            temperature=0.3,
            max_tokens=request.max_length + 50,
            top_p=1.0,
        )

        outputs = llm.generate([full_prompt], sampling_params)
        summary = outputs[0].outputs[0].text.strip()

        # Truncate if still too long
        if len(summary) > request.max_length:
            summary = summary[:request.max_length - 3] + "..."

        return {
            "summary": summary,
            "original_length": len(request.text),
            "summary_length": len(summary),
            "model": MODEL_NAME
        }

    except Exception as e:
        logger.error(f"Summarization error: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@app.post("/v1/patient/medical-summary", response_model=PatientMedicalSummaryResponse)
async def generate_patient_medical_summary(request: PatientMedicalSummaryRequest):
    """
    Generate a comprehensive medical summary from patient event reports.

    This endpoint analyzes multiple event reports and extracts:
    - Key medical facts (allergies, conditions, treatments)
    - Important events and changes in patient status
    - A narrative summary of the patient's medical history

    Designed for overnight batch processing of patient records.
    """
    import json
    from datetime import datetime

    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please wait and try again."
        )

    if not request.events:
        return PatientMedicalSummaryResponse(
            patient_id=request.patient_id,
            patient_name=request.patient_name,
            summary="Aucun rapport de soin à analyser.",
            key_facts=KeyMedicalFacts(),
            events_analyzed=0,
            date_range={"start": "", "end": ""},
            generated_at=datetime.now().isoformat(),
            model=MODEL_NAME
        )

    try:
        from vllm import SamplingParams

        # Sort events by date
        sorted_events = sorted(request.events, key=lambda x: x.date)
        date_start = sorted_events[0].date
        date_end = sorted_events[-1].date

        # Format reports for the prompt
        # Limit to most recent reports if too many (to fit in context)
        max_reports = 200  # Adjust based on model context size
        events_to_analyze = sorted_events[-max_reports:] if len(sorted_events) > max_reports else sorted_events

        reports_text = "\n\n".join([
            f"[{event.date}] {event.report}"
            for event in events_to_analyze
            if event.report and event.report.strip()
        ])

        # Build prompts
        system_prompt = MEDICAL_SUMMARY_SYSTEM_PROMPT_FR
        user_prompt = MEDICAL_SUMMARY_USER_PROMPT_FR.format(
            event_count=len(events_to_analyze),
            patient_name=request.patient_name,
            date_start=date_start,
            date_end=date_end,
            reports=reports_text
        )

        full_prompt = f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=0.2,  # Low temperature for consistency
            max_tokens=request.max_summary_length,
            top_p=0.95,
        )

        # Generate response
        logger.info(f"Generating medical summary for patient {request.patient_id} ({len(events_to_analyze)} events)")
        outputs = llm.generate([full_prompt], sampling_params)
        response_text = outputs[0].outputs[0].text.strip()

        # Parse JSON response
        try:
            # Clean up response - remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            result = json.loads(response_text.strip())

            key_facts = KeyMedicalFacts(
                allergies=result.get("key_facts", {}).get("allergies", []) or [],
                chronic_conditions=result.get("key_facts", {}).get("chronic_conditions", []) or [],
                current_treatments=result.get("key_facts", {}).get("current_treatments", []) or [],
                hospitalizations=result.get("key_facts", {}).get("hospitalizations", []) or [],
                important_events=result.get("key_facts", {}).get("important_events", []) or [],
                mobility_status=result.get("key_facts", {}).get("mobility_status"),
                cognitive_status=result.get("key_facts", {}).get("cognitive_status"),
                skin_conditions=result.get("key_facts", {}).get("skin_conditions", []) or [],
                vital_signs_alerts=result.get("key_facts", {}).get("vital_signs_alerts", []) or [],
                social_context=result.get("key_facts", {}).get("social_context")
            )

            summary = result.get("summary", "Résumé non disponible.")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response, using raw text: {e}")
            # Fallback: use raw response as summary
            summary = response_text
            key_facts = KeyMedicalFacts()

        return PatientMedicalSummaryResponse(
            patient_id=request.patient_id,
            patient_name=request.patient_name,
            summary=summary,
            key_facts=key_facts,
            events_analyzed=len(events_to_analyze),
            date_range={"start": date_start, "end": date_end},
            generated_at=datetime.now().isoformat(),
            model=MODEL_NAME
        )

    except Exception as e:
        logger.error(f"Medical summary generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    logger.info(f"Starting DeepSeek Chat Completion server on {HOST}:{PORT}")
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
