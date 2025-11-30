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
