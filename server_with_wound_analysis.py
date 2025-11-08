#!/usr/bin/env python3
"""
DeepSeek-OCR + Wound Analysis FastAPI server
OpenAI-compatible API for:
- OCR inference (DeepSeek-OCR for prescriptions)
- Wound analysis (GPT-4 Vision or Claude Vision)

Security:
- Service API Key required for wound analysis endpoints
- IP whitelist support
"""

import os
import logging
import base64
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Vision API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VISION_PROVIDER = os.getenv("VISION_PROVIDER", "openai")  # openai or anthropic

# Security configuration
SERVICE_API_KEY = os.getenv("SERVICE_API_KEY")  # API Key for service-to-service auth
ALLOWED_IPS = os.getenv("ALLOWED_IPS", "").split(",") if os.getenv("ALLOWED_IPS") else []
ALLOWED_DOMAINS = os.getenv("ALLOWED_DOMAINS", "").split(",") if os.getenv("ALLOWED_DOMAINS") else []

# Initialize FastAPI app
app = FastAPI(
    title="nunoOcr API",
    description="Unified API for OCR (prescriptions) and Wound Analysis",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (will be initialized on startup)
llm = None


class ChatMessage(BaseModel):
    role: str
    content: str | list


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 2000
    top_p: Optional[float] = 1.0


class WoundAnalysisResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize the vLLM model on startup."""
    global llm

    logger.info(f"Loading model: {MODEL_NAME}")
    logger.info("This may take several minutes on first run (downloading model)...")

    try:
        from vllm import LLM, SamplingParams

        # Initialize vLLM with DeepSeek-OCR
        llm = LLM(
            model=MODEL_NAME,
            trust_remote_code=True,
            max_model_len=8192,
            gpu_memory_utilization=0.9,
        )

        logger.info(f"Model loaded successfully: {MODEL_NAME}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Server will start but OCR inference will fail until model is loaded")


def verify_service_api_key(authorization: Optional[str] = Header(None)) -> bool:
    """
    Verify service API key for protected endpoints.

    Returns:
        True if authorized, raises HTTPException otherwise
    """
    # If no SERVICE_API_KEY is configured, allow all (backward compatibility)
    if not SERVICE_API_KEY:
        logger.warning("SERVICE_API_KEY not configured - endpoints are not protected!")
        return True

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Authorization required",
                "message": "Service API Key required. Set 'Authorization: Bearer YOUR_SERVICE_KEY'"
            }
        )

    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Invalid authorization format",
                "message": "Use 'Authorization: Bearer YOUR_SERVICE_KEY'"
            }
        )

    provided_key = authorization[7:]  # Remove "Bearer "

    if provided_key != SERVICE_API_KEY:
        logger.warning(f"Invalid service API key attempt")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Invalid service API key",
                "message": "The provided service API key is incorrect"
            }
        )

    return True


def verify_ip_whitelist(request: Request) -> bool:
    """
    Verify client IP is in whitelist if configured.

    Returns:
        True if authorized, raises HTTPException otherwise
    """
    if not ALLOWED_IPS:
        return True  # No whitelist configured

    client_ip = request.client.host

    # Check if IP is in whitelist
    if client_ip not in ALLOWED_IPS:
        logger.warning(f"Rejected request from non-whitelisted IP: {client_ip}")
        raise HTTPException(
            status_code=403,
            detail={
                "error": "IP not allowed",
                "message": f"Your IP ({client_ip}) is not authorized to access this service"
            }
        )

    return True


@app.get("/health")
async def health_check():
    """Health check endpoint (public - no auth required)."""
    vision_configured = OPENAI_API_KEY is not None
    service_protected = SERVICE_API_KEY is not None
    ip_whitelist_enabled = len(ALLOWED_IPS) > 0

    return {
        "status": "ok" if llm else "initializing",
        "ocr_model": MODEL_NAME,
        "ocr_ready": llm is not None,
        "vision_provider": VISION_PROVIDER,
        "vision_configured": vision_configured,
        "security": {
            "service_api_key_required": service_protected,
            "ip_whitelist_enabled": ip_whitelist_enabled,
            "allowed_ips_count": len(ALLOWED_IPS) if ip_whitelist_enabled else 0
        }
    }


@app.get("/v1/models")
async def list_models():
    """List available models."""
    models = [
        {
            "id": MODEL_NAME,
            "object": "model",
            "created": 1234567890,
            "owned_by": "deepseek-ai",
            "purpose": "OCR for prescriptions"
        }
    ]

    if OPENAI_API_KEY:
        models.append({
            "id": f"{VISION_PROVIDER}-vision",
            "object": "model",
            "created": 1234567890,
            "owned_by": VISION_PROVIDER,
            "purpose": "Wound analysis"
        })

    return {
        "object": "list",
        "data": models
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint for OCR (prescriptions).
    """
    if llm is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please wait and try again."
        )

    try:
        from vllm import SamplingParams

        # Extract messages and prepare prompt
        messages = request.messages
        prompt_parts = []

        for msg in messages:
            role = msg.role
            content = msg.content

            if isinstance(content, str):
                prompt_parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "input_text":
                            prompt_parts.append(f"{role}: {item.get('text', '')}")

        prompt = "\n".join(prompt_parts)

        # Sampling parameters
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
        )

        # Generate response
        logger.info(f"Generating OCR response for prompt length: {len(prompt)}")
        outputs = llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text

        # Format OpenAI-compatible response
        response = {
            "id": f"chatcmpl-{os.urandom(12).hex()}",
            "object": "chat.completion",
            "created": 1234567890,
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


def _image_to_base64(file_content: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(file_content).decode('utf-8')


def _analyze_with_openai_vision(image_base64: str) -> Dict[str, Any]:
    """Analyze wound using OpenAI GPT-4 Vision."""
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured"
        )

    url = "https://api.openai.com/v1/chat/completions"

    system_prompt = (
        "Vous êtes un assistant d'aide à la documentation médicale pour professionnels de santé. "
        "Vous aidez à documenter et décrire des plaies pour le suivi médical. "
        "Ceci est utilisé par des médecins et infirmières qualifiés dans un contexte professionnel. "
        "Retournez UNIQUEMENT un objet JSON EN FRANÇAIS avec une description objective de la plaie visible."
    )

    user_prompt = (
        "Je suis un professionnel de santé documentant cette plaie pour le dossier patient. "
        "Analysez cette image et retournez un JSON avec les champs suivants (EN FRANÇAIS):\n"
        "{\n"
        '  "type_plaie": "type de plaie observée",\n'
        '  "localisation": "partie du corps visible",\n'
        '  "dimensions": {"longueur_cm": X, "largeur_cm": Y, "profondeur_cm": Z},\n'
        '  "stade_cicatrisation": "description du stade",\n'
        '  "methode_fermeture": "points de suture/agrafes/steristrips/non applicable",\n'
        '  "nombre_points": nombre ou null,\n'
        '  "signes_infection": ["liste", "des", "signes"],\n'
        '  "complications": ["liste"],\n'
        '  "etat_general": "description générale",\n'
        '  "confiance": "élevée/moyenne/faible",\n'
        '  "notes": "observations supplémentaires"\n'
        "}\n\n"
        "Restez factuel et objectif. Si un champ n'est pas visible, mettez null ou une liste vide."
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000,
        "temperature": 0.0
    }

    logger.info("Calling OpenAI GPT-4 Vision API")
    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
        raise HTTPException(
            status_code=response.status_code,
            detail=f"OpenAI API error: {response.text}"
        )

    result = response.json()
    content = result['choices'][0]['message']['content']

    # Try to parse JSON from response
    import json
    try:
        # Extract JSON from markdown code blocks if present
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()

        data = json.loads(content)
        return data
    except json.JSONDecodeError:
        # Return raw text if not valid JSON
        return {"raw_response": content}


def _analyze_with_claude_vision(image_base64: str) -> Dict[str, Any]:
    """Analyze wound using Anthropic Claude Vision."""
    if not OPENAI_API_KEY:  # Using same env var for simplicity
        raise HTTPException(
            status_code=503,
            detail="Anthropic API key not configured"
        )

    url = "https://api.anthropic.com/v1/messages"

    user_prompt = (
        "Je suis un professionnel de santé documentant cette plaie pour le dossier patient. "
        "Analysez cette image et retournez un JSON avec les champs suivants (EN FRANÇAIS):\n"
        "{\n"
        '  "type_plaie": "type de plaie observée",\n'
        '  "localisation": "partie du corps visible",\n'
        '  "dimensions": {"longueur_cm": X, "largeur_cm": Y},\n'
        '  "stade_cicatrisation": "description du stade",\n'
        '  "signes_infection": ["liste"],\n'
        '  "etat_general": "description",\n'
        '  "confiance": "élevée/moyenne/faible"\n'
        "}"
    )

    headers = {
        "Content-Type": "application/json",
        "x-api-key": OPENAI_API_KEY,
        "anthropic-version": "2023-06-01"
    }

    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 1000,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }
        ]
    }

    logger.info("Calling Anthropic Claude Vision API")
    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        logger.error(f"Claude API error: {response.status_code} - {response.text}")
        raise HTTPException(
            status_code=response.status_code,
            detail=f"Claude API error: {response.text}"
        )

    result = response.json()
    content = result['content'][0]['text']

    # Try to parse JSON
    import json
    try:
        if '```json' in content:
            content = content.split('```json')[1].split('```')[0].strip()
        elif '```' in content:
            content = content.split('```')[1].split('```')[0].strip()

        data = json.loads(content)
        return data
    except json.JSONDecodeError:
        return {"raw_response": content}


@app.post("/v1/analyze-wound", response_model=WoundAnalysisResponse)
async def analyze_wound(
    request: Request,
    wound_image: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    """
    Analyze wound image using vision AI (GPT-4 Vision or Claude).

    Security:
    - Requires SERVICE_API_KEY in Authorization header
    - Optional IP whitelist

    Usage from Django:
        POST http://46.224.6.193:8765/v1/analyze-wound
        Authorization: Bearer YOUR_SERVICE_API_KEY
        Content-Type: multipart/form-data

        Body:
            wound_image: <file>

    Returns:
        JSON with structured wound analysis in French
    """
    # Verify security
    verify_service_api_key(authorization)
    verify_ip_whitelist(request)

    try:
        # Read image
        image_content = await wound_image.read()

        # Validate size (5MB max)
        if len(image_content) > 5 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Image trop grande (max 5MB)"
            )

        # Convert to base64
        image_base64 = _image_to_base64(image_content)

        # Call appropriate vision API
        if VISION_PROVIDER == "openai":
            data = _analyze_with_openai_vision(image_base64)
        elif VISION_PROVIDER == "anthropic":
            data = _analyze_with_claude_vision(image_base64)
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid vision provider: {VISION_PROVIDER}"
            )

        logger.info("Wound analysis completed successfully")

        return WoundAnalysisResponse(
            success=True,
            data=data
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Wound analysis error: {e}")
        return WoundAnalysisResponse(
            success=False,
            error=str(e)
        )


@app.post("/v1/compare-wound-progress")
async def compare_wound_progress(
    request: Request,
    images: List[UploadFile] = File(...),
    dates: Optional[str] = None,  # Comma-separated dates
    authorization: Optional[str] = Header(None)
):
    """
    Compare multiple wound images over time to assess healing progress.

    Security:
    - Requires SERVICE_API_KEY in Authorization header
    - Optional IP whitelist

    Usage:
        POST http://46.224.6.193:8765/v1/compare-wound-progress
        Authorization: Bearer YOUR_SERVICE_API_KEY

        Body (multipart/form-data):
            images: file1, file2, file3
            dates: "2025-01-01,2025-01-07,2025-01-14"
    """
    # Verify security
    verify_service_api_key(authorization)
    verify_ip_whitelist(request)

    try:
        if len(images) < 2:
            raise HTTPException(
                status_code=400,
                detail="Au moins 2 images requises pour la comparaison"
            )

        # Parse dates
        date_list = dates.split(",") if dates else [f"Image {i+1}" for i in range(len(images))]

        # Analyze each image
        analyses = []
        for i, img in enumerate(images):
            content = await img.read()
            image_base64 = _image_to_base64(content)

            if VISION_PROVIDER == "openai":
                data = _analyze_with_openai_vision(image_base64)
            else:
                data = _analyze_with_claude_vision(image_base64)

            analyses.append({
                "date": date_list[i] if i < len(date_list) else f"Image {i+1}",
                "analysis": data
            })

        # Generate progression summary
        progression = {
            "total_images": len(images),
            "images": analyses,
            "progression_notes": "Comparer les stades de cicatrisation entre les dates"
        }

        return WoundAnalysisResponse(
            success=True,
            data=progression
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Progression analysis error: {e}")
        return WoundAnalysisResponse(
            success=False,
            error=str(e)
        )


if __name__ == "__main__":
    logger.info(f"Starting nunoOcr server on {HOST}:{PORT}")
    logger.info(f"OCR Model: {MODEL_NAME}")
    logger.info(f"Vision Provider: {VISION_PROVIDER}")
    logger.info(f"Vision API configured: {OPENAI_API_KEY is not None}")

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
