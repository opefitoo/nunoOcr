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
import io
import logging
import base64
import asyncio
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import requests
import torch
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Version info
VERSION = "4.3.0"  # Unified OCR + Wound Analysis with transformers
GIT_COMMIT = os.getenv("GIT_COMMIT", "unknown")
BUILD_DATE = datetime.now().isoformat()

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

# Global model instances (transformers-based)
model = None
processor = None
device = None


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


def load_model():
    """Load the DeepSeek-OCR model using transformers."""
    global model, processor, device

    try:
        from transformers import AutoModel, AutoTokenizer
        import types

        # Monkey patch: Handle missing Flash Attention gracefully
        try:
            import transformers.models.llama.modeling_llama as llama_module
            if not hasattr(llama_module, 'LlamaFlashAttention2'):
                logger.info("Patching missing LlamaFlashAttention2 for CPU compatibility")
                class DummyFlashAttention:
                    pass
                llama_module.LlamaFlashAttention2 = DummyFlashAttention
        except Exception as e:
            logger.warning(f"Could not patch flash attention: {e}")

        # CRITICAL: Globally monkey-patch torch.Tensor.cuda() for CPU compatibility
        logger.info("Globally patching torch.Tensor.cuda() for CPU compatibility...")
        original_cuda = torch.Tensor.cuda

        def cpu_compatible_cuda(self, device_arg=None, **kwargs):
            """Redirect .cuda() calls to .cpu() when CUDA is unavailable"""
            if not torch.cuda.is_available():
                return self
            return original_cuda(self, device_arg, **kwargs)

        torch.Tensor.cuda = cpu_compatible_cuda
        logger.info("✅ torch.Tensor.cuda() globally patched")

        # Patch .bfloat16() calls for CPU compatibility
        logger.info("Patching torch.Tensor.bfloat16() for CPU compatibility...")
        original_bfloat16 = torch.Tensor.bfloat16

        def cpu_compatible_bfloat16(self):
            """Convert bfloat16() to float32() on CPU for compatibility"""
            if not torch.cuda.is_available():
                return self.float()
            return original_bfloat16(self)

        torch.Tensor.bfloat16 = cpu_compatible_bfloat16
        logger.info("✅ torch.Tensor.bfloat16() patched to use float32 on CPU")

        # Patch .to() method to prevent bfloat16 conversion on CPU
        logger.info("Patching torch.Tensor.to() to prevent bfloat16 on CPU...")
        original_to = torch.Tensor.to

        def cpu_compatible_to(self, *args, **kwargs):
            """Intercept .to() calls and replace bfloat16 with float32 on CPU"""
            if not torch.cuda.is_available():
                if args and args[0] == torch.bfloat16:
                    args = (torch.float32,) + args[1:]
                elif 'dtype' in kwargs and kwargs['dtype'] == torch.bfloat16:
                    kwargs['dtype'] = torch.float32
            return original_to(self, *args, **kwargs)

        torch.Tensor.to = cpu_compatible_to
        logger.info("✅ torch.Tensor.to() patched to prevent bfloat16 on CPU")

        # Fix transformers version incompatibility: DynamicCache API changes
        logger.info("Patching DynamicCache for API compatibility...")
        from transformers.cache_utils import DynamicCache

        # Add missing seen_tokens property
        if not hasattr(DynamicCache, 'seen_tokens'):
            @property
            def seen_tokens(self):
                """Compatibility property for older model code"""
                if hasattr(self, 'get_seq_length'):
                    return self.get_seq_length()
                if self.key_cache and len(self.key_cache) > 0 and self.key_cache[0] is not None:
                    return self.key_cache[0].shape[2]
                return 0

            DynamicCache.seen_tokens = seen_tokens
            logger.info("✅ DynamicCache.seen_tokens property added")

        # Add missing get_max_length method
        if not hasattr(DynamicCache, 'get_max_length'):
            def get_max_length(self):
                """Return maximum cache length (None means unlimited)"""
                return None

            DynamicCache.get_max_length = get_max_length
            logger.info("✅ DynamicCache.get_max_length() method added")

        # Add missing get_usable_length method
        if not hasattr(DynamicCache, 'get_usable_length'):
            def get_usable_length(self, seq_length=None):
                """Return usable cache length for the current sequence"""
                if hasattr(self, 'get_seq_length'):
                    return self.get_seq_length()
                if self.key_cache and len(self.key_cache) > 0 and self.key_cache[0] is not None:
                    return self.key_cache[0].shape[2]
                return 0

            DynamicCache.get_usable_length = get_usable_length
            logger.info("✅ DynamicCache.get_usable_length() method added")

        logger.info(f"Loading model: {MODEL_NAME}")
        logger.info("This may take several minutes on first run (downloading model)...")

        # Detect device (CPU or GPU if available)
        if torch.cuda.is_available():
            device = "cuda"
            torch_dtype = torch.float16
            logger.info("GPU detected, using CUDA")
        else:
            device = "cpu"
            torch_dtype = torch.float32
            logger.info("No GPU detected, using CPU (CX53 32GB server)")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        processor = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        # Load model
        logger.info("Loading model weights...")
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        )

        model.eval()

        # Force all model parameters to float32 on CPU
        if device == "cpu":
            logger.info("Converting all model parameters to float32 for CPU compatibility...")
            model = model.float()
            for name, buffer in model.named_buffers():
                if buffer.dtype == torch.bfloat16:
                    buffer.data = buffer.data.float()
            logger.info("✅ All model weights converted to float32")

            logger.info("Setting global default dtype to float32 for CPU...")
            torch.set_default_dtype(torch.float32)
            logger.info("✅ Global default dtype set to float32")

        logger.info(f"✅ Model loaded successfully on {device}")
        logger.info(f"   Model: {MODEL_NAME}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Dtype: {torch_dtype}")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error("Server will start but OCR inference will fail until model is loaded")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    logger.info(f"Starting nunoOcr server on {HOST}:{PORT}")
    load_model()


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
        "status": "ok" if model else "initializing",
        "ocr_model": MODEL_NAME,
        "ocr_ready": model is not None,
        "ocr_device": device if device else "unknown",
        "ocr_engine": "transformers",
        "vision_provider": VISION_PROVIDER,
        "vision_configured": vision_configured,
        "version": VERSION,
        "git_commit": GIT_COMMIT,
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


def extract_image_from_content(content: List[Dict]) -> Optional[Image.Image]:
    """Extract PIL Image from message content."""
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "input_image":
                image_url = item.get("image_url", "")

                # Handle base64 data URI
                if image_url.startswith("data:image"):
                    try:
                        # Extract base64 data
                        base64_data = image_url.split(",")[1]
                        image_bytes = base64.b64decode(base64_data)
                        image = Image.open(io.BytesIO(image_bytes))
                        return image.convert("RGB")
                    except Exception as e:
                        logger.error(f"Failed to decode image: {e}")

    return None


def extract_text_prompt(content: List[Dict]) -> str:
    """Extract text prompt from message content."""
    text_parts = []

    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "input_text":
                text_parts.append(item.get("text", ""))

    return " ".join(text_parts)


def run_ocr_inference(image: Image.Image, full_prompt: str) -> str:
    """
    Run OCR inference synchronously (blocking).

    This function is designed to be called from a thread pool to avoid blocking
    the main event loop.
    """
    import tempfile
    import shutil
    import sys
    from io import StringIO

    # Initialize paths for cleanup
    temp_image_path = None
    temp_output_dir = None

    try:
        # Create temp file for input image
        tmp_img_fd, temp_image_path = tempfile.mkstemp(suffix='.png', dir='/tmp')
        os.close(tmp_img_fd)
        image.save(temp_image_path)
        logger.info(f"Saved image to: {temp_image_path}")

        # Create temp directory for output
        temp_output_dir = tempfile.mkdtemp(dir='/tmp')
        logger.info(f"Created temp output directory: {temp_output_dir}")

        # Capture stdout since model.infer() prints but doesn't return the text
        logger.info("Calling CPU-patched model.infer()...")
        captured_output = StringIO()
        original_stdout = sys.stdout

        try:
            # Redirect stdout to capture printed output
            sys.stdout = captured_output

            generated_text = model.infer(
                tokenizer=processor,
                prompt=full_prompt,
                image_file=temp_image_path,
                output_path=temp_output_dir,
                save_results=False,
                eval_mode=False
            )
        finally:
            # Restore original stdout
            sys.stdout = original_stdout

        # Get captured text
        captured_text = captured_output.getvalue()

        # Clean the captured text - remove debug output
        def clean_ocr_output(text: str) -> str:
            """Remove debug lines from OCR output."""
            lines = text.split('\n')
            cleaned_lines = []

            for line in lines:
                # Skip debug lines
                if any(debug_marker in line for debug_marker in [
                    'BASE:', 'PATCHES:', 'torch.Size', '===========',
                    'UserWarning', 'FutureWarning', 'DeprecationWarning',
                    '/usr/local/lib/python', 'warnings.warn'
                ]):
                    continue

                # Skip empty lines at the start
                if not cleaned_lines and not line.strip():
                    continue

                cleaned_lines.append(line)

            return '\n'.join(cleaned_lines).strip()

        # Use captured stdout if model.infer() returned None
        if generated_text is None or not generated_text.strip():
            generated_text = clean_ocr_output(captured_text)
            logger.info(f"✅ OCR completed - captured from stdout (length: {len(generated_text)} chars)")
        else:
            logger.info(f"✅ OCR completed - returned by model (length: {len(generated_text)} chars)")

        if not generated_text:
            raise ValueError("No OCR output generated - both return value and stdout are empty")

        return generated_text

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
        # Clean up temp files
        if temp_image_path and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
            logger.info(f"Cleaned up temp image: {temp_image_path}")
        if temp_output_dir and os.path.exists(temp_output_dir):
            shutil.rmtree(temp_output_dir)
            logger.info(f"Cleaned up temp output dir: {temp_output_dir}")


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with vision support for OCR (prescriptions).
    """
    if model is None or processor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet. Please wait and try again."
        )

    try:
        # Extract messages
        messages = request.messages

        # Find image and text from messages
        image = None
        system_prompt = ""
        user_prompt = ""

        for msg in messages:
            role = msg.role
            content = msg.content

            if role == "system":
                system_prompt = content if isinstance(content, str) else ""

            elif role == "user":
                if isinstance(content, list):
                    # Extract image
                    extracted_image = extract_image_from_content(content)
                    if extracted_image:
                        image = extracted_image

                    # Extract text prompt
                    user_prompt = extract_text_prompt(content)
                elif isinstance(content, str):
                    user_prompt = content

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="No image found in request. Please provide an image."
            )

        # Combine prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}".strip()
        if not full_prompt:
            full_prompt = "Extract all text from this image."

        logger.info(f"Processing OCR request (image size: {image.size})")
        logger.info(f"Prompt: {full_prompt[:100]}...")

        # Run OCR inference in a thread pool to avoid blocking the event loop
        logger.info("Starting OCR in background thread...")
        generated_text = await asyncio.to_thread(run_ocr_inference, image, full_prompt)

        # Format OpenAI-compatible response
        response = {
            "id": f"chatcmpl-{os.urandom(12).hex()}",
            "object": "chat.completion",
            "created": int(datetime.now().timestamp()),
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
                "prompt_tokens": len(full_prompt.split()),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(full_prompt.split()) + len(generated_text.split())
            }
        }

        return response

    except Exception as e:
        logger.error(f"OCR inference error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )


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


@app.post("/v2/analyze-wound")
async def analyze_wound_v2_sse(
    request: Request,
    wound_image: UploadFile = File(...),
    authorization: Optional[str] = Header(None)
):
    """
    Analyze wound image using vision AI with Server-Sent Events (SSE) for real-time progress.

    This endpoint streams progress updates to the client as the analysis progresses.

    Security:
    - Requires SERVICE_API_KEY in Authorization header
    - Optional IP whitelist

    Usage from client (JavaScript example):
        const eventSource = new EventSource('/v2/analyze-wound');

        eventSource.addEventListener('progress', (e) => {
            const data = JSON.parse(e.data);
            console.log(data.message, data.percent);
        });

        eventSource.addEventListener('result', (e) => {
            const data = JSON.parse(e.data);
            console.log('Analysis complete:', data);
            eventSource.close();
        });

        eventSource.addEventListener('error', (e) => {
            const data = JSON.parse(e.data);
            console.error('Error:', data.error);
            eventSource.close();
        });

    Or using fetch with streaming:
        const response = await fetch('/v2/analyze-wound', {
            method: 'POST',
            headers: {
                'Authorization': 'Bearer YOUR_SERVICE_API_KEY'
            },
            body: formData
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();

        while (true) {
            const {done, value} = await reader.read();
            if (done) break;

            const text = decoder.decode(value);
            const lines = text.split('\\n\\n');

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = JSON.parse(line.slice(6));
                    console.log(data);
                }
            }
        }

    Returns:
        Server-Sent Events stream with:
        - event: progress - Progress updates with percentage
        - event: result - Final analysis result
        - event: error - Error information
    """
    # Verify security
    verify_service_api_key(authorization)
    verify_ip_whitelist(request)

    async def event_generator():
        """Generate SSE events for real-time progress updates."""
        try:
            # Send initial progress
            yield f"event: progress\ndata: {{'message': 'Réception de l\\'image...', 'percent': 0}}\n\n"

            # Read image
            image_content = await wound_image.read()

            # Validate size (5MB max)
            if len(image_content) > 5 * 1024 * 1024:
                error_data = {
                    "success": False,
                    "error": "Image trop grande (max 5MB)"
                }
                yield f"event: error\ndata: {error_data}\n\n"
                return

            yield f"event: progress\ndata: {{'message': 'Image reçue, préparation...', 'percent': 20}}\n\n"

            # Convert to base64
            image_base64 = _image_to_base64(image_content)

            yield f"event: progress\ndata: {{'message': 'Envoi vers l\\'API Vision...', 'percent': 40}}\n\n"

            # Call appropriate vision API
            if VISION_PROVIDER == "openai":
                yield f"event: progress\ndata: {{'message': 'Analyse en cours (OpenAI GPT-4 Vision)...', 'percent': 60}}\n\n"
                data = _analyze_with_openai_vision(image_base64)
            elif VISION_PROVIDER == "anthropic":
                yield f"event: progress\ndata: {{'message': 'Analyse en cours (Claude Vision)...', 'percent': 60}}\n\n"
                data = _analyze_with_claude_vision(image_base64)
            else:
                error_data = {
                    "success": False,
                    "error": f"Invalid vision provider: {VISION_PROVIDER}"
                }
                yield f"event: error\ndata: {error_data}\n\n"
                return

            yield f"event: progress\ndata: {{'message': 'Analyse terminée, formatage des résultats...', 'percent': 90}}\n\n"

            logger.info("Wound analysis completed successfully")

            # Send final result
            result = {
                "success": True,
                "data": data,
                "percent": 100
            }

            # Format as JSON string for SSE
            import json
            result_json = json.dumps(result, ensure_ascii=False)

            yield f"event: result\ndata: {result_json}\n\n"

            # Send completion message
            yield f"event: complete\ndata: {{'message': 'Analyse terminée'}}\n\n"

        except HTTPException as e:
            error_data = {
                "success": False,
                "error": str(e.detail),
                "status_code": e.status_code
            }
            import json
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
        except Exception as e:
            logger.error(f"Wound analysis error: {e}")
            error_data = {
                "success": False,
                "error": str(e)
            }
            import json
            yield f"event: error\ndata: {json.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
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


# =============================================================================
# Text Processing Endpoints (using OpenAI GPT for text-only tasks)
# =============================================================================

# Prompts for report cleanup
REPORT_CLEANUP_SYSTEM_PROMPT_FR = """Tu es un assistant spécialisé dans la correction et l'assainissement de rapports de soins infirmiers en français.

Ton rôle:
1. Corriger les fautes d'orthographe et de grammaire
2. Corriger la ponctuation
3. Standardiser les abréviations médicales courantes (TA, FC, SpO2, etc.)
4. Garder un style professionnel et concis
5. REFORMULER de manière neutre et professionnelle tout contenu sensible

ASSAINISSEMENT JURIDIQUE - Tu DOIS reformuler ou supprimer:
- Toute mention de violence physique (coups, gifles, contention forcée non prescrite)
- Toute mention d'agressivité du soignant envers le patient
- Tout langage inapproprié, insultant ou dégradant
- Toute mention de négligence ou maltraitance
- Tout commentaire subjectif négatif sur le patient ou sa famille

REFORMULATIONS SUGGÉRÉES:
- "patient agité" au lieu de descriptions de confrontation
- "refus de soins" au lieu de détails sur résistance physique
- "communication difficile" au lieu de langage conflictuel
- "soins adaptés à l'état du patient" pour les situations complexes

Règles STRICTES:
- GARDER toutes les informations médicales objectives (constantes, soins effectués, observations cliniques)
- Utiliser les abréviations médicales standard
- Retourner UNIQUEMENT le texte corrigé, sans explication ni commentaire
- Le rapport doit rester factuel et professionnel"""

REPORT_CLEANUP_USER_PROMPT_FR = """Corrige et assainis ce rapport de soin infirmier. Reformule tout contenu juridiquement sensible de manière professionnelle. Retourne uniquement le texte corrigé:

{text}"""


class ReportCleanupRequest(BaseModel):
    """Request for cleaning up nursing reports."""
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


# =============================================================================
# Patient Medical Summary Models and Prompts
# =============================================================================

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


def _cleanup_text_with_openai(text: str) -> str:
    """Clean up nursing report text using OpenAI GPT-4o-mini."""
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured"
        )

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    payload = {
        "model": "gpt-4o-mini",  # Cheaper and faster for text tasks
        "messages": [
            {
                "role": "system",
                "content": REPORT_CLEANUP_SYSTEM_PROMPT_FR
            },
            {
                "role": "user",
                "content": REPORT_CLEANUP_USER_PROMPT_FR.format(text=text)
            }
        ],
        "max_tokens": len(text) * 2 + 100,
        "temperature": 0.1  # Low temperature for consistency
    }

    logger.info(f"Calling OpenAI GPT-4o-mini for text cleanup (text length: {len(text)})")
    response = requests.post(url, headers=headers, json=payload, timeout=30)

    if response.status_code != 200:
        logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
        raise HTTPException(
            status_code=response.status_code,
            detail=f"OpenAI API error: {response.text}"
        )

    result = response.json()
    cleaned_text = result['choices'][0]['message']['content'].strip()

    # Remove quotes if model added them
    if cleaned_text.startswith('"') and cleaned_text.endswith('"'):
        cleaned_text = cleaned_text[1:-1]
    if cleaned_text.startswith("'") and cleaned_text.endswith("'"):
        cleaned_text = cleaned_text[1:-1]

    return cleaned_text


@app.post("/v1/report/cleanup", response_model=ReportCleanupResponse)
async def cleanup_report(request: ReportCleanupRequest):
    """
    Clean up nursing report text using OpenAI GPT-4o-mini.
    Corrects typos, grammar, and standardizes medical abbreviations.

    This is a TEXT-ONLY endpoint (no image required).

    Usage:
        POST /v1/report/cleanup
        {"text": "Pansemant fait, plaie propre"}

    Returns:
        {
            "original_text": "Pansemant fait, plaie propre",
            "cleaned_text": "Pansement fait, plaie propre.",
            "prompt_used": "...",
            "changes_made": true,
            "model": "gpt-4o-mini"
        }
    """
    if not request.text or not request.text.strip():
        return ReportCleanupResponse(
            original_text=request.text,
            cleaned_text=request.text,
            prompt_used="",
            changes_made=False,
            model="gpt-4o-mini"
        )

    try:
        cleaned_text = _cleanup_text_with_openai(request.text)

        return ReportCleanupResponse(
            original_text=request.text,
            cleaned_text=cleaned_text,
            prompt_used=f"System: {REPORT_CLEANUP_SYSTEM_PROMPT_FR}\n\nUser: {REPORT_CLEANUP_USER_PROMPT_FR.format(text=request.text)}",
            changes_made=(cleaned_text != request.text),
            model="gpt-4o-mini"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Report cleanup error: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@app.post("/v1/report/cleanup/preview")
async def cleanup_report_preview(request: ReportCleanupRequest):
    """
    Preview the prompts that would be used for cleanup without executing.
    Useful for debugging and transparency.

    Usage:
        POST /v1/report/cleanup/preview
        {"text": "Pansemant fait"}

    Returns:
        {
            "system_prompt": "...",
            "user_prompt": "...",
            "original_text": "Pansemant fait",
            "model": "gpt-4o-mini"
        }
    """
    return {
        "system_prompt": REPORT_CLEANUP_SYSTEM_PROMPT_FR,
        "user_prompt": REPORT_CLEANUP_USER_PROMPT_FR.format(text=request.text),
        "full_prompt": f"System: {REPORT_CLEANUP_SYSTEM_PROMPT_FR}\n\nUser: {REPORT_CLEANUP_USER_PROMPT_FR.format(text=request.text)}",
        "original_text": request.text,
        "model": "gpt-4o-mini"
    }


@app.post("/v1/patient/medical-summary", response_model=PatientMedicalSummaryResponse)
async def generate_patient_medical_summary(
    request: PatientMedicalSummaryRequest,
    authorization: Optional[str] = Header(None)
):
    """
    Generate a comprehensive medical summary from patient event reports.

    This endpoint analyzes multiple event reports and extracts:
    - Key medical facts (allergies, conditions, treatments)
    - Important events and changes in patient status
    - A narrative summary of the patient's medical history

    Designed for overnight batch processing of patient records.

    Security:
    - Requires SERVICE_API_KEY in Authorization header

    Usage:
        POST /v1/patient/medical-summary
        Authorization: Bearer YOUR_SERVICE_API_KEY
        {
            "patient_id": 1365,
            "patient_name": "Dupont Jean",
            "events": [
                {"date": "2024-01-15", "report": "Pansement plaie tibia..."},
                {"date": "2024-02-20", "report": "Tension élevée 160/95..."}
            ]
        }
    """
    import json

    # Verify API key
    verify_service_api_key(authorization)

    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured for medical summary generation"
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
            model="gpt-4o-mini"
        )

    try:
        # Sort events by date
        sorted_events = sorted(request.events, key=lambda x: x.date)
        date_start = sorted_events[0].date
        date_end = sorted_events[-1].date

        # Format reports for the prompt
        # Limit to most recent reports if too many (to fit in context)
        max_reports = 200
        events_to_analyze = sorted_events[-max_reports:] if len(sorted_events) > max_reports else sorted_events

        reports_text = "\n\n".join([
            f"[{event.date}] {event.report}"
            for event in events_to_analyze
            if event.report and event.report.strip()
        ])

        # Build prompts
        user_prompt = MEDICAL_SUMMARY_USER_PROMPT_FR.format(
            event_count=len(events_to_analyze),
            patient_name=request.patient_name,
            date_start=date_start,
            date_end=date_end,
            reports=reports_text
        )

        # Call OpenAI
        logger.info(f"Generating medical summary for patient {request.patient_id} ({len(events_to_analyze)} events)")

        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
        }
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": MEDICAL_SUMMARY_SYSTEM_PROMPT_FR},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": request.max_summary_length
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()

        response_text = result['choices'][0]['message']['content'].strip()

        # Parse JSON response
        try:
            # Clean up response - remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            parsed_result = json.loads(response_text.strip())

            key_facts = KeyMedicalFacts(
                allergies=parsed_result.get("key_facts", {}).get("allergies", []) or [],
                chronic_conditions=parsed_result.get("key_facts", {}).get("chronic_conditions", []) or [],
                current_treatments=parsed_result.get("key_facts", {}).get("current_treatments", []) or [],
                hospitalizations=parsed_result.get("key_facts", {}).get("hospitalizations", []) or [],
                important_events=parsed_result.get("key_facts", {}).get("important_events", []) or [],
                mobility_status=parsed_result.get("key_facts", {}).get("mobility_status"),
                cognitive_status=parsed_result.get("key_facts", {}).get("cognitive_status"),
                skin_conditions=parsed_result.get("key_facts", {}).get("skin_conditions", []) or [],
                vital_signs_alerts=parsed_result.get("key_facts", {}).get("vital_signs_alerts", []) or [],
                social_context=parsed_result.get("key_facts", {}).get("social_context")
            )

            summary = parsed_result.get("summary", "Résumé non disponible.")

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response, using raw text: {e}")
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
            model="gpt-4o-mini"
        )

    except requests.exceptions.Timeout:
        logger.error(f"Timeout generating medical summary for patient {request.patient_id}")
        raise HTTPException(status_code=504, detail="Request timed out")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request error generating medical summary: {e}")
        raise HTTPException(status_code=502, detail=f"External API error: {str(e)}")
    except Exception as e:
        logger.error(f"Medical summary generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")


@app.post("/v1/text/chat")
async def text_chat(request: ChatCompletionRequest):
    """
    Text-only chat completion endpoint using OpenAI GPT-4o-mini.
    Unlike /v1/chat/completions, this does NOT require an image.

    Usage:
        POST /v1/text/chat
        {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ]
        }
    """
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=503,
            detail="OpenAI API key not configured"
        )

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }

    # Convert messages to OpenAI format
    messages = []
    for msg in request.messages:
        if isinstance(msg.content, str):
            messages.append({"role": msg.role, "content": msg.content})
        elif isinstance(msg.content, list):
            # Extract text only
            text_parts = [p.get("text", "") for p in msg.content if isinstance(p, dict) and p.get("type") == "text"]
            messages.append({"role": msg.role, "content": " ".join(text_parts)})

    payload = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p
    }

    logger.info(f"Calling OpenAI GPT-4o-mini for text chat")
    response = requests.post(url, headers=headers, json=payload, timeout=60)

    if response.status_code != 200:
        logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
        raise HTTPException(
            status_code=response.status_code,
            detail=f"OpenAI API error: {response.text}"
        )

    return response.json()


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
