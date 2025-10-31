#!/usr/bin/env python3
"""
DeepSeek-OCR FastAPI server using transformers (CPU-compatible)

This is a CPU-optimized version that uses transformers directly
instead of vLLM. It's slower but works reliably on CPU servers.

Performance:
- CPU: 10-20 seconds per page
- GPU: 2-5 seconds per page (if available)

Memory: 8-12GB RAM required
"""

import os
import io
import base64
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

import torch
from PIL import Image
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-OCR")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Initialize FastAPI app
app = FastAPI(
    title="DeepSeek-OCR API",
    description="CPU-optimized OCR API using DeepSeek-OCR model with transformers",
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

# Global model instances
model = None
processor = None
device = None


class ChatMessage(BaseModel):
    role: str
    content: str | list


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.0
    max_tokens: Optional[int] = 2000
    top_p: Optional[float] = 1.0


def load_model():
    """Load the DeepSeek-OCR model using transformers."""
    global model, processor, device

    try:
        from transformers import AutoModel, AutoTokenizer

        # Monkey patch: Handle missing Flash Attention gracefully
        # The model tries to import LlamaFlashAttention2 which doesn't exist
        # We patch it to use standard attention instead
        try:
            import transformers.models.llama.modeling_llama as llama_module
            if not hasattr(llama_module, 'LlamaFlashAttention2'):
                logger.info("Patching missing LlamaFlashAttention2 for CPU compatibility")
                # Create a dummy class that will never be used
                # The model will fall back to eager attention
                class DummyFlashAttention:
                    pass
                llama_module.LlamaFlashAttention2 = DummyFlashAttention
        except Exception as e:
            logger.warning(f"Could not patch flash attention: {e}")

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
            logger.info("No GPU detected, using CPU (slower but works)")

        # Load tokenizer
        logger.info("Loading tokenizer...")
        processor = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )

        # Load model
        logger.info("Loading model weights...")
        # DeepSeek-OCR uses a custom model class, must use AutoModel with trust_remote_code
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device,
            low_cpu_mem_usage=True,  # Optimize for CPU
            attn_implementation="eager",  # Disable flash attention (CPU compatible)
        )

        model.eval()  # Set to evaluation mode

        logger.info(f"✅ Model loaded successfully on {device}")
        logger.info(f"   Model: {MODEL_NAME}")
        logger.info(f"   Device: {device}")
        logger.info(f"   Dtype: {torch_dtype}")

        return True

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error("Server will start but inference will fail until model is loaded")
        return False


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup."""
    logger.info(f"Starting DeepSeek-OCR server on {HOST}:{PORT}")
    load_model()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "DeepSeek-OCR API",
        "version": "2.0.0",
        "status": "ok" if model is not None else "initializing",
        "engine": "transformers",
        "device": device if device else "unknown"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model is None:
        return {
            "status": "initializing",
            "model": MODEL_NAME,
            "device": "loading"
        }

    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": device,
        "engine": "transformers"
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(datetime.now().timestamp()),
                "owned_by": "deepseek-ai",
                "engine": "transformers"
            }
        ]
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


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint with vision support.

    Handles OCR requests with images in messages.
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

        # Save image temporarily for model.infer()
        import tempfile
        import os

        # Initialize paths for cleanup
        temp_image_path = None
        temp_output_dir = None

        try:
            # Create temp file for input image
            tmp_img_fd, temp_image_path = tempfile.mkstemp(suffix='.png', dir='/tmp')
            os.close(tmp_img_fd)  # Close image file descriptor
            image.save(temp_image_path)  # Save PIL image
            logger.info(f"Created and saved image to: {temp_image_path}")

            # Create temp directory for output (model.infer() expects a directory)
            temp_output_dir = tempfile.mkdtemp(dir='/tmp')
            logger.info(f"Created temp output directory: {temp_output_dir}")

            # Use DeepSeek-OCR's custom infer method
            logger.info("Calling model.infer()...")
            generated_text = model.infer(
                tokenizer=processor,
                prompt=full_prompt,
                image_file=temp_image_path,
                output_path=temp_output_dir,  # Provide temp output directory
                save_results=False,  # Don't save results to file
                eval_mode=False  # Inference mode, not evaluation
            )
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            logger.error(f"Temp image path was: {temp_image_path}")
            raise
        finally:
            # Clean up temp files
            if temp_image_path and os.path.exists(temp_image_path):
                os.unlink(temp_image_path)
                logger.info(f"Cleaned up temp image: {temp_image_path}")
            if temp_output_dir and os.path.exists(temp_output_dir):
                import shutil
                shutil.rmtree(temp_output_dir)
                logger.info(f"Cleaned up temp output dir: {temp_output_dir}")

        logger.info(f"✅ OCR completed (output length: {len(generated_text)} chars)")

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


if __name__ == "__main__":
    logger.info(f"Starting DeepSeek-OCR server on {HOST}:{PORT}")
    logger.info(f"Engine: transformers (CPU-compatible)")
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
