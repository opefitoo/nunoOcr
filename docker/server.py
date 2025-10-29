#!/usr/bin/env python3
"""
DeepSeek-OCR FastAPI server using vLLM
OpenAI-compatible API for OCR inference
"""

import os
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
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
    description="OpenAI-compatible OCR API using DeepSeek-OCR model",
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
            max_model_len=8192,  # Adjust based on your needs
            gpu_memory_utilization=0.9,  # Use 90% of GPU memory
            # For CPU-only mode, uncomment:
            # tensor_parallel_size=1,
        )

        logger.info(f"Model loaded successfully: {MODEL_NAME}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("Server will start but inference will fail until model is loaded")


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

    Handles OCR requests with images in messages.
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

        # Build prompt from messages
        # For DeepSeek-OCR, we need to format the prompt appropriately
        prompt_parts = []

        for msg in messages:
            role = msg.role
            content = msg.content

            if isinstance(content, str):
                prompt_parts.append(f"{role}: {content}")
            elif isinstance(content, list):
                # Handle multi-modal content (text + image)
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "input_text":
                            prompt_parts.append(f"{role}: {item.get('text', '')}")
                        elif item.get("type") == "input_image":
                            # Image URL is included in the content
                            # vLLM handles this internally
                            pass

        prompt = "\n".join(prompt_parts)

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


if __name__ == "__main__":
    logger.info(f"Starting DeepSeek-OCR server on {HOST}:{PORT}")
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info"
    )
