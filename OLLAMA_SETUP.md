# Ollama Setup for nunoOcr - FREE Local LLM

This guide explains how to use Ollama with nunoOcr for **FREE** AI-powered medical summary generation instead of paid OpenAI.

## Cost Comparison

| Provider | Model | Daily Cost (154 patients) | Monthly Cost |
|----------|-------|---------------------------|--------------|
| OpenAI | gpt-4o-mini | ~$0.36 | ~$10.94 |
| **Ollama** | **qwen2.5:7b** | **$0 (FREE)** | **$0** |

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Start services
docker-compose -f docker-compose.ollama.yml up -d

# Pull the model (one-time)
docker exec nunoocr_ollama ollama pull qwen2.5:7b

# Verify
curl http://localhost:11434/api/tags
curl http://localhost:8765/health
```

### Option 2: Standalone Ollama

```bash
# Install Ollama (macOS/Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Or on macOS with Homebrew
brew install ollama

# Start Ollama server
ollama serve

# Pull the model
ollama pull qwen2.5:7b

# Test
ollama run qwen2.5:7b "Bonjour, comment allez-vous?"
```

## Recommended Models

| Model | Size | RAM | Speed | French | Medical | Best For |
|-------|------|-----|-------|--------|---------|----------|
| **qwen2.5:7b** | 4.7GB | 8-10GB | Medium | Excellent | Good | **Recommended** |
| qwen2.5:3b | 2GB | 4-6GB | Fast | Good | Fair | Low resources |
| qwen2.5:14b | 9GB | 16GB | Slow | Excellent | Very Good | Better quality |
| llama3.2:8b | 4.7GB | 8-10GB | Medium | Good | Good | Alternative |
| mistral:7b | 4.1GB | 8GB | Fast | Good | Fair | Speed priority |

### Pull other models

```bash
# Lighter model (less RAM)
ollama pull qwen2.5:3b

# Better quality (more RAM)
ollama pull qwen2.5:14b

# Alternative
ollama pull llama3.2
```

## Environment Variables

```bash
# Use Ollama instead of OpenAI
LLM_PROVIDER=ollama

# Ollama server URL
OLLAMA_URL=http://localhost:11434

# Model to use
LLM_MODEL=qwen2.5:7b
```

## Switching Between Providers

### Use Ollama (FREE)
```bash
export LLM_PROVIDER=ollama
export OLLAMA_URL=http://localhost:11434
export LLM_MODEL=qwen2.5:7b
```

### Use OpenAI (Paid)
```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-your-key-here
```

## Hardware Requirements

### Minimum (qwen2.5:3b)
- CPU: 4 cores
- RAM: 6GB
- Disk: 5GB

### Recommended (qwen2.5:7b)
- CPU: 8 cores
- RAM: 12GB
- Disk: 10GB

### Optimal (qwen2.5:14b)
- CPU: 8+ cores
- RAM: 20GB
- Disk: 15GB

## Testing

### Test Ollama directly
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "qwen2.5:7b",
  "messages": [
    {"role": "user", "content": "Résume ce rapport médical: Patient présente hypertension, diabète type 2."}
  ],
  "stream": false
}'
```

### Test via nunoOcr
```bash
curl http://localhost:8765/health
# Check: "llm_provider": "ollama", "llm_available": true
```

## Troubleshooting

### Ollama not responding
```bash
# Check if running
curl http://localhost:11434/api/tags

# Restart
docker restart nunoocr_ollama
# or
ollama serve
```

### Model not found
```bash
# List available models
ollama list

# Pull model
ollama pull qwen2.5:7b
```

### Out of memory
```bash
# Use smaller model
export LLM_MODEL=qwen2.5:3b

# Or increase swap
sudo sysctl vm.swappiness=60
```

### Slow responses
- Normal: 30-60 seconds for large summaries on CPU
- For faster responses, use GPU or smaller model

## Performance Tips

1. **Use SSD**: Models load faster from SSD
2. **More RAM**: Reduces swapping, faster inference
3. **Fewer events**: Limit to 100 events per request for faster responses
4. **Batch at night**: Run summaries during off-peak hours

## Sources

- [Ollama](https://ollama.com/)
- [Qwen2.5 on Ollama](https://ollama.com/library/qwen2.5)
- [Qwen Blog](https://qwenlm.github.io/blog/qwen2.5/)
- [Best Open-Source LLMs](https://blog.n8n.io/open-source-llm/)
