# Checking OCR Service Logs

## What to look for in Dokploy Logs

### ✅ Good Signs (Downloading)
```
INFO:     Starting DeepSeek-OCR server
Loading model: deepseek-ai/DeepSeek-OCR
Downloading (...)
```

### ✅ Good Signs (Ready)
```
Model loaded successfully: deepseek-ai/DeepSeek-OCR
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### ⚠️ Warning Signs
```
Error: CUDA out of memory
Error: Not enough RAM
Connection timeout
Failed to download model
```

### ❌ Critical Issues
```
ModuleNotFoundError
ImportError: vllm
Killed (Out of memory)
```

## Steps to Check

1. **Dokploy UI:**
   - Go to NuNoOcr service
   - Click "Logs" tab
   - Scroll to the BOTTOM
   - Copy the last 30-50 lines

2. **Look for these keywords:**
   - `Loading model`
   - `Downloading`
   - `Error`
   - `Failed`
   - `complete`

3. **If you see "Application startup complete":**
   - The service should be ready!
   - Try refreshing Django upload page

4. **If stuck on "initializing" for >20 minutes:**
   - There might be a resource issue
   - Check server RAM/CPU in Monitoring tab
