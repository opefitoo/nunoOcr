# nunoOcr v2 API - Real-Time Wound Analysis with SSE

## 🚀 What's New in v2

The v2 API adds **Server-Sent Events (SSE)** support for real-time progress updates during wound analysis.

### Key Features

- ✅ **Real-time progress updates** - See analysis progress as it happens (0% → 100%)
- ✅ **Event-driven architecture** - Progress, result, error, and complete events
- ✅ **Same security as v1** - SERVICE_API_KEY and IP whitelist support
- ✅ **Backward compatible** - v1 endpoints remain unchanged
- ✅ **No timeouts** - Streaming prevents long request timeouts
- ✅ **Better UX** - Show users exactly what's happening

## 📡 API Endpoints

### v2 Endpoint (NEW)

```
POST /v2/analyze-wound
```

**Returns:** Server-Sent Events stream with real-time progress

**Events:**
- `progress` - Analysis progress updates with percentage
- `result` - Final analysis result (JSON)
- `error` - Error information
- `complete` - Analysis completion signal

### v1 Endpoints (UNCHANGED)

```
POST /v1/analyze-wound          # Simple request/response
POST /v1/compare-wound-progress # Multi-image comparison
```

## 🔧 Usage Examples

### JavaScript (Browser)

```javascript
const formData = new FormData();
formData.append('wound_image', fileInput.files[0]);

const response = await fetch('http://46.224.6.193:8765/v2/analyze-wound', {
    method: 'POST',
    headers: {
        'Authorization': 'Bearer YOUR_SERVICE_API_KEY'
    },
    body: formData
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = '';

while (true) {
    const {done, value} = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, {stream: true});
    const lines = buffer.split('\\n\\n');
    buffer = lines.pop();

    for (const line of lines) {
        if (!line.trim()) continue;

        const eventMatch = line.match(/event: (\\w+)\\ndata: (.+)/s);
        if (eventMatch) {
            const [, eventType, dataStr] = eventMatch;
            const data = JSON.parse(dataStr);

            if (eventType === 'progress') {
                console.log(`${data.percent}% - ${data.message}`);
                updateProgressBar(data.percent);
            } else if (eventType === 'result') {
                console.log('Analysis complete:', data.data);
                displayResults(data.data);
            } else if (eventType === 'error') {
                console.error('Error:', data.error);
            }
        }
    }
}
```

### Python

```python
import requests
import json

files = {'wound_image': open('wound.jpg', 'rb')}
headers = {'Authorization': 'Bearer YOUR_SERVICE_API_KEY'}

response = requests.post(
    'http://46.224.6.193:8765/v2/analyze-wound',
    headers=headers,
    files=files,
    stream=True
)

buffer = ""
for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
    if chunk:
        buffer += chunk

        while '\\n\\n' in buffer:
            message, buffer = buffer.split('\\n\\n', 1)
            lines = message.strip().split('\\n')

            event_type = None
            data = None

            for line in lines:
                if line.startswith('event: '):
                    event_type = line[7:]
                elif line.startswith('data: '):
                    data = json.loads(line[6:])

            if event_type == 'progress':
                print(f"{data['percent']}% - {data['message']}")
            elif event_type == 'result':
                print("Analysis complete:", data['data'])
```

### cURL (Testing)

```bash
curl -N -X POST http://46.224.6.193:8765/v2/analyze-wound \
  -H "Authorization: Bearer YOUR_SERVICE_API_KEY" \
  -F "wound_image=@wound.jpg"
```

**Note:** The `-N` flag disables buffering to see events in real-time.

## 📊 Progress Events

The v2 endpoint sends progress events at key stages:

| Percent | Message | Description |
|---------|---------|-------------|
| 0% | Réception de l'image... | Initial connection |
| 20% | Image reçue, préparation... | Image validated and read |
| 40% | Envoi vers l'API Vision... | Converting to base64 |
| 60% | Analyse en cours (GPT-4/Claude)... | Vision API processing |
| 90% | Analyse terminée, formatage... | Formatting results |
| 100% | Result event sent | Analysis complete |

## 🎯 Event Format

### Progress Event
```
event: progress
data: {"message": "Analyse en cours...", "percent": 60}

```

### Result Event
```
event: result
data: {"success": true, "data": {...}, "percent": 100}

```

### Error Event
```
event: error
data: {"success": false, "error": "Error message", "status_code": 500}

```

### Complete Event
```
event: complete
data: {"message": "Analyse terminée"}

```

## 🔐 Security

Same security model as v1:

1. **Service API Key** (recommended)
   ```bash
   Authorization: Bearer nuno_service_xxxxx
   ```

2. **IP Whitelist** (optional)
   - Configure via `ALLOWED_IPS` environment variable
   - Example: `ALLOWED_IPS=128.140.12.236`

3. **Image Size Validation**
   - Max 5MB per image
   - Returns 400 error if exceeded

## 🧪 Testing

### Interactive HTML Client

Open `examples/client_v2_sse.html` in your browser for an interactive test interface.

### Python CLI Client

```bash
python examples/client_v2_sse.py wound.jpg [service_api_key]
```

## 📝 Response Format

The final result event contains the same structured data as v1:

```json
{
  "success": true,
  "data": {
    "type_plaie": "Plaie chirurgicale",
    "localisation": "Abdomen",
    "dimensions": {
      "longueur_cm": 15,
      "largeur_cm": 2
    },
    "stade_cicatrisation": "Cicatrisation primaire en cours",
    "methode_fermeture": "Points de suture",
    "nombre_points": 12,
    "signes_infection": [],
    "complications": [],
    "etat_general": "Bon état, pas de rougeur",
    "confiance": "élevée",
    "notes": "Plaie propre et sèche"
  },
  "percent": 100
}
```

## 🆚 v1 vs v2 Comparison

| Feature | v1 (/v1/analyze-wound) | v2 (/v2/analyze-wound) |
|---------|------------------------|------------------------|
| **Response Type** | JSON (single response) | SSE stream (multiple events) |
| **Progress Updates** | ❌ No | ✅ Yes (real-time) |
| **Timeout Risk** | ⚠️ Possible for slow analysis | ✅ No (streaming) |
| **Client Complexity** | Simple fetch() | SSE parsing required |
| **Use Case** | Simple integrations | Rich UX with progress |
| **Security** | ✅ SERVICE_API_KEY + IP whitelist | ✅ SERVICE_API_KEY + IP whitelist |
| **Result Format** | Same JSON structure | Same JSON structure |

## 🔄 Migration from v1 to v2

Both endpoints use the same authentication and return the same result format. To migrate:

1. **Keep using v1** if you don't need progress updates
2. **Switch to v2** if you want real-time feedback:
   - Change endpoint from `/v1/analyze-wound` to `/v2/analyze-wound`
   - Add SSE parsing logic to handle events
   - Display progress to users

## 🛠️ Environment Variables

Same as v1:

```bash
# Vision API
OPENAI_API_KEY=sk-proj-...
VISION_PROVIDER=openai  # or "anthropic"

# Security
SERVICE_API_KEY=nuno_service_...
ALLOWED_IPS=128.140.12.236  # Optional
```

## 📚 Additional Resources

- [Server-Sent Events MDN](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)
- [FastAPI StreamingResponse](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [requests streaming](https://requests.readthedocs.io/en/latest/user/advanced/#streaming-requests)

## 🐛 Troubleshooting

### Events not arriving in real-time

**Problem:** All events arrive at once instead of streaming.

**Solution:** Check for buffering in reverse proxies:
```nginx
# Nginx
proxy_buffering off;
proxy_cache off;
```

Add header to disable buffering:
```
X-Accel-Buffering: no
```

### Connection closes unexpectedly

**Problem:** Stream closes before completion.

**Solution:**
- Check firewall/proxy timeouts
- Ensure `Connection: keep-alive` header
- Monitor server logs for errors

### CORS issues in browser

**Problem:** Browser blocks SSE requests.

**Solution:** Server already has CORS enabled for all origins:
```python
allow_origins=["*"]
```

If you need specific origin control, modify `server_with_wound_analysis.py`.

---

**Version:** 4.3.0+
**Last Updated:** 2025-01-08
**Status:** ✅ Production Ready
