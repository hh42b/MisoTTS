import os
import io
import torch
import torchaudio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Set environment variables for HF
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ["NO_TORCH_COMPILE"] = "1"

from generator import DEFAULT_MISO_TTS_REPO_ID, load_miso_8b, Segment

app = FastAPI(title="MisoTTS API")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for now, restrict to frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global generator instance
generator = None

class TTSRequest(BaseModel):
    text: str
    speaker: int = 0
    max_audio_length_ms: int = 30000

@app.on_event("startup")
async def startup_event():
    global generator
    print("Initializing MisoTTS model...")
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    model_source = os.environ.get("MISO_TTS_8B_MODEL", DEFAULT_MISO_TTS_REPO_ID)
    print(f"Loading model from: {model_source}")
    
    # Load the generator
    generator = load_miso_8b(device, model_path_or_repo_id=model_source)
    print("MisoTTS model loaded successfully.")

@app.get("/api/health")
def health_check():
    if generator is not None:
        return {"status": "ok", "model": "loaded"}
    return {"status": "starting", "model": "not loaded"}

@app.post("/api/tts")
def generate_tts(req: TTSRequest):
    if generator is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")
    
    print(f"Generating TTS for text: {req.text[:50]}... Speaker: {req.speaker}")
    
    try:
        # Generate audio tensor
        audio_tensor = generator.generate(
            text=req.text,
            speaker=req.speaker,
            context=[], # No context for individual slide generation
            max_audio_length_ms=req.max_audio_length_ms,
        )
        
        # We need to save the tensor to a wav file in memory
        buffer = io.BytesIO()
        
        # torchaudio expects (channels, time), so unsqueeze audio
        audio_to_save = audio_tensor.unsqueeze(0).cpu()
        torchaudio.save(
            buffer,
            audio_to_save,
            generator.sample_rate,
            format="wav"
        )
        
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav")
        
    except Exception as e:
        print(f"Error generating TTS: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("tts_server:app", host="0.0.0.0", port=8000, reload=False)
