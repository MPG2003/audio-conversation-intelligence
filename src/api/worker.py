import asyncio
import uuid
import time
from typing import Any, Dict

# Global dictionary to hold job status and results
JOBS: Dict[str, Dict[str, Any]] = {}
queue = asyncio.Queue()

async def process_audio_job(job_id: str, tmp_path: str, filename: str, transcriber, diarize_fn, analyze_fn):
    JOBS[job_id]["status"] = "processing"
    try:
        # We run the heavy tasks in a thread pool to avoid blocking the event loop
        transcription = await asyncio.to_thread(transcriber.transcribe, tmp_path)
        diarization = await asyncio.to_thread(diarize_fn, tmp_path, transcription.segments)
        
        stages = [
            {"id": "upload", "title": "Audio uploaded", "description": filename, "status": "completed"},
            {"id": "transcription", "title": "Speech transcription", "description": "Audio converted to text", "status": "completed"},
            {"id": "diarization", "title": "Speaker diarization", "description": "Agent and customer turns aligned", "status": "completed"},
            {"id": "privacy", "title": "Local PII extraction", "description": "Sensitive details redacted before LLaMA", "status": "completed"},
            {"id": "analysis", "title": "Feature extraction", "description": "Sales features extracted", "status": "completed"},
            {"id": "prediction", "title": "Conversion scoring", "description": "Lead score calculated", "status": "completed"},
        ]
        
        result = await analyze_fn(
            transcription.text,
            source_name=filename,
            source_type="audio",
            language=transcription.language,
            transcription_confidence=transcription.confidence,
            whisper_model=transcriber.model_size,
            pipeline=stages,
            diarization=diarization,
            start_time=JOBS[job_id]["started_at"],
        )
        
        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["result"] = result
        
    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
    finally:
        # Cleanup temp file
        import os
        try:
            os.remove(tmp_path)
        except OSError:
            pass

async def background_worker(transcriber, diarize_fn, analyze_fn):
    print("Background worker started.")
    while True:
        job = await queue.get()
        job_id = job["job_id"]
        tmp_path = job["tmp_path"]
        filename = job["filename"]
        
        try:
            await process_audio_job(job_id, tmp_path, filename, transcriber, diarize_fn, analyze_fn)
        except Exception as e:
            print(f"Worker error on job {job_id}: {e}")
        finally:
            queue.task_done()
