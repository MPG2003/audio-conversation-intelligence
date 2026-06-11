# Sales Conversation Intelligence

This project analyzes sales conversations from audio or text and exposes the workflow through:

- a FastAPI backend in `src/api/server.py`
- a Next.js frontend in `frontend/`
- supporting data, models, notebooks, and documentation

## Project Structure

- `src/` Python application code
- `frontend/` React frontend
- `data/` raw and processed datasets
- `audio/` sample audio files
- `models/` trained conversion-model artifacts
- `notebooks/` experiment notebooks
- `docs/` project documentation
- `scripts/` helper scripts

Generated dependencies and build output are intentionally excluded. Recreate them with
`python -m venv .venv`, `pip install -r requirements.txt`, and `npm install` in `frontend/`.

## Configuration

Copy `.env.example` to `.env` and `frontend/.env.example` to `frontend/.env.local`, then
set the required credentials. Never commit populated environment files.

Audio processing requires `ffmpeg` to be installed and available on `PATH`.

## Llama 3 Prompts

- App prompt location: `src/aspect_sentiment/engine.py`
- Copy/paste reader prompt: `docs/llama3_reader_prompt.md`

The app prompt is JSON-only because the frontend expects structured extraction results. Use the docs prompt when you want Llama 3 to explain files, logs, notebooks, transcripts, or datasets in normal language.

## Run The App

Backend:

```bash
.venv\Scripts\python.exe -m uvicorn src.api.server:app --reload --port 8000
```

To use Llama 3 through an API key for feature extraction:

```bash
set LLAMA_API_KEY=your_api_key_here
set LLAMA_MODEL=llama-3.3-70b-versatile
set LLAMA_API_URL=https://api.groq.com/openai/v1/chat/completions
.venv\Scripts\python.exe -m uvicorn src.api.server:app --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.
