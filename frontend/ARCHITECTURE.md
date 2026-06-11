# AI Platform Architecture & API Integration

This document outlines the architecture for connecting the Next.js frontend to the AI backend (Whisper, LLaMA 3, XGBoost).

## 1. Global State Management (Zustand)
File: `src/store/useAppStore.ts`

The application uses Zustand for centralized state management across sections, keeping the UI synchronized. 
The state flows through these primary phases:
1. **`audioFile` & `uploadProgress`**: Tracks the audio file during Whisper transcription. `isUploading` dictates the UI loading state.
2. **`transcription`**: Stores the raw or edited text returned from Whisper.
3. **`features`**: Populated after LLaMA 3 analysis (`isExtracting`). Contains NLP extracted signals.
4. **`prediction`**: Populated by the XGBoost endpoint (`isPredicting`).
5. **`error`**: A global error state tracked centrally and dismissed contextually in the UI sections.

## 2. API Service Layer
File: `src/services/api.ts`

### Configuration
The Axios client automatically targets your backend using:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api
```
*(By default, it falls back to `http://localhost:8000/api` if the environment variable is missing.)*

### Interceptor & Retry Handling
AI inference endpoints (especially Whisper/LLaMA) can timeout or fail due to high load. We implemented a custom Axios interceptor in `api.ts`.
- **Upload (`/upload`)**: Retries 2 times with a 3000ms backoff.
- **Extraction (`/analyze`)**: Retries 1 time with a 2000ms backoff.
- **Prediction (`/predict`)**: Retries 1 time with a 2000ms backoff.

### Endpoints
1. **Whisper Transcription**
   - **Method**: `POST /upload`
   - **Payload**: `multipart/form-data` with an `audio` File.
   - **Triggered in**: `UploadSection.tsx`

2. **LLaMA 3 Feature Extraction**
   - **Method**: `POST /analyze`
   - **Payload**: `{ text: transcription }`
   - **Triggered in**: `ConversationInputSection.tsx`

3. **XGBoost Prediction Model**
   - **Method**: `POST /predict`
   - **Payload**: `{ features: extracted_features_object }`
   - **Triggered in**: `ExtractionSection.tsx`

## 3. UI Flow & Error Handling
The UI sections act as a progressive pipeline:
- `UploadSection` displays real-time progress while calling `/upload`.
- `ConversationInputSection` lets users view/edit the transcript and triggers `/analyze`.
- `ExtractionSection` shows the dynamic LLaMA 3 extraction cards and triggers the `/predict` workflow.
- `PredictionSection` displays the final XGBoost probabilities and insights.

**Errors** are explicitly caught in try/catch blocks within the sections, pushed to the `useAppStore`'s `error` state, and rendered gracefully inline with a red `dismiss` functionality. This ensures the 3D premium experience doesn't break due to a raw backend failure.
