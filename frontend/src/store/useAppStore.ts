import { create } from 'zustand'

export interface AppState {
  isUploading: boolean;
  uploadProgress: number;
  audioFile: File | null;
  transcription: string;
  isExtracting: boolean;
  features: {
    sentiment: number;
    emotion: string;
    buyingIntent: string;
    budgetDetected: boolean;
    objections: string[];
    rawFeatures: { name: string; label: string }[];
    extractionProvider?: string;
    diarizedTranscript?: {
      speaker: 'Customer' | 'Agent' | string;
      rawSpeaker?: string;
      text: string;
      start?: number | null;
      end?: number | null;
    }[];
    privacy?: {
      entities: { type: string; value: string; source: string }[];
      grouped: Record<string, string[]>;
      redactionCount: number;
      provider: string;
    };
    customerBehaviorSummary?: {
      focus: string;
      intentSignals: number;
      hesitationScore: number;
      urgencySignals: number;
      objectionSignals: number;
      wordCount: number;
      privacySafe: boolean;
    };
    conversationSummary?: {
      overview: string;
      customerNeed: string;
      keyPoints: string[];
      outcome: string;
      nextAction: string;
      confidence: number;
      provider?: string;
    };
    conversionScore?: {
      probability: number;
      label: string;
      confidence: number;
    } | null;
    audioQuality?: {
      label: string;
      confidence: number;
      language?: string | null;
      whisperModel?: string | null;
    } | null;
  } | null;
  isPredicting: boolean;
  prediction: {
    probability: number;
    risk: string;
    insights: string[];
    nextSteps?: string[];
  } | null;
  error: string | null;
  followUpRefreshKey: number;

  // Live Stream State
  recordingState: 'idle' | 'recording' | 'processing' | 'analyzing' | 'completed';
  liveTranscript: string;
  socketStatus: 'disconnected' | 'connected';
  
  setUploading: (status: boolean) => void;
  setUploadProgress: (progress: number) => void;
  setAudioFile: (file: File | null) => void;
  setTranscription: (text: string) => void;
  setExtracting: (status: boolean) => void;
  setFeatures: (features: AppState['features']) => void;
  setPredicting: (status: boolean) => void;
  setPrediction: (prediction: AppState['prediction']) => void;
  setError: (error: string | null) => void;
  refreshFollowUpAlerts: () => void;

  setRecordingState: (state: AppState['recordingState']) => void;
  setLiveTranscript: (text: string | ((prev: string) => string)) => void;
  setSocketStatus: (status: AppState['socketStatus']) => void;
}

export const useAppStore = create<AppState>((set) => ({
  isUploading: false,
  uploadProgress: 0,
  audioFile: null,
  transcription: "",
  isExtracting: false,
  features: null,
  isPredicting: false,
  prediction: null,
  error: null,
  followUpRefreshKey: 0,
  
  recordingState: 'idle',
  liveTranscript: "",
  socketStatus: 'disconnected',
  
  setUploading: (status) => set({ isUploading: status }),
  setUploadProgress: (progress) => set({ uploadProgress: progress }),
  setAudioFile: (file) => set({ audioFile: file }),
  setTranscription: (text) => set({ transcription: text }),
  setExtracting: (status) => set({ isExtracting: status }),
  setFeatures: (features) => set({ features }),
  setPredicting: (status) => set({ isPredicting: status }),
  setPrediction: (prediction) => set({ prediction }),
  setError: (error) => set({ error }),
  refreshFollowUpAlerts: () => set((state) => ({ followUpRefreshKey: state.followUpRefreshKey + 1 })),

  setRecordingState: (state) => set({ recordingState: state }),
  setLiveTranscript: (text) => set((state) => ({
    liveTranscript: typeof text === 'function' ? text(state.liveTranscript) : text
  })),
  setSocketStatus: (status) => set({ socketStatus: status }),
}))
