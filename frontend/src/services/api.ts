import axios from 'axios';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Custom retry interceptor for robust AI backend handling
apiClient.interceptors.response.use(undefined, async (error) => {
  const config = error.config as any;
  if (!config || !config.retry) {
    return Promise.reject(error);
  }
  
  config.retryCount = config.retryCount || 0;
  
  if (config.retryCount >= config.retry) {
    return Promise.reject(error);
  }
  
  config.retryCount += 1;
  const backoff = new Promise(resolve => {
    setTimeout(() => {
      resolve(null);
    }, config.retryDelay || 2000);
  });
  
  await backoff;
  return apiClient(config);
});

export type BackendAnalysis = {
  transcript: string;
  diarizedTranscript?: Array<{
    speaker: 'Customer' | 'Agent' | string;
    rawSpeaker?: string;
    text: string;
    start?: number | null;
    end?: number | null;
  }>;
  customerTranscript?: string;
  customerBehavioralTranscript?: string;
  agentTranscript?: string;
  privacy?: {
    entities: Array<{ type: string; value: string; source: string }>;
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
  rawFeatures?: Array<{
    name: string;
    label: string;
  }>;
  pipelineFeatures?: {
    sentiment_score?: number;
    confidence_score?: number;
    hesitation_score?: number;
    delay_flag?: number;
    feature_count?: number;
    brand_count?: number;
    interaction_length?: number;
    extraction_provider?: string;
  };
  prediction?: {
    prediction: number;
    probability: number;
    label: string;
    reasons: string[];
  };
  followUpAlerts?: FollowUpAlert[];
  products?: Array<{
    name: string;
    sentiment: string;
    score: number;
    confidence: number;
    context?: string;
  }>;
  summary?: {
    averageScore?: number;
    dominant?: string;
    totalProducts?: number;
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
  metadata?: {
    extractionQuality?: Record<string, number | string>;
    extractionProvider?: string;
  };
};

export type FollowUpAlert = {
  id: string;
  follow_up_required: boolean;
  customer_name: string;
  company_name: string;
  action_needed: string;
  priority: 'High' | 'Medium' | 'Low';
  reason: string;
  source_text: string;
  created_date: string;
  status: 'Pending' | 'Completed';
  source_name?: string;
  source_type?: string;
};

const toUiFeatures = (analysis: BackendAnalysis) => {
  const products = analysis.products || [];
  const reasons = analysis.prediction?.reasons || [];
  const rawFeatures = analysis.rawFeatures || [];
  // Extract explicit objections from LLaMA features first
  const explicitObjections = rawFeatures
    .filter((f) => f.label === 'OBJECTION' || f.label === 'OBJECTION_TYPE')
    .map((f) => f.name || (f as any).value);

  const productObjections = products
    .filter((product) => product.sentiment === 'negative')
    .map((product) => product.context || product.name);

  const allObjections = [...explicitObjections, ...productObjections].slice(0, 5);
  
  // Filter out positive sentiments from fallback reasons
  const fallbackReasons = reasons.filter((r) => 
    !r.toLowerCase().includes('positive') && 
    !r.toLowerCase().includes('buying intent')
  );
  const pipelineSentiment = analysis.pipelineFeatures?.sentiment_score;
  const modelLabel = analysis.prediction?.label || analysis.conversionScore?.label;

  const rawDominant = analysis.summary?.dominant || 'neutral';
  // If the backend returns our new rich labels (e.g., "Strong Buying Intent"), they will be spaced out strings.
  const formattedEmotion = rawDominant.includes(' ') 
    ? rawDominant.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
    : rawDominant.charAt(0).toUpperCase() + rawDominant.slice(1);

  const intentFeature = rawFeatures.find((f) => f.label === 'INTENT');
  const buyingIntentStr = intentFeature 
    ? (intentFeature.name || (intentFeature as any).value) 
    : (modelLabel ? modelLabel.charAt(0).toUpperCase() + modelLabel.slice(1) : (products.length > 0 ? 'Medium' : 'Unknown'));
  const extractionProviderValue =
    analysis.pipelineFeatures?.extraction_provider || analysis.metadata?.extractionProvider || 'llama';

  return {
    sentiment: Math.max(0, Math.min(1, ((pipelineSentiment ?? analysis.summary?.averageScore ?? 0) + 1) / 2)),
    emotion: formattedEmotion,
    buyingIntent: buyingIntentStr,
    budgetDetected: rawFeatures.some((feature) => feature.label === 'BUDGET') || products.some((product) => product.name.toLowerCase().includes('budget') || /\d/.test(product.name)),
    objections: allObjections.length ? allObjections : fallbackReasons,
    rawFeatures,
    extractionProvider: String(extractionProviderValue),
    diarizedTranscript: analysis.diarizedTranscript || [],
    privacy: analysis.privacy,
    customerBehaviorSummary: analysis.customerBehaviorSummary,
    conversationSummary: analysis.conversationSummary,
    conversionScore: analysis.conversionScore,
    audioQuality: analysis.audioQuality,
    // Store original prediction data for the next step
    _rawPrediction: analysis.prediction,
    _rawSummary: analysis.summary,
    _rawFeaturesCount: analysis.rawFeatures?.length || 0,
  };
};

const toUiPrediction = (analysis: BackendAnalysis) => {
  const probability = analysis.prediction?.probability ?? analysis.conversionScore?.probability ?? 0;
  const risk = probability >= 0.7 ? 'Low' : probability >= 0.4 ? 'Medium' : 'High';
  const productCount = analysis.rawFeatures?.length || analysis.summary?.totalProducts || analysis.products?.length || 0;

  const insights = analysis.prediction?.reasons?.length ? analysis.prediction.reasons : [
      `${productCount} sales signal${productCount === 1 ? '' : 's'} detected`,
      `Dominant sentiment is ${analysis.summary?.dominant || 'neutral'}`,
      analysis.conversionScore
        ? `Lead classified as ${analysis.conversionScore.label}`
        : 'Prediction model inferred from features',
    ];

  const nextSteps = [];
  if (risk === 'High') {
    nextSteps.push('Offer flexible payment options (e.g., No-Cost EMI) to lower the entry barrier.');
    nextSteps.push('Follow up within 24 hours specifically addressing their primary objection.');
  } else if (risk === 'Medium') {
    nextSteps.push('Highlight the long-term value and warranty of the product.');
    nextSteps.push('Share case studies or testimonials related to their specific use-case.');
  } else {
    nextSteps.push('Send the checkout link immediately to capitalize on high intent.');
    nextSteps.push('Attempt to upsell an extended warranty or premium accessories.');
  }

  if (insights.some(i => i.toLowerCase().includes('hesitant') || i.toLowerCase().includes('postponed'))) {
    nextSteps.unshift('Identify their exact bottleneck (budget vs feature) to clear hesitation.');
  }

  return {
    probability,
    risk,
    insights,
    nextSteps,
  };
};

export const apiService = {
  uploadAudio: async (file: File) => {
    const formData = new FormData();
    formData.append('audio', file);
    
    const response = await apiClient.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      // @ts-ignore
      retry: 2,
      retryDelay: 3000,
    });
    
    const jobId = response.data.job_id;
    if (!jobId) {
       // Fallback in case the backend wasn't fully restarted and returns the old response format directly
       const analysis = response.data as BackendAnalysis;
       return {
         analysis,
         transcription: analysis.transcript,
         features: toUiFeatures(analysis),
         prediction: toUiPrediction(analysis),
       };
    }

    let jobData = response.data;
    while (jobData.status === 'pending' || jobData.status === 'processing') {
      await new Promise(resolve => setTimeout(resolve, 2000));
      const jobRes = await apiClient.get(`/jobs/${jobId}`);
      jobData = jobRes.data;
    }

    if (jobData.status === 'failed') {
      throw new Error(jobData.error || 'Background audio processing failed');
    }

    const analysis = jobData.result as BackendAnalysis;
    return {
      analysis,
      transcription: analysis.transcript,
      features: toUiFeatures(analysis),
      prediction: toUiPrediction(analysis),
    };
  },

  extractFeatures: async (transcription: string) => {
    const response = await apiClient.post('/analyze', { text: transcription }, {
      // @ts-ignore
      retry: 1,
      retryDelay: 2000,
    });
    const analysis = response.data as BackendAnalysis;
    return {
      analysis,
      transcription: analysis.transcript,
      features: toUiFeatures(analysis),
      prediction: toUiPrediction(analysis),
    };
  },

  predictConversion: async (features: any) => {
    // The backend actually computes prediction during the upload/analyze steps.
    // We simulate a network delay here to maintain the premium UX animation flow.
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    return toUiPrediction({
      transcript: '',
      prediction: features._rawPrediction,
      summary: features._rawSummary,
      conversionScore: features.conversionScore,
    });
  },

  getFollowUpAlerts: async (filters?: { priority?: string; status?: string; customerName?: string }) => {
    const response = await apiClient.get('/follow-up-alerts', {
      params: {
        priority: filters?.priority || undefined,
        status: filters?.status || undefined,
        customer_name: filters?.customerName || undefined,
      },
    });
    return response.data.alerts as FollowUpAlert[];
  },

  updateFollowUpStatus: async (alertId: string, status: FollowUpAlert['status']) => {
    const response = await apiClient.patch(`/follow-up-alerts/${alertId}`, { status });
    return response.data.alert as FollowUpAlert;
  }
};

export default apiClient;
