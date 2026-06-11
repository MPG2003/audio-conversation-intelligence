'use client';

import { motion } from 'framer-motion';
import { BrainCircuit, Activity, DollarSign, Target, AlertTriangle, Loader2, ShieldCheck, UserRound, MessagesSquare, Volume2, FileText, ListChecks, CheckCircle } from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';
import { apiService } from '@/services/api';

const cardVariants = {
  hidden: { opacity: 0, y: 20 },
  visible: (i: number) => ({
    opacity: 1,
    y: 0,
    transition: {
      delay: i * 0.1,
      duration: 0.5,
      ease: "easeOut" as const
    }
  })
};

export function ExtractionSection() {
  const { isExtracting, features, setPredicting, error, setError } = useAppStore();

  const handlePredict = async () => {
    if (!features) return;

    setPredicting(true);
    try {
      const prediction = await apiService.predictConversion(features);
      useAppStore.getState().setPrediction(prediction);
      setPredicting(false);
      document.getElementById('prediction')?.scrollIntoView({ behavior: 'smooth' });
    } catch (err: any) {
      console.error(err);
      setPredicting(false);
      setError(err?.message || 'Prediction failed. Check that XGBoost API is running.');
    }
  };

  return (
    <section id="extraction" className="min-h-screen py-24 px-8 relative max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.6 }}
        className="mb-16 text-center"
      >
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-ai-purple/10 mb-6">
          <BrainCircuit className="w-8 h-8 text-ai-purple" />
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-4 text-gray-900 dark:text-white">
          Deep <span className="text-ai-purple">Feature Extraction</span>
        </h2>
        <p className="text-lg text-gray-500 max-w-2xl mx-auto">
          Our NLP models analyze the transcript to identify key linguistic patterns, emotional states, and buyer signals.
        </p>

        {features && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-6 inline-flex items-center gap-2 px-4 py-2 rounded-full border border-gray-200 dark:border-gray-800 glass">
            <span className="text-sm text-gray-500 font-medium">Extraction Engine:</span>
            <span className={`text-sm font-bold flex items-center gap-1.5 ${features.extractionProvider === 'llama' ? 'text-ai-purple' : 'text-orange-500'}`}>
              <div className={`w-2 h-2 rounded-full ${features.extractionProvider === 'llama' ? 'bg-ai-purple' : 'bg-orange-500'} animate-pulse`} />
              {features.extractionProvider === 'llama' ? 'LLaMA 3 (Groq)' : 'Local Fallback'}
            </span>
          </motion.div>
        )}

        {error && error.includes('Prediction failed') && (
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-red-500 text-sm mt-4 p-3 bg-red-500/10 rounded-xl inline-block">
            {error}
          </motion.p>
        )}
      </motion.div>

      {isExtracting ? (
        <div className="flex flex-col items-center justify-center h-64 gap-6">
          <Loader2 className="w-12 h-12 text-ai-purple animate-spin" />
          <p className="text-xl font-medium animate-pulse text-ai-purple">Extracting features with LLaMA 3...</p>
        </div>
      ) : features ? (
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Sentiment Score */}
          <motion.div 
            custom={0} variants={cardVariants} initial="hidden" whileInView="visible" viewport={{ once: true }}
            className="glass-panel p-6 rounded-3xl relative overflow-hidden group"
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-ai-blue/10 rounded-full blur-2xl -mr-10 -mt-10 transition-transform group-hover:scale-150" />
            <Activity className="w-8 h-8 text-ai-blue mb-4" />
            <p className="text-sm text-gray-500 font-medium mb-1">Sentiment Evaluation</p>
            <div className="flex items-end gap-2 mt-2">
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white leading-tight">{features.emotion}</h3>
            </div>
            {/* Progress Bar */}
            <div className="w-full h-2 bg-gray-200 dark:bg-gray-800 rounded-full mt-4 overflow-hidden">
              <motion.div 
                initial={{ width: 0 }} 
                animate={{ width: `${features.sentiment * 100}%` }} 
                transition={{ duration: 1, delay: 0.5 }}
                className="h-full bg-gradient-to-r from-ai-blue to-ai-cyan" 
              />
            </div>
          </motion.div>

          {/* Emotion */}
          <motion.div 
            custom={1} variants={cardVariants} initial="hidden" whileInView="visible" viewport={{ once: true }}
            className="glass-panel p-6 rounded-3xl relative overflow-hidden group"
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-ai-purple/10 rounded-full blur-2xl -mr-10 -mt-10 transition-transform group-hover:scale-150" />
            <Target className="w-8 h-8 text-ai-purple mb-4" />
            <p className="text-sm text-gray-500 font-medium mb-1">Dominant Emotion</p>
            <h3 className="text-3xl font-bold text-gray-900 dark:text-white mt-2">{features.emotion}</h3>
          </motion.div>

          {/* Buying Intent */}
          <motion.div 
            custom={2} variants={cardVariants} initial="hidden" whileInView="visible" viewport={{ once: true }}
            className="glass-panel p-6 rounded-3xl relative overflow-hidden group"
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-orange-500/10 rounded-full blur-2xl -mr-10 -mt-10 transition-transform group-hover:scale-150" />
            <DollarSign className="w-8 h-8 text-orange-500 mb-4" />
            <p className="text-sm text-gray-500 font-medium mb-1">Buying Intent</p>
            <h3 className="text-3xl font-bold text-gray-900 dark:text-white mt-2">{features.buyingIntent}</h3>
            {features.budgetDetected && (
              <div className="mt-4 inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-green-500/10 text-green-500 text-xs font-medium border border-green-500/20">
                <CheckCircle className="w-3 h-3" /> Budget Mentioned
              </div>
            )}
          </motion.div>

          {/* Extracted Signals */}
          <motion.div 
            custom={3} variants={cardVariants} initial="hidden" whileInView="visible" viewport={{ once: true }}
            className="glass-panel p-6 rounded-3xl lg:col-span-3 border-l-4 border-l-ai-cyan"
          >
            <div className="flex items-center gap-3 mb-4">
              <BrainCircuit className="w-6 h-6 text-ai-cyan" />
              <h3 className="text-xl font-semibold dark:text-white">Extracted Key Signals</h3>
            </div>
            <div className="flex gap-3 flex-wrap">
              {features.rawFeatures && features.rawFeatures.map((f, i) => (
                <div key={i} className="px-4 py-2 rounded-lg bg-ai-cyan/10 border border-ai-cyan/20 flex flex-col">
                  <span className="text-xs text-ai-cyan/70 uppercase tracking-wider font-bold mb-0.5">{f.label}</span>
                  <span className="text-ai-cyan font-medium text-base capitalize">{f.name}</span>
                </div>
              ))}
              {(!features.rawFeatures || features.rawFeatures.length === 0) && (
                <p className="text-gray-500">No specific products or budget signals detected.</p>
              )}
            </div>
          </motion.div>

          {/* Conversation Summary */}
          <motion.div
            custom={4} variants={cardVariants} initial="hidden" whileInView="visible" viewport={{ once: true }}
            className="glass-panel p-6 rounded-3xl lg:col-span-3 border-l-4 border-l-ai-blue"
          >
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-5">
              <div className="flex items-center gap-3">
                <FileText className="w-6 h-6 text-ai-blue" />
                <h3 className="text-xl font-semibold dark:text-white">Call Summary</h3>
              </div>
              {features.conversationSummary?.provider && (
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-ai-blue/10 border border-ai-blue/20 text-ai-blue text-xs font-bold w-fit">
                  {features.conversationSummary.provider.startsWith('llama') ? 'LLaMA Summary' : 'Local Summary'}
                  {typeof features.conversationSummary.confidence === 'number' && ` · ${Math.round(features.conversationSummary.confidence * 100)}%`}
                </div>
              )}
            </div>

            {features.conversationSummary ? (
              <div className="grid lg:grid-cols-[1.2fr_0.8fr] gap-5">
                <div>
                  <p className="text-sm text-gray-500 font-medium mb-2">Overview</p>
                  <p className="text-gray-800 dark:text-gray-100 leading-relaxed">{features.conversationSummary.overview}</p>

                  <div className="grid md:grid-cols-2 gap-3 mt-5">
                    <div className="rounded-xl bg-gray-50 dark:bg-gray-900/50 p-4">
                      <span className="block text-xs uppercase font-bold tracking-wider text-gray-500 mb-1">Customer Need</span>
                      <p className="text-sm text-gray-800 dark:text-gray-100">{features.conversationSummary.customerNeed}</p>
                    </div>
                    <div className="rounded-xl bg-gray-50 dark:bg-gray-900/50 p-4">
                      <span className="block text-xs uppercase font-bold tracking-wider text-gray-500 mb-1">Outcome</span>
                      <p className="text-sm text-gray-800 dark:text-gray-100">{features.conversationSummary.outcome}</p>
                    </div>
                  </div>
                </div>

                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <ListChecks className="w-4 h-4 text-ai-blue" />
                    <p className="text-sm font-semibold text-gray-700 dark:text-gray-200">Key Points</p>
                  </div>
                  <div className="space-y-2">
                    {features.conversationSummary.keyPoints.map((point, index) => (
                      <div key={`${point}-${index}`} className="flex gap-2 rounded-xl bg-ai-blue/10 border border-ai-blue/20 p-3">
                        <CheckCircle className="w-4 h-4 text-ai-blue mt-0.5 shrink-0" />
                        <p className="text-sm text-gray-700 dark:text-gray-200">{point}</p>
                      </div>
                    ))}
                  </div>
                  <div className="mt-4 rounded-xl bg-green-500/10 border border-green-500/20 p-4">
                    <span className="block text-xs uppercase font-bold tracking-wider text-green-500 mb-1">Next Action</span>
                    <p className="text-sm text-gray-800 dark:text-gray-100">{features.conversationSummary.nextAction}</p>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-gray-500 text-sm">No summary generated yet.</p>
            )}
          </motion.div>

          {/* Speaker classification and audio quality */}
          <motion.div
            custom={5} variants={cardVariants} initial="hidden" whileInView="visible" viewport={{ once: true }}
            className="glass-panel p-6 rounded-3xl lg:col-span-3 border-l-4 border-l-ai-purple"
          >
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-4">
              <div className="flex items-center gap-3">
                <MessagesSquare className="w-6 h-6 text-ai-purple" />
                <h3 className="text-xl font-semibold dark:text-white">Speaker Classification</h3>
              </div>
              {features.audioQuality && (
                <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-ai-purple/10 border border-ai-purple/20 text-ai-purple text-xs font-bold w-fit">
                  <Volume2 className="w-3.5 h-3.5" />
                  Audio {features.audioQuality.label} · {Math.round(features.audioQuality.confidence * 100)}%
                </div>
              )}
            </div>
            <div className="grid md:grid-cols-2 gap-3">
              {features.diarizedTranscript?.length ? features.diarizedTranscript.slice(0, 6).map((turn, index) => {
                const isCustomer = turn.speaker === 'Customer';
                return (
                  <div key={`${turn.speaker}-${index}`} className={`rounded-xl p-3 border ${isCustomer ? 'bg-ai-cyan/10 border-ai-cyan/20' : 'bg-ai-purple/10 border-ai-purple/20'}`}>
                    <span className={`block text-[10px] uppercase font-bold tracking-wider mb-1 ${isCustomer ? 'text-ai-cyan' : 'text-ai-purple'}`}>
                      {turn.speaker}
                    </span>
                    <p className="text-sm text-gray-700 dark:text-gray-200 line-clamp-3">{turn.text}</p>
                  </div>
                );
              }) : (
                <p className="text-gray-500 text-sm">No speaker turns available yet.</p>
              )}
            </div>
          </motion.div>

          {/* Privacy-safe customer profile */}
          <motion.div
            custom={6} variants={cardVariants} initial="hidden" whileInView="visible" viewport={{ once: true }}
            className="glass-panel p-6 rounded-3xl lg:col-span-3 border-l-4 border-l-green-500"
          >
            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-5">
              <div className="flex items-center gap-3">
                <ShieldCheck className="w-6 h-6 text-green-500" />
                <h3 className="text-xl font-semibold dark:text-white">Privacy-Safe Local Extraction</h3>
              </div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-green-500/10 border border-green-500/20 text-green-500 text-xs font-bold w-fit">
                {features.privacy?.redactionCount || 0} PII item{(features.privacy?.redactionCount || 0) === 1 ? '' : 's'} redacted before LLaMA
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <UserRound className="w-4 h-4 text-green-500" />
                  <p className="text-sm font-semibold text-gray-700 dark:text-gray-200">Structured Details</p>
                </div>
                <div className="flex gap-2 flex-wrap">
                  {features.privacy?.entities?.length ? features.privacy.entities.map((entity, index) => (
                    <div key={`${entity.type}-${index}`} className="px-3 py-2 rounded-lg bg-green-500/10 border border-green-500/20">
                      <span className="block text-[10px] text-green-500/80 uppercase font-bold tracking-wider">{entity.type.replaceAll('_', ' ')}</span>
                      <span className="text-sm text-green-600 dark:text-green-400 font-medium">{entity.value}</span>
                    </div>
                  )) : (
                    <p className="text-gray-500 text-sm">No sensitive details detected locally.</p>
                  )}
                </div>
              </div>

              <div>
                <p className="text-sm font-semibold text-gray-700 dark:text-gray-200 mb-3">Customer Behavioral Summary</p>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div className="rounded-xl bg-gray-50 dark:bg-gray-900/50 p-3">
                    <span className="block text-gray-500">Intent Signals</span>
                    <strong className="text-gray-900 dark:text-white">{features.customerBehaviorSummary?.intentSignals ?? 0}</strong>
                  </div>
                  <div className="rounded-xl bg-gray-50 dark:bg-gray-900/50 p-3">
                    <span className="block text-gray-500">Hesitation</span>
                    <strong className="text-gray-900 dark:text-white">{features.customerBehaviorSummary?.hesitationScore ?? 0}</strong>
                  </div>
                  <div className="rounded-xl bg-gray-50 dark:bg-gray-900/50 p-3">
                    <span className="block text-gray-500">Urgency</span>
                    <strong className="text-gray-900 dark:text-white">{features.customerBehaviorSummary?.urgencySignals ?? 0}</strong>
                  </div>
                  <div className="rounded-xl bg-gray-50 dark:bg-gray-900/50 p-3">
                    <span className="block text-gray-500">Customer Words</span>
                    <strong className="text-gray-900 dark:text-white">{features.customerBehaviorSummary?.wordCount ?? 0}</strong>
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Objections */}
          <motion.div 
            custom={7} variants={cardVariants} initial="hidden" whileInView="visible" viewport={{ once: true }}
            className="glass-panel p-6 rounded-3xl lg:col-span-3 border-l-4 border-l-red-500"
          >
            <div className="flex items-center gap-3 mb-4">
              <AlertTriangle className="w-6 h-6 text-red-500" />
              <h3 className="text-xl font-semibold dark:text-white">Detected Objections</h3>
            </div>
            <div className="flex gap-3 flex-wrap">
              {features.objections.map((obj, i) => (
                <div key={i} className="px-4 py-2 rounded-lg bg-red-500/10 border border-red-500/20 text-red-500 font-medium text-sm">
                  {obj}
                </div>
              ))}
              {features.objections.length === 0 && (
                <p className="text-gray-500">No major objections detected.</p>
              )}
            </div>
          </motion.div>

          <div className="lg:col-span-3 flex justify-center mt-8">
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1 }}
              onClick={handlePredict}
              className="group relative inline-flex items-center justify-center px-10 py-4 font-bold text-white transition-all duration-200 bg-gradient-to-r from-ai-purple to-ai-blue border border-transparent rounded-full shadow-[0_0_40px_rgba(191,90,242,0.4)] hover:shadow-[0_0_60px_rgba(191,90,242,0.6)] focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ai-purple"
            >
              Run Prediction Model
            </motion.button>
          </div>
        </div>
      ) : (
        <div className="flex items-center justify-center h-64 glass-panel rounded-3xl border-dashed">
          <p className="text-gray-500">Extract features from the conversation input to see results here.</p>
        </div>
      )}
    </section>
  );
}
