'use client';

import { motion } from 'framer-motion';
import { Sparkles, Edit3 } from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';
import { useEffect, useState } from 'react';
import { apiService } from '@/services/api';

export function ConversationInputSection() {
  const { transcription, setTranscription, setExtracting, error, setError } = useAppStore();
  const [localText, setLocalText] = useState('');

  // Sync with global state
  useEffect(() => {
    if (transcription) {
      setLocalText(transcription);
    }
  }, [transcription]);

  const handleAnalyze = async () => {
    if (!localText.trim()) return;
    
    setTranscription(localText);
    setExtracting(true);
    
    try {
      const response = await apiService.extractFeatures(localText);
      setTranscription(response.transcription);
      useAppStore.getState().setFeatures(response.features);
      useAppStore.getState().setPrediction(null);
      useAppStore.getState().refreshFollowUpAlerts();
      setExtracting(false);
      document.getElementById('extraction')?.scrollIntoView({ behavior: 'smooth' });
    } catch (err: any) {
      console.error(err);
      setExtracting(false);
      setError(err?.message || 'Backend feature extraction failed. Check LLaMA 3 connection.');
    }
  };

  return (
    <section id="input" className="min-h-[80vh] py-24 px-8 relative max-w-5xl mx-auto flex flex-col justify-center">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.6 }}
        className="mb-12 flex items-center gap-4"
      >
        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-ai-purple to-ai-blue flex items-center justify-center shadow-lg">
          <Edit3 className="w-6 h-6 text-white" />
        </div>
        <div>
          <h2 className="text-3xl font-bold text-gray-900 dark:text-white">Conversation Transcript</h2>
          <p className="text-gray-500">Edit the AI-generated transcript or paste your own text.</p>
          {error && error.includes('feature extraction') && (
            <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-red-500 text-sm mt-2">
              {error}
            </motion.p>
          )}
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6, delay: 0.2 }}
        className="relative"
      >
        <div className="absolute -inset-1 bg-gradient-to-r from-ai-blue via-ai-purple to-ai-cyan rounded-3xl blur opacity-20" />
        
        <div className="relative glass-panel rounded-3xl p-2 flex flex-col">
          <textarea
            value={localText}
            onChange={(e) => setLocalText(e.target.value)}
            placeholder="Start typing or upload audio to see transcription here..."
            className="w-full min-h-[300px] bg-transparent border-0 resize-none p-6 text-lg text-gray-800 dark:text-gray-200 focus:ring-0 focus:outline-none placeholder:text-gray-400"
          />
          
          <div className="flex items-center justify-between p-4 border-t border-gray-200 dark:border-gray-800/50">
            <span className="text-sm text-gray-500">
              {localText.split(/\s+/).filter(w => w.length > 0).length} words
            </span>
            
            <button 
              onClick={handleAnalyze}
              disabled={!localText.trim()}
              className="inline-flex items-center justify-center px-6 py-2.5 font-medium text-white transition-all duration-200 bg-ai-blue rounded-full hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ai-blue disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Sparkles className="w-4 h-4 mr-2" />
              Extract Features
            </button>
          </div>
        </div>
      </motion.div>
    </section>
  );
}
