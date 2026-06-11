'use client';

import { motion } from 'framer-motion';
import { LineChart, ArrowUpRight, Loader2, Zap } from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';

export function PredictionSection() {
  const { isPredicting, prediction, error } = useAppStore();

  return (
    <section id="prediction" className="min-h-screen py-24 px-8 relative max-w-5xl mx-auto flex flex-col justify-center">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.6 }}
        className="mb-16 text-center"
      >
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-ai-cyan/10 mb-6">
          <LineChart className="w-8 h-8 text-ai-cyan" />
        </div>
        <h2 className="text-4xl md:text-5xl font-bold mb-4 text-gray-900 dark:text-white">
          Conversion <span className="text-ai-cyan">Prediction</span>
        </h2>
        <p className="text-lg text-gray-500 max-w-2xl mx-auto">
          Our XGBoost model analyzes the extracted features against historical data to predict the likelihood of conversion.
        </p>
        {error && error.includes('Prediction failed') && (
          <motion.p initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-red-500 text-sm mt-4 p-3 bg-red-500/10 rounded-xl inline-block">
            {error}
          </motion.p>
        )}
      </motion.div>

      {isPredicting ? (
        <div className="flex flex-col items-center justify-center h-64 gap-6">
          <Loader2 className="w-12 h-12 text-ai-cyan animate-spin" />
          <p className="text-xl font-medium animate-pulse text-ai-cyan">Running XGBoost Inference...</p>
        </div>
      ) : prediction ? (
        <div className="grid md:grid-cols-12 gap-8">
          {/* Main Gauge / Probability */}
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="md:col-span-5 glass-panel p-10 rounded-3xl flex flex-col items-center justify-center relative overflow-hidden"
          >
            <div className="absolute inset-0 bg-gradient-to-b from-ai-cyan/5 to-transparent" />
            <div className="relative w-48 h-48 mb-6">
              {/* Fake Gauge */}
              <svg className="w-full h-full transform -rotate-90" viewBox="0 0 100 100">
                <circle 
                  cx="50" cy="50" r="45" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="8" 
                  className="text-gray-200 dark:text-gray-800"
                />
                <motion.circle 
                  cx="50" cy="50" r="45" 
                  fill="none" 
                  stroke="currentColor" 
                  strokeWidth="8" 
                  strokeDasharray="283"
                  initial={{ strokeDashoffset: 283 }}
                  animate={{ strokeDashoffset: 283 - (283 * prediction.probability) }}
                  transition={{ duration: 1.5, ease: "easeOut" }}
                  className="text-ai-cyan"
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex flex-col items-center justify-center">
                <span className="text-5xl font-black text-gray-900 dark:text-white">{Math.round(prediction.probability * 100)}<span className="text-2xl text-gray-400">%</span></span>
                <span className="text-sm font-medium text-gray-500 mt-1">Probability</span>
              </div>
            </div>
            
            <div className="text-center mt-6">
              <h3 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                {prediction.probability >= 0.7 ? 'High Conversion Probability' : prediction.probability >= 0.4 ? 'Moderate Conversion Probability' : 'Low Conversion Probability'}
              </h3>
              <p className="text-gray-500">
                {prediction.probability >= 0.7 
                  ? 'Based on strong feature correlation and clear buying signals.'
                  : prediction.probability >= 0.4
                  ? 'Customer shows interest but requires further nurturing.'
                  : 'High hesitation or missing budget alignment detected.'}
              </p>
            </div>
          </motion.div>

          {/* Insights & Risk */}
          <div className="md:col-span-7 flex flex-col gap-6">
            <motion.div 
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
              className="glass-panel p-6 rounded-3xl flex items-center justify-between"
            >
              <div>
                <p className="text-sm text-gray-500 font-medium mb-1">Assessed Risk Level</p>
                <h4 className={`text-2xl font-bold flex items-center gap-2 ${prediction.risk === 'Low' ? 'text-green-500' : prediction.risk === 'Medium' ? 'text-yellow-500' : 'text-red-500'}`}>
                  {prediction.risk} Risk <ArrowUpRight className="w-5 h-5" />
                </h4>
              </div>
              <div className={`w-16 h-16 rounded-full flex items-center justify-center ${prediction.risk === 'Low' ? 'bg-green-500/10' : prediction.risk === 'Medium' ? 'bg-yellow-500/10' : 'bg-red-500/10'}`}>
                <Zap className={`w-8 h-8 ${prediction.risk === 'Low' ? 'text-green-500' : prediction.risk === 'Medium' ? 'text-yellow-500' : 'text-red-500'}`} />
              </div>
            </motion.div>

            <motion.div 
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.4 }}
              className="glass-panel p-6 rounded-3xl border-l-4 border-l-ai-cyan"
            >
              <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Reasons for Prediction</h4>
              <ul className="space-y-4">
                {prediction.insights.map((insight, i) => (
                  <motion.li 
                    key={i}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.5 + (i * 0.1) }}
                    className="flex items-start gap-3 p-3 rounded-xl bg-gray-50 dark:bg-gray-900/50"
                  >
                    <div className="w-6 h-6 rounded-full bg-ai-cyan/20 flex items-center justify-center shrink-0 mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-ai-cyan" />
                    </div>
                    <span className="text-gray-700 dark:text-gray-300">{insight}</span>
                  </motion.li>
                ))}
              </ul>
            </motion.div>

            <motion.div 
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.6 }}
              className="glass-panel p-6 rounded-3xl border-l-4 border-l-ai-purple"
            >
              <h4 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Conversion Playbook (Next Steps)</h4>
              <ul className="space-y-4">
                {prediction.nextSteps?.map((step, i) => (
                  <motion.li 
                    key={i}
                    initial={{ opacity: 0, x: 10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.7 + (i * 0.1) }}
                    className="flex items-start gap-3 p-3 rounded-xl bg-ai-purple/5"
                  >
                    <div className="w-6 h-6 rounded-full bg-ai-purple/20 flex items-center justify-center shrink-0 mt-0.5">
                      <div className="w-2 h-2 rounded-full bg-ai-purple" />
                    </div>
                    <span className="text-gray-700 dark:text-gray-300 font-medium">{step}</span>
                  </motion.li>
                ))}
              </ul>
            </motion.div>

          </div>
        </div>
      ) : (
        <div className="flex items-center justify-center h-64 glass-panel rounded-3xl border-dashed">
          <p className="text-gray-500">Run the extraction model first to see predictions.</p>
        </div>
      )}
    </section>
  );
}
