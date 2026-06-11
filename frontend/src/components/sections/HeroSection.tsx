'use client';

import { motion } from 'framer-motion';
import { ArrowRight, Sparkles } from 'lucide-react';

export function HeroSection() {
  return (
    <section id="hero" className="relative min-h-screen flex items-center justify-center overflow-hidden py-20 px-8">
      {/* Background Effects */}
      <div className="absolute inset-0 z-0">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-ai-blue/20 rounded-full blur-[128px] animate-pulse-glow" />
        <div className="absolute bottom-1/4 right-1/4 w-[500px] h-[500px] bg-ai-cyan/10 rounded-full blur-[128px]" />
      </div>

      <div className="absolute inset-0 z-0 flex items-center justify-center opacity-30 pointer-events-none">
        <img 
          src="/hero-3d.png" 
          alt="Background AI Sphere" 
          className="w-full max-w-[800px] h-auto object-contain animate-pulse-glow mix-blend-luminosity"
        />
      </div>

      <div className="relative z-10 max-w-5xl mx-auto text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border border-ai-blue/30 mb-8"
        >
          <Sparkles className="w-4 h-4 text-ai-blue" />
          <span className="text-sm font-medium bg-clip-text text-transparent bg-gradient-to-r from-ai-blue to-ai-purple">
            Next-Gen Audio Intelligence
          </span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.2, ease: "easeOut" }}
          className="text-6xl md:text-8xl font-black tracking-tight mb-8 text-gray-900 dark:text-white leading-[1.1]"
        >
          Transform Conversations into <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-ai-blue via-ai-purple to-ai-cyan">
            Predictive Intelligence
          </span>
        </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4, ease: "easeOut" }}
            className="text-xl md:text-2xl text-gray-600 dark:text-gray-400 max-w-3xl mx-auto mb-12 font-light"
          >
            Upload your sales calls, extract high-value features, and predict conversion outcomes with extreme accuracy using our cutting-edge AI models.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.6 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
          <button 
            onClick={() => document.getElementById('upload')?.scrollIntoView({ behavior: 'smooth' })}
            className="group relative inline-flex items-center justify-center px-8 py-4 font-bold text-white transition-all duration-200 bg-ai-blue border border-transparent rounded-full hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-ai-blue"
          >
            Start Analyzing Now
            <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
            <div className="absolute inset-0 h-full w-full rounded-full group-hover:animate-ping opacity-20 bg-white" />
          </button>
          
          <button 
            onClick={() => document.getElementById('prediction')?.scrollIntoView({ behavior: 'smooth' })}
            className="inline-flex items-center justify-center px-8 py-4 font-semibold text-gray-900 dark:text-white transition-all duration-200 glass rounded-full hover:bg-white/40 dark:hover:bg-white/10"
          >
            View Predictions
          </button>
          </motion.div>
      </div>      
      {/* Scroll Indicator */}
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.5, duration: 1 }}
        className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
      >
        <span className="text-xs text-gray-400 uppercase tracking-widest">Scroll</span>
        <div className="w-[1px] h-12 bg-gradient-to-b from-gray-400 to-transparent" />
      </motion.div>
    </section>
  );
}
