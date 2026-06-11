'use client';

import { motion } from 'framer-motion';
import { UploadCloud, FileAudio, CheckCircle, Loader2 } from 'lucide-react';
import { useCallback, useState } from 'react';
import { useAppStore } from '@/store/useAppStore';
import { apiService } from '@/services/api';

export function UploadSection() {
  const [isDragging, setIsDragging] = useState(false);
  const { 
    isUploading, 
    uploadProgress, 
    audioFile, 
    setUploading, 
    setUploadProgress, 
    setAudioFile,
    setTranscription,
    setExtracting,
    error,
    setError
  } = useAppStore();

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFile = async (file: File) => {
    if (!file || !file.type.startsWith('audio/')) {
      alert('Please upload an audio file');
      return;
    }
    
    setAudioFile(file);
    setUploading(true);
    setUploadProgress(0);

    const interval = setInterval(() => {
      const current = useAppStore.getState().uploadProgress;
      if (current >= 95) {
        clearInterval(interval);
        setUploadProgress(95);
      } else {
        setUploadProgress(current + 5);
      }
    }, 100);

    try {
      const response = await apiService.uploadAudio(file);
      clearInterval(interval);
      setUploadProgress(100);
      setUploading(false);
      setTranscription(response.transcription);
      setExtracting(true);
      useAppStore.getState().setFeatures(response.features);
      useAppStore.getState().setPrediction(null);
      useAppStore.getState().refreshFollowUpAlerts();
      setExtracting(false);
      document.getElementById('extraction')?.scrollIntoView({ behavior: 'smooth' });

    } catch (err: any) {
      clearInterval(interval);
      setUploading(false);
      console.error(err);
      setError(err?.message || 'Backend transcription failed. Please ensure the Whisper API is running.');
    }
  };

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer.files;
    if (files?.length) {
      handleFile(files[0]);
    }
  }, []);

  return (
    <section id="upload" className="min-h-screen py-24 px-8 relative flex flex-col justify-center max-w-6xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true, margin: "-100px" }}
        transition={{ duration: 0.6 }}
        className="mb-16 text-center"
      >
        <h2 className="text-4xl md:text-5xl font-bold mb-4 text-gray-900 dark:text-white">
          Secure Audio <span className="text-ai-blue">Upload</span>
        </h2>
        <p className="text-lg text-gray-500 dark:text-gray-400">
          Drop your conversation audio here. Our pipeline instantly transcribes and pre-processes the data for analysis.
        </p>
        
        {error && (
          <motion.div 
            initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} 
            className="mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/20 text-red-500 text-sm font-medium inline-block mx-auto"
          >
            {error}
            <button onClick={() => setError(null)} className="ml-4 underline hover:text-red-400">Dismiss</button>
          </motion.div>
        )}
      </motion.div>

      <div className="grid md:grid-cols-2 gap-12">
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            className={`
              relative group flex flex-col items-center justify-center p-12 h-[400px]
              border-2 border-dashed rounded-3xl transition-all duration-300
              ${isDragging 
                ? 'border-ai-blue bg-ai-blue/5' 
                : 'border-gray-300 dark:border-gray-700 hover:border-ai-blue/50 glass'}
            `}
          >
            <input 
              type="file" 
              accept="audio/*" 
              className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              onChange={(e) => e.target.files && handleFile(e.target.files[0])}
            />
            
            <div className="w-20 h-20 mb-6 rounded-full glass border border-gray-200 dark:border-gray-800 flex items-center justify-center group-hover:scale-110 transition-transform duration-300">
              <UploadCloud className={`w-10 h-10 ${isDragging ? 'text-ai-blue' : 'text-gray-400'}`} />
            </div>
            
            <h3 className="text-xl font-semibold mb-2 text-gray-900 dark:text-white">Drag & Drop Audio</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 text-center max-w-xs mb-6">
              Supports MP3, WAV, M4A up to 50MB. Stereo or Mono channel.
            </p>
            
            <button className="px-6 py-2.5 rounded-full bg-gray-900 dark:bg-white text-white dark:text-gray-900 text-sm font-medium hover:scale-105 transition-transform">
              Browse Files
            </button>
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="flex flex-col justify-center"
        >
          {audioFile ? (
            <div className="glass-panel p-8 rounded-3xl relative overflow-hidden">
              {isUploading && (
                <div 
                  className="absolute bottom-0 left-0 h-1 bg-gradient-to-r from-ai-blue to-ai-cyan transition-all duration-300"
                  style={{ width: `${uploadProgress}%` }}
                />
              )}
              
              <div className="flex items-start gap-4 mb-8">
                <div className="w-12 h-12 rounded-xl bg-ai-blue/10 flex items-center justify-center shrink-0">
                  <FileAudio className="w-6 h-6 text-ai-blue" />
                </div>
                <div>
                  <h4 className="font-semibold text-gray-900 dark:text-white line-clamp-1">{audioFile.name}</h4>
                  <p className="text-sm text-gray-500">{(audioFile.size / (1024 * 1024)).toFixed(2)} MB • Audio File</p>
                </div>
                <div className="ml-auto">
                  {isUploading ? (
                    <Loader2 className="w-6 h-6 text-ai-blue animate-spin" />
                  ) : (
                    <CheckCircle className="w-6 h-6 text-green-500" />
                  )}
                </div>
              </div>

              <div className="space-y-4">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-500">Status</span>
                  <span className="font-medium text-gray-900 dark:text-white">
                    {isUploading ? 'Uploading & Transcribing...' : 'Ready for Analysis'}
                  </span>
                </div>
                {isUploading && (
                  <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
                    <div 
                      className="bg-ai-blue h-2 rounded-full transition-all duration-300" 
                      style={{ width: `${uploadProgress}%` }}
                    />
                  </div>
                )}
                
                {/* Fake waveform animation when processing */}
                <div className="flex items-center justify-center gap-1 h-12 mt-4">
                  {[...Array(20)].map((_, i) => (
                    <motion.div
                      key={i}
                      animate={{ height: isUploading ? [10, 40, 10] : 4 }}
                      transition={{ 
                        repeat: Infinity, 
                        duration: 1, 
                        delay: i * 0.05,
                        ease: "easeInOut"
                      }}
                      className={`w-1.5 rounded-full ${isUploading ? 'bg-ai-blue' : 'bg-gray-300 dark:bg-gray-700'}`}
                      style={{ height: 4 }}
                    />
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center p-12 glass border border-dashed border-gray-200 dark:border-gray-800 rounded-3xl opacity-50">
              <p className="text-center text-gray-500">Upload a file to see progress and details.</p>
            </div>
          )}
        </motion.div>
      </div>
    </section>
  );
}
