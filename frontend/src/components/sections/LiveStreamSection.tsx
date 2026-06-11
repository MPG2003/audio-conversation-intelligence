'use client';

import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Mic, Square, Loader2, Activity, Radio, RadioTower } from 'lucide-react';
import { useAppStore } from '@/store/useAppStore';
import { apiService } from '@/services/api';

export function LiveStreamSection() {
  const { 
    recordingState, setRecordingState, 
    liveTranscript, setLiveTranscript,
    socketStatus, setSocketStatus,
    features,
    setTranscription, setExtracting, setFeatures, setPrediction, setError
  } = useAppStore();

  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [recordedBytes, setRecordedBytes] = useState(0);
  const audioChunksRef = useRef<Blob[]>([]);
  const sourceStreamRef = useRef<MediaStream | null>(null);
  const microphoneStreamRef = useRef<MediaStream | null>(null);
  const mixedAudioContextRef = useRef<AudioContext | null>(null);
  const stopInProgressRef = useRef(false);
  const transcriptEndRef = useRef<HTMLDivElement>(null);

  // Auto-scroll transcript
  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [liveTranscript]);

  const startCapture = async () => {
    try {
      setError(null);
      setLiveTranscript('');
      setTranscription('');
      setFeatures(null);
      setPrediction(null);
      setRecordedBytes(0);
      audioChunksRef.current = [];
      stopInProgressRef.current = false;

      // getDisplayMedia captures the meeting tab audio. getUserMedia captures the local speaker.
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: true, // required by API, but we'll ignore it
        audio: true
      });
      sourceStreamRef.current = stream;

      // Extract only the audio track
      const audioTrack = stream.getAudioTracks()[0];
      if (!audioTrack) {
        stream.getTracks().forEach((track) => track.stop());
        sourceStreamRef.current = null;
        throw new Error("No audio track found in the selected tab.");
      }

      let audioStream = new MediaStream([audioTrack]);
      try {
        const microphoneStream = await navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
          },
        });
        microphoneStreamRef.current = microphoneStream;

        const AudioContextClass = window.AudioContext || (window as any).webkitAudioContext;
        const audioContext = new AudioContextClass();
        const destination = audioContext.createMediaStreamDestination();
        audioContext.createMediaStreamSource(stream).connect(destination);
        audioContext.createMediaStreamSource(microphoneStream).connect(destination);
        mixedAudioContextRef.current = audioContext;
        audioStream = destination.stream;
      } catch (micError) {
        console.warn('Microphone capture unavailable; recording shared tab audio only.', micError);
      }

      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm';
      const recorder = new MediaRecorder(audioStream, { mimeType });

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
          setRecordedBytes((current) => current + event.data.size);
        }
      };

      recorder.onstop = () => {
        void finalizeRecording();
      };

      recorder.start(1000);
      setMediaRecorder(recorder);
      setRecordingState('recording');
      setSocketStatus('connected');

      // Stop handling if user clicks 'stop sharing' in browser
      audioTrack.onended = () => {
        stopCapture();
      };

    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Failed to capture browser audio. Please ensure you shared a tab with audio.');
      setRecordingState('idle');
      setSocketStatus('disconnected');
      stopCaptureDevices();
    }
  };

  const stopCaptureDevices = () => {
    sourceStreamRef.current?.getTracks().forEach(t => t.stop());
    sourceStreamRef.current = null;
    microphoneStreamRef.current?.getTracks().forEach(t => t.stop());
    microphoneStreamRef.current = null;
    void mixedAudioContextRef.current?.close();
    mixedAudioContextRef.current = null;
  };

  const stopCapture = () => {
    if (stopInProgressRef.current) return;
    stopInProgressRef.current = true;

    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stream.getAudioTracks().forEach((track) => {
        track.onended = null;
      });
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(t => t.stop());
      stopCaptureDevices();
    } else {
      setRecordingState('idle');
      setSocketStatus('disconnected');
      stopCaptureDevices();
    }
    setMediaRecorder(null);
    setRecordingState('processing');
  };

  const finalizeRecording = async () => {
    const chunks = audioChunksRef.current;
    if (!chunks.length) {
      setError('No audio was captured. Please share a tab with audio and try again.');
      setRecordingState('idle');
      setSocketStatus('disconnected');
      stopInProgressRef.current = false;
      return;
    }

    const recordedAt = new Date().toISOString().replace(/[:.]/g, '-');
    const audioBlob = new Blob(chunks, { type: 'audio/webm' });
    const audioFile = new File([audioBlob], `meeting-capture-${recordedAt}.webm`, { type: 'audio/webm' });

    setSocketStatus('disconnected');
    setRecordingState('analyzing');
    setExtracting(true);
    try {
      const result = await apiService.uploadAudio(audioFile);
      setLiveTranscript(result.transcription);
      setTranscription(result.transcription);
      setFeatures(result.features);
      setPrediction(result.prediction);
      useAppStore.getState().refreshFollowUpAlerts();
      setExtracting(false);
      setRecordingState('completed');
      stopInProgressRef.current = false;
      document.getElementById('extraction')?.scrollIntoView({ behavior: 'smooth' });
    } catch (err: any) {
      console.error(err);
      setExtracting(false);
      setRecordingState('idle');
      stopInProgressRef.current = false;
      setError(err?.message || 'Failed to process the finalized recording.');
    }
  };

  const capturedSizeMb = (recordedBytes / (1024 * 1024)).toFixed(2);

  return (
    <section id="live-stream" className="py-24 px-8 max-w-5xl mx-auto relative z-10">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
        className="glass-panel p-8 md:p-12 rounded-3xl relative overflow-hidden"
      >
        <div className="absolute top-0 right-0 w-64 h-64 bg-ai-blue/5 rounded-full blur-3xl pointer-events-none" />
        
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-8">
          <div>
            <h2 className="text-3xl font-bold text-gray-900 dark:text-white mb-2 flex items-center gap-3">
              <Activity className="w-8 h-8 text-ai-blue" />
              Live Browser Meeting Capture
            </h2>
            <p className="text-gray-500">
              Capture meeting tab audio and your microphone, then analyze one finalized recording.
            </p>
          </div>
          
          <div className="mt-6 md:mt-0 flex items-center gap-4">
            <div className={`flex items-center gap-2 text-sm font-medium px-3 py-1.5 rounded-full border ${
              socketStatus === 'connected' ? 'bg-green-500/10 border-green-500/20 text-green-500' :
              'bg-gray-500/10 border-gray-500/20 text-gray-500'
            }`}>
              {socketStatus === 'connected' ? <RadioTower className="w-4 h-4" /> : <Radio className="w-4 h-4" />}
              {socketStatus === 'connected' ? 'Capturing' : 'Recorder idle'}
            </div>

            {recordingState === 'idle' || recordingState === 'completed' ? (
              <button
                onClick={() => {
                  if (recordingState === 'completed') {
                    setLiveTranscript('');
                    setTranscription('');
                  }
                  startCapture();
                }}
                className="inline-flex items-center gap-2 px-6 py-3 bg-ai-blue text-white rounded-full font-bold hover:bg-blue-600 transition-colors shadow-lg shadow-ai-blue/20"
              >
                <Mic className="w-5 h-5" />
                {recordingState === 'completed' ? 'Start New Capture' : 'Start Capture'}
              </button>
            ) : recordingState === 'recording' ? (
              <button
                onClick={stopCapture}
                className="inline-flex items-center gap-2 px-6 py-3 bg-red-500 text-white rounded-full font-bold hover:bg-red-600 transition-colors shadow-lg shadow-red-500/20 animate-pulse"
              >
                <Square className="w-5 h-5" />
                Stop Capture
              </button>
            ) : (
              <button
                disabled
                className="inline-flex items-center gap-2 px-6 py-3 bg-gray-500 text-white rounded-full font-bold cursor-not-allowed opacity-70"
              >
                <Loader2 className="w-5 h-5 animate-spin" />
                {recordingState === 'processing' ? 'Finalizing Audio...' : 'Transcribing & Analyzing...'}
              </button>
            )}
          </div>
        </div>

        {/* Live Transcript Area */}
        <div className="relative rounded-2xl bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-800 p-6 min-h-[300px] max-h-[400px] overflow-y-auto mb-6">
          {recordingState === 'recording' && (
            <div className="absolute top-4 right-4 flex items-center gap-2 bg-red-500/10 text-red-500 px-3 py-1 rounded-full text-xs font-bold border border-red-500/20">
              <div className="w-2 h-2 rounded-full bg-red-500 animate-ping" />
              RECORDING TAB AUDIO
            </div>
          )}
          
          {features?.diarizedTranscript?.length ? (
            <div className="space-y-3">
              {features.diarizedTranscript.map((turn, index) => {
                const isCustomer = turn.speaker === 'Customer';
                return (
                  <div
                    key={`${turn.speaker}-${index}`}
                    className={`rounded-2xl border p-4 ${
                      isCustomer
                        ? 'bg-ai-cyan/10 border-ai-cyan/20'
                        : 'bg-ai-purple/10 border-ai-purple/20'
                    }`}
                  >
                    <div className={`text-xs font-bold uppercase tracking-wider mb-2 ${isCustomer ? 'text-ai-cyan' : 'text-ai-purple'}`}>
                      {turn.speaker}
                    </div>
                    <p className="text-gray-800 dark:text-gray-200 leading-relaxed font-medium">
                      {turn.text}
                    </p>
                  </div>
                );
              })}
              <div ref={transcriptEndRef} />
            </div>
          ) : liveTranscript ? (
            <div className="space-y-4">
              <p className="text-gray-800 dark:text-gray-200 leading-relaxed whitespace-pre-wrap font-medium">
                {liveTranscript}
              </p>
              <div ref={transcriptEndRef} />
            </div>
          ) : recordingState === 'recording' ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-500">
              <Activity className="w-12 h-12 mb-4 text-ai-blue animate-pulse" />
              <p className="font-medium text-gray-700 dark:text-gray-200">Capturing meeting audio locally</p>
              <p className="text-sm mt-2">{capturedSizeMb} MB buffered from tab and microphone. Transcript appears after you stop recording.</p>
            </div>
          ) : recordingState === 'processing' || recordingState === 'analyzing' ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-500">
              <Loader2 className="w-12 h-12 mb-4 text-ai-blue animate-spin" />
              <p className="font-medium text-gray-700 dark:text-gray-200">
                {recordingState === 'processing' ? 'Finalizing one complete WebM file' : 'Running Whisper and sales intelligence'}
              </p>
            </div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-gray-400">
              <Mic className="w-12 h-12 mb-4 opacity-20" />
              <p>Click Start Capture and select the browser tab containing your meeting.</p>
            </div>
          )}
        </div>
      </motion.div>
    </section>
  );
}
