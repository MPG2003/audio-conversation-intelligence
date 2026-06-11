import { HeroSection } from '@/components/sections/HeroSection';
import { LiveStreamSection } from '@/components/sections/LiveStreamSection';
import { UploadSection } from '@/components/sections/UploadSection';
import { ConversationInputSection } from '@/components/sections/ConversationInputSection';
import { ExtractionSection } from '@/components/sections/ExtractionSection';
import { PredictionSection } from '@/components/sections/PredictionSection';
import { FollowUpAlertsSection } from '@/components/sections/FollowUpAlertsSection';

export default function Home() {
  return (
    <div className="flex flex-col relative w-full overflow-x-hidden">
      {/* Global Background Effects for smooth scrolling visual consistency */}
      <div className="fixed inset-0 z-[-1] pointer-events-none">
        <div className="absolute top-[20%] left-[10%] w-[600px] h-[600px] bg-ai-blue/5 rounded-full blur-[150px]" />
        <div className="absolute bottom-[20%] right-[10%] w-[600px] h-[600px] bg-ai-purple/5 rounded-full blur-[150px]" />
      </div>

      <HeroSection />
      
      <div className="w-full max-w-7xl mx-auto border-t border-gray-200/50 dark:border-gray-800/50 my-12" />
      
      <LiveStreamSection />
      
      <div className="w-full max-w-7xl mx-auto border-t border-gray-200/50 dark:border-gray-800/50 my-12" />
      
      <UploadSection />
      
      <div className="w-full max-w-7xl mx-auto border-t border-gray-200/50 dark:border-gray-800/50 my-12" />
      
      <ConversationInputSection />
      
      <div className="w-full max-w-7xl mx-auto border-t border-gray-200/50 dark:border-gray-800/50 my-12" />
      
      <ExtractionSection />
      
      <div className="w-full max-w-7xl mx-auto border-t border-gray-200/50 dark:border-gray-800/50 my-12" />
      
      <PredictionSection />

      <div className="w-full max-w-7xl mx-auto border-t border-gray-200/50 dark:border-gray-800/50 my-12" />

      <FollowUpAlertsSection />
      
      {/* Footer / Settings Section Placeholder */}
      <footer id="settings" className="w-full py-12 border-t border-gray-200 dark:border-gray-800 mt-24 bg-white/50 dark:bg-black/20 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-8 flex flex-col md:flex-row items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded bg-gradient-to-br from-ai-blue to-ai-cyan flex items-center justify-center">
              <span className="text-white font-bold text-xs">N</span>
            </div>
            <span className="font-semibold text-gray-900 dark:text-white">Nexus AI</span>
          </div>
          <p className="text-sm text-gray-500 mt-4 md:mt-0">© 2026 Nexus AI Platform. All rights reserved.</p>
          <div className="flex gap-4 mt-4 md:mt-0">
            <button className="text-sm text-gray-500 hover:text-ai-blue transition-colors">API Docs</button>
            <button className="text-sm text-gray-500 hover:text-ai-blue transition-colors">Settings</button>
          </div>
        </div>
      </footer>
    </div>
  );
}
