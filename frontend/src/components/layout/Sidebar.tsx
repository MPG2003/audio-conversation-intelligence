'use client';

import { motion } from 'framer-motion';
import { 
  Home, 
  UploadCloud, 
  MessageSquare, 
  BrainCircuit, 
  LineChart, 
  Bell,
  Settings 
} from 'lucide-react';
import { useEffect, useState } from 'react';
import { cn } from '@/lib/utils';

const navItems = [
  { id: 'hero', label: 'Home', icon: Home },
  { id: 'upload', label: 'Upload Audio', icon: UploadCloud },
  { id: 'input', label: 'Conversation', icon: MessageSquare },
  { id: 'extraction', label: 'Feature Extraction', icon: BrainCircuit },
  { id: 'prediction', label: 'Prediction', icon: LineChart },
  { id: 'follow-up-alerts', label: 'Follow-Up Alerts', icon: Bell },
  { id: 'settings', label: 'Settings', icon: Settings },
];

export function Sidebar() {
  const [activeSection, setActiveSection] = useState('hero');

  useEffect(() => {
    const handleScroll = () => {
      const sections = navItems.map(item => document.getElementById(item.id));
      const scrollPosition = window.scrollY + window.innerHeight / 3;

      for (const section of sections) {
        if (
          section && 
          section.offsetTop <= scrollPosition && 
          (section.offsetTop + section.offsetHeight) > scrollPosition
        ) {
          setActiveSection(section.id);
        }
      }
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const scrollToSection = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      window.scrollTo({
        top: element.offsetTop,
        behavior: 'smooth',
      });
      setActiveSection(id);
    }
  };

  return (
    <motion.aside 
      initial={{ x: -100, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
      className="fixed left-0 top-0 h-screen w-64 glass-panel z-50 flex flex-col justify-between py-8 px-4"
    >
      <div>
        <div className="flex items-center gap-3 px-4 mb-12">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-ai-blue to-ai-cyan flex items-center justify-center shadow-lg shadow-ai-blue/30">
            <BrainCircuit className="text-white w-6 h-6" />
          </div>
          <span className="font-bold text-xl tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-gray-900 to-gray-600 dark:from-white dark:to-gray-400">
            Nexus AI
          </span>
        </div>

        <nav className="flex flex-col gap-2">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeSection === item.id;
            
            return (
              <button
                key={item.id}
                onClick={() => scrollToSection(item.id)}
                className={cn(
                  "relative flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all duration-300 group",
                  isActive 
                    ? "text-ai-blue dark:text-white" 
                    : "text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-white"
                )}
              >
                {isActive && (
                  <motion.div
                    layoutId="activeTab"
                    className="absolute inset-0 bg-ai-blue/10 dark:bg-white/10 rounded-lg shadow-sm border border-ai-blue/20 dark:border-white/10"
                    transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                  />
                )}
                <Icon className={cn("w-5 h-5 relative z-10 transition-colors", isActive ? "text-ai-blue dark:text-white" : "group-hover:text-ai-blue dark:group-hover:text-white")} />
                <span className="relative z-10">{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      <div className="px-4">
        <div className="p-4 rounded-xl bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 border border-gray-200 dark:border-gray-800">
          <p className="text-xs text-gray-500 dark:text-gray-400 font-medium mb-2">System Status</p>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span className="text-sm font-medium text-gray-900 dark:text-gray-200">All systems operational</span>
          </div>
        </div>
      </div>
    </motion.aside>
  );
}
