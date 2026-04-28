import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Dashboard from './components/Dashboard/Dashboard'
import Upload from './components/VideoUploader/Upload'
import Results from './components/Analysis/Results'

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen">
        <nav className="bg-white dark:bg-gray-900 shadow-sm">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex justify-between h-16 items-center">
              <div className="flex items-center gap-3">
                <ShieldIcon className="w-8 h-8 text-primary-600" />
                <span className="text-xl font-bold text-gray-900 dark:text-white">
                  DeepFake Sentinel
                </span>
              </div>
              <div className="flex gap-6">
                <a href="/" className="text-gray-600 hover:text-primary-600 dark:text-gray-300">
                  Dashboard
                </a>
                <a href="/upload" className="text-gray-600 hover:text-primary-600 dark:text-gray-300">
                  Analyze
                </a>
              </div>
            </div>
          </div>
        </nav>

        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/upload" element={<Upload />} />
          <Route path="/results/:jobId" element={<Results />} />
        </Routes>
      </div>
    </BrowserRouter>
  )
}

function ShieldIcon({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
      <path d="M9 12l2 2 4-4" />
    </svg>
  )
}

export default App
