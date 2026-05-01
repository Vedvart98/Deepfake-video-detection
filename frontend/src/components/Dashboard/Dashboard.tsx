import { useQuery } from '@tanstack/react-query'
import { checkHealth, getModelInfo } from '@/services/api'

export default function Dashboard() {
  const healthQuery = useQuery({
    queryKey: ['health'],
    queryFn: checkHealth,
    refetchInterval: 30000,
  })
  
  const modelQuery = useQuery({
    queryKey: ['model'],
    queryFn: getModelInfo,
  })
  
  return (
    <div className="max-w-7xl mx-auto py-12 px-4">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
          Dashboard
        </h1>
        <p className="text-gray-600 dark:text-gray-400">
          Overview of the DeepFake Sentinel system
        </p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <StatusCard
          title="System Status"
          value={healthQuery.data?.status === 'healthy' ? 'Online' : 'Offline'}
          color={healthQuery.data?.status === 'healthy' ? 'success' : 'danger'}
        />
        <StatusCard
          title="GPU Available"
          value={healthQuery.data?.cuda_available ? 'Yes' : 'No'}
          color={healthQuery.data?.cuda_available ? 'success' : 'warning'}
        />
        <StatusCard
          title="Model"
          value={modelQuery.data?.name || 'Loading...'}
          color="primary"
        />
      </div>
      
      <div className="card">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          How It Works
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StepCard
            step={1}
            title="Upload"
            description="Upload a video file"
          />
          <StepCard
            step={2}
            title="Detect Faces"
            description="YOLOv8 identifies faces"
          />
          <StepCard
            step={3}
            title="Analyze"
            description="CNN + ViT hybrid analysis"
          />
          <StepCard
            step={4}
            title="Results"
            description="Get detailed report"
          />
        </div>
      </div>
      
      <div className="card mt-8">
        <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
          Quick Actions
        </h2>
        <div className="flex gap-4">
          <a
            href="/upload"
            className="btn-primary"
          >
            Analyze New Video
          </a>
        </div>
      </div>
    </div>
  )
}

function StatusCard({ title, value, color }: { title: string; value: string; color: 'success' | 'danger' | 'warning' | 'primary' }) {
  const colors = {
    success: 'text-success-600 bg-success-100 dark:bg-success-900/30',
    danger: 'text-danger-600 bg-danger-100 dark:bg-danger-900/30',
    warning: 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/30',
    primary: 'text-primary-600 bg-primary-100 dark:bg-primary-900/30',
  }
  
  return (
    <div className="card">
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">{title}</p>
      <p className={`text-2xl font-bold ${colors[color].split(' ')[0]}`}>
        {value}
      </p>
    </div>
  )
}

function StepCard({ step, title, description }: { step: number; title: string; description: string }) {
  return (
    <div className="flex items-start gap-3">
      <div className="w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-600 flex items-center justify-center font-bold text-sm flex-shrink-0">
        {step}
      </div>
      <div>
        <h3 className="font-medium text-gray-900 dark:text-white">{title}</h3>
        <p className="text-sm text-gray-500 dark:text-gray-400">{description}</p>
      </div>
    </div>
  )
}
