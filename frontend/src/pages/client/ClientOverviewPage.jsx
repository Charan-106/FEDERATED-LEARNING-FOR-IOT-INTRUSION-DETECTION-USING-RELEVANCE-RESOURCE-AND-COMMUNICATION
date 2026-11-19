// Client Overview Page - Edge node dashboard

import { useEffect, useState } from 'react'
import { Upload, Brain, BarChart3, FileStack } from 'lucide-react'
import { clientApi } from '../../services/api'

export default function ClientOverviewPage() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    clientApi.getDashboard()
      .then(setData)
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div className="text-gray-500">Loading dashboard...</div>
  if (error) return <div className="text-red-500">Error: {error}</div>

  const { clientProfile, totalSubmissions, latestRound, clientEligibility, activeModel, arcScoreSummary } = data || {}

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Welcome, {clientProfile?.name}
        </h1>
        <p className="text-sm text-gray-500 mt-1">
          Client ID: <span className="font-mono text-arc-600">{clientProfile?.clientId}</span>
        </p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card text-center">
          <div className="inline-flex items-center justify-center w-10 h-10 bg-arc-100 dark:bg-arc-900/40 rounded-xl mb-3">
            <FileStack size={18} className="text-arc-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">{totalSubmissions ?? 0}</div>
          <div className="text-sm text-gray-500 mt-1">Total Submissions</div>
        </div>

        <div className="card text-center">
          <div className="inline-flex items-center justify-center w-10 h-10 bg-green-100 dark:bg-green-900/40 rounded-xl mb-3">
            <BarChart3 size={18} className="text-green-600" />
          </div>
          <div className="text-2xl font-bold text-gray-900 dark:text-white">
            {arcScoreSummary?.average != null ? arcScoreSummary.average.toFixed(1) : '—'}
          </div>
          <div className="text-sm text-gray-500 mt-1">Avg ARC Score</div>
        </div>

        <div className="card text-center">
          <div className="inline-flex items-center justify-center w-10 h-10 bg-purple-100 dark:bg-purple-900/40 rounded-xl mb-3">
            <Brain size={18} className="text-purple-600" />
          </div>
          <div className="text-sm font-semibold text-gray-900 dark:text-white truncate">
            {activeModel?.version ?? 'N/A'}
          </div>
          <div className="text-sm text-gray-500 mt-1">Active Model</div>
        </div>

        <div className="card text-center">
          <div className="inline-flex items-center justify-center w-10 h-10 bg-yellow-100 dark:bg-yellow-900/40 rounded-xl mb-3">
            <Upload size={18} className="text-yellow-600" />
          </div>
          <div className="text-sm font-semibold text-gray-900 dark:text-white">
            {latestRound ? `Round #${latestRound.roundNumber}` : 'None'}
          </div>
          <div className="text-sm text-gray-500 mt-1">
            {latestRound?.status ?? 'No active round'}
          </div>
        </div>
      </div>

      {latestRound?.status === 'ACTIVE' && (
        <div className={`card border ${
          clientEligibility?.hasSubmitted
            ? 'bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-700'
            : clientEligibility && !clientEligibility.isTargeted
            ? 'bg-gray-50 dark:bg-gray-800/40 border-gray-200 dark:border-gray-700'
            : 'bg-arc-50 dark:bg-arc-900/20 border-arc-200 dark:border-arc-700'
        }`}>
          <h2 className="text-base font-semibold text-gray-900 dark:text-white">
            {clientEligibility?.hasSubmitted
              ? `✅ Round #${latestRound.roundNumber} — Update Submitted & Registered`
              : clientEligibility && !clientEligibility.isTargeted
              ? `ℹ️ Round #${latestRound.roundNumber} is ACTIVE (Not Targeted)`
              : `🔔 Round #${latestRound.roundNumber} is ACTIVE`}
          </h2>
          <p className="text-sm text-gray-600 dark:text-gray-300 mt-1">
            {clientEligibility?.hasSubmitted
              ? `Your local model update for Round #${latestRound.roundNumber} has been verified and registered. Awaiting server aggregation.`
              : clientEligibility && !clientEligibility.isTargeted
              ? `This client node is not designated to participate in Round #${latestRound.roundNumber}.`
              : 'You are eligible to participate. Train on the global model and submit your update via the Training tab.'}
          </p>
        </div>
      )}
    </div>
  )
}
