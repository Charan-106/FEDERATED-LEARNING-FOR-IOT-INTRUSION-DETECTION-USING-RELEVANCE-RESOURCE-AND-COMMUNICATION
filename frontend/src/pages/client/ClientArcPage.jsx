// Client ARC Results Page - Multi-Criteria Evaluation Breakdown
// Section 5.E: Explains Relevance, Resource, Communication components and shows personal scores

import { useEffect, useState } from 'react'
import {
  BarChart3,
  Award,
  Cpu,
  Wifi,
  Database,
  HelpCircle,
  TrendingUp,
  RefreshCw,
  Info
} from 'lucide-react'
import { clientApi } from '../../services/api'

export default function ClientArcPage() {
  const [arcHistory, setArcHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    loadArcScores()
  }, [])

  function loadArcScores() {
    setLoading(true)
    clientApi.getArcHistory()
      .then((data) => setArcHistory(data.arcScores || []))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }

  const latestScore = arcHistory[0] || null

  if (loading) {
    return <div className="card text-center py-12 text-gray-500">Loading your ARC evaluation scores...</div>
  }

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <BarChart3 className="text-arc-600" size={26} />
            ARC Evaluation & Client Importance
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Understanding your multi-criteria client score: Relevance, Resource, and Communication.
          </p>
        </div>

        <button
          onClick={loadArcScores}
          className="btn-secondary self-start sm:self-auto flex items-center gap-1.5 text-xs py-2 px-3"
        >
          <RefreshCw size={14} /> Refresh Scores
        </button>
      </div>

      {error && (
        <div className="card bg-red-50 dark:bg-red-900/20 text-red-600 p-4 text-sm">
          {error}
        </div>
      )}

      {/* Latest Score Spotlight */}
      {latestScore ? (
        <div className="card bg-gradient-to-br from-arc-50 via-white to-purple-50 dark:from-gray-800 dark:via-gray-800 dark:to-gray-900 border-2 border-arc-300 dark:border-arc-700 shadow-md">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div>
              <div className="flex items-center gap-2">
                <span className="badge-purple text-xs font-semibold uppercase">Latest Round #{latestScore.round?.roundNumber || '1'}</span>
                <span className="text-xs text-gray-400 font-mono">Formula v{latestScore.formulaVersion}</span>
              </div>
              <h2 className="text-2xl font-black text-gray-900 dark:text-white mt-2 flex items-center gap-2">
                <Award className="text-arc-600" size={28} />
                Overall ARC Score: {latestScore.overallArcScore?.toFixed(1)} / 100
              </h2>
              <p className="text-sm text-gray-600 dark:text-gray-300 mt-1 max-w-xl">
                Your model update was evaluated using the weighted formula:
                <code className="ml-1 bg-white dark:bg-gray-700 px-1.5 py-0.5 rounded text-xs font-mono text-arc-600">
                  ARC = 0.40·R + 0.30·S + 0.30·C
                </code>
              </p>
            </div>

            {/* Component Summary Pills */}
            <div className="grid grid-cols-3 gap-3 min-w-[320px]">
              <div className="bg-white dark:bg-gray-800 p-3 rounded-xl border border-blue-200 dark:border-blue-900 text-center">
                <div className="text-xs text-blue-600 font-bold uppercase tracking-wider">Relevance</div>
                <div className="text-2xl font-black text-blue-700 dark:text-blue-400 mt-1">
                  {latestScore.relevanceScore?.toFixed(1)}
                </div>
                <div className="text-[10px] text-gray-400">Weight: 40%</div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded-xl border border-green-200 dark:border-green-900 text-center">
                <div className="text-xs text-green-600 font-bold uppercase tracking-wider">Resource</div>
                <div className="text-2xl font-black text-green-700 dark:text-green-400 mt-1">
                  {latestScore.resourceScore?.toFixed(1)}
                </div>
                <div className="text-[10px] text-gray-400">Weight: 30%</div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded-xl border border-purple-200 dark:border-purple-900 text-center">
                <div className="text-xs text-purple-600 font-bold uppercase tracking-wider">Communication</div>
                <div className="text-2xl font-black text-purple-700 dark:text-purple-400 mt-1">
                  {latestScore.communicationScore?.toFixed(1)}
                </div>
                <div className="text-[10px] text-gray-400">Weight: 30%</div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="card text-center py-12">
          <Award size={36} className="mx-auto text-gray-300 dark:text-gray-600 mb-2" />
          <h3 className="font-semibold text-gray-700 dark:text-gray-200">No ARC Evaluation Yet</h3>
          <p className="text-sm text-gray-500 mt-1">
            Submit a local training update to receive your first ARC evaluation.
          </p>
        </div>
      )}

      {/* Explanatory Cards: What does each score mean? (Master Prompt requirement) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Relevance Card */}
        <div className="card border-t-4 border-t-blue-500 space-y-3">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-blue-50 dark:bg-blue-900/30 text-blue-600 rounded-lg">
              <Database size={20} />
            </div>
            <div>
              <h3 className="font-bold text-gray-900 dark:text-white">Relevance (R)</h3>
              <p className="text-xs text-gray-400">Data Quality & Diversity</p>
            </div>
          </div>
          <p className="text-xs text-gray-600 dark:text-gray-300">
            Measures the quality and diversity of your local training data without revealing raw feature vectors.
          </p>
          <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg text-xs space-y-1.5 font-mono">
            <div className="text-blue-600 font-bold">R_i = 0.60·Q_i + 0.40·D_i</div>
            <div className="text-gray-500">• Q_i: Sample completeness & absence of nulls/infinities</div>
            <div className="text-gray-500">• D_i: Class distribution entropy across attack categories</div>
          </div>
        </div>

        {/* Resource Card */}
        <div className="card border-t-4 border-t-green-500 space-y-3">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-green-50 dark:bg-green-900/30 text-green-600 rounded-lg">
              <Cpu size={20} />
            </div>
            <div>
              <h3 className="font-bold text-gray-900 dark:text-white">Resource (S)</h3>
              <p className="text-xs text-gray-400">Compute & Hardware Headroom</p>
            </div>
          </div>
          <p className="text-xs text-gray-600 dark:text-gray-300">
            Evaluates your device's capacity to execute local SGD epochs without causing thermal or memory bottlenecks.
          </p>
          <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg text-xs space-y-1.5 font-mono">
            <div className="text-green-600 font-bold">S_i = 0.40·M_i + 0.40·P_i + 0.20·A_i</div>
            <div className="text-gray-500">• M_i: Available RAM headroom percentage</div>
            <div className="text-gray-500">• P_i: CPU/GPU idle percentage during training</div>
            <div className="text-gray-500">• A_i: Sample throughput rate per second</div>
          </div>
        </div>

        {/* Communication Card */}
        <div className="card border-t-4 border-t-purple-500 space-y-3">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-purple-50 dark:bg-purple-900/30 text-purple-600 rounded-lg">
              <Wifi size={20} />
            </div>
            <div>
              <h3 className="font-bold text-gray-900 dark:text-white">Communication (C)</h3>
              <p className="text-xs text-gray-400">Throughput & Link Stability</p>
            </div>
          </div>
          <p className="text-xs text-gray-600 dark:text-gray-300">
            Assesses the network link between your node and the central server to minimize aggregation latency.
          </p>
          <div className="bg-gray-50 dark:bg-gray-800 p-3 rounded-lg text-xs space-y-1.5 font-mono">
            <div className="text-purple-600 font-bold">C_i = 0.50·B_i + 0.30·L_i + 0.20·T_i</div>
            <div className="text-gray-500">• B_i: Upload transfer rate (normalized to 10 Mbps)</div>
            <div className="text-gray-500">• L_i: Round-trip ping latency (lower is higher score)</div>
            <div className="text-gray-500">• T_i: Packet transmission reliability ratio</div>
          </div>
        </div>
      </div>

      {/* Historical ARC Evaluation Vertical Cards */}
      {arcHistory.length > 0 && (
        <div className="space-y-4">
          <h2 className="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <TrendingUp size={20} className="text-arc-600" />
            ARC Score History Across Rounds
          </h2>

          <div className="space-y-3">
            {arcHistory.map((score, idx) => (
              <div
                key={score.id || idx}
                className="card border border-gray-200 dark:border-gray-700 hover:border-arc-300 transition-all"
              >
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-gray-100 dark:border-gray-800 pb-3">
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-gray-900 dark:text-white text-sm">
                      Round #{score.round?.roundNumber || '1'}
                    </span>
                    <span className="badge-purple font-mono text-xs">
                      Score: {score.overallArcScore?.toFixed(2)}
                    </span>
                  </div>
                  <span className="text-xs text-gray-400">
                    {new Date(score.calculatedAt).toLocaleString()}
                  </span>
                </div>

                <div className="grid grid-cols-3 gap-4 pt-3 text-center">
                  <div>
                    <span className="text-xs text-gray-500">Relevance (40%)</span>
                    <div className="text-base font-bold text-blue-600 mt-0.5">
                      {score.relevanceScore?.toFixed(1)} / 100
                    </div>
                  </div>
                  <div>
                    <span className="text-xs text-gray-500">Resource (30%)</span>
                    <div className="text-base font-bold text-green-600 mt-0.5">
                      {score.resourceScore?.toFixed(1)} / 100
                    </div>
                  </div>
                  <div>
                    <span className="text-xs text-gray-500">Communication (30%)</span>
                    <div className="text-base font-bold text-purple-600 mt-0.5">
                      {score.communicationScore?.toFixed(1)} / 100
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
