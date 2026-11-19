// Client Submissions Page - Vertically arranged submission history
// Section 5.D: Strict vertical card arrangement for edge client model submissions

import { useEffect, useState } from 'react'
import { FileStack, CheckCircle, Clock, XCircle, GitMerge, Award, ChevronDown, ChevronUp, RefreshCw } from 'lucide-react'
import { clientApi } from '../../services/api'

export default function ClientSubmissionsPage() {
  const [submissions, setSubmissions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [expandedId, setExpandedId] = useState(null)

  useEffect(() => {
    loadSubmissions()
  }, [])

  function loadSubmissions() {
    setLoading(true)
    clientApi.getSubmissions()
      .then((data) => setSubmissions(data.submissions || []))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }

  function getStatusBadge(status) {
    switch (status) {
      case 'ACCEPTED':
        return <span className="badge-green flex items-center gap-1"><CheckCircle size={12} /> Accepted</span>
      case 'REJECTED':
        return <span className="badge-red flex items-center gap-1"><XCircle size={12} /> Rejected</span>
      case 'AGGREGATED':
        return <span className="badge-purple flex items-center gap-1"><GitMerge size={12} /> Aggregated</span>
      case 'PENDING':
      default:
        return <span className="badge-yellow flex items-center gap-1"><Clock size={12} /> Pending Review</span>
    }
  }

  if (loading) {
    return <div className="card text-center py-12 text-gray-500">Loading your submissions...</div>
  }

  return (
    <div className="space-y-6 max-w-4xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <FileStack className="text-arc-600" size={26} />
            My Model Submissions
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            History of local PyTorch checkpoint updates submitted to the federated coordinator.
          </p>
        </div>

        <button
          onClick={loadSubmissions}
          className="btn-secondary self-start sm:self-auto flex items-center gap-1.5 text-xs py-2 px-3"
        >
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {error && (
        <div className="card bg-red-50 dark:bg-red-900/20 text-red-600 p-4 text-sm">
          {error}
        </div>
      )}

      {/* Submissions Vertical List (Mandatory Vertical Card Arrangement) */}
      {submissions.length === 0 ? (
        <div className="card text-center py-16">
          <FileStack size={40} className="mx-auto text-gray-300 dark:text-gray-600 mb-3" />
          <h3 className="text-base font-semibold text-gray-700 dark:text-gray-200">No Submissions Yet</h3>
          <p className="text-sm text-gray-500 mt-1">
            Complete a local training session in the Training tab to submit your first update.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {submissions.map((sub) => {
            const isExpanded = expandedId === sub.id
            const arc = sub.arcScores?.[0]

            return (
              <div
                key={sub.id}
                className="card border border-gray-200 dark:border-gray-700 hover:border-arc-300 dark:hover:border-arc-700 transition-all shadow-sm"
              >
                {/* Main Card Header */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-gray-100 dark:border-gray-800 pb-3">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-mono text-sm font-bold text-gray-900 dark:text-white">
                      Submission #{sub.id.substring(0, 8)}
                    </span>
                    <span className="badge-blue text-xs font-semibold">
                      Round #{sub.round?.roundNumber || '1'}
                    </span>
                    {getStatusBadge(sub.status)}
                  </div>

                  <span className="text-xs text-gray-400">
                    {new Date(sub.submittedAt).toLocaleString()}
                  </span>
                </div>

                {/* Primary Metrics Row */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 py-3">
                  <div>
                    <span className="text-xs text-gray-500">Local Accuracy</span>
                    <p className="text-base font-bold text-green-600">
                      {sub.localAccuracy != null ? `${sub.localAccuracy}%` : '—'}
                    </p>
                  </div>

                  <div>
                    <span className="text-xs text-gray-500">Local Loss</span>
                    <p className="text-base font-bold text-gray-800 dark:text-gray-200">
                      {sub.localLoss != null ? sub.localLoss.toFixed(4) : '—'}
                    </p>
                  </div>

                  <div>
                    <span className="text-xs text-gray-500">Samples Trained</span>
                    <p className="text-base font-bold text-purple-600">
                      {sub.samplesTrained != null ? sub.samplesTrained.toLocaleString() : '—'}
                    </p>
                  </div>

                  <div>
                    <span className="text-xs text-gray-500">Overall ARC Score</span>
                    <p className="text-base font-bold text-arc-600 flex items-center gap-1">
                      <Award size={15} />
                      {sub.arcScore != null ? sub.arcScore.toFixed(1) : arc?.overallArcScore?.toFixed(1) || 'Pending'}
                    </p>
                  </div>
                </div>

                {/* Toggle details */}
                <div className="pt-2 border-t border-gray-100 dark:border-gray-800 flex justify-between items-center">
                  <button
                    onClick={() => setExpandedId(isExpanded ? null : sub.id)}
                    className="text-xs text-arc-600 dark:text-arc-400 font-medium hover:underline flex items-center gap-1"
                  >
                    {isExpanded ? (
                      <>
                        <ChevronUp size={14} /> Hide Telemetry & ARC Breakdown
                      </>
                    ) : (
                      <>
                        <ChevronDown size={14} /> View Telemetry & ARC Breakdown
                      </>
                    )}
                  </button>

                  <span className="text-xs text-gray-400 font-mono">
                    SHA-256: {sub.checksumSha256 ? `${sub.checksumSha256.substring(0, 10)}...` : 'Verified'}
                  </span>
                </div>

                {/* Expanded Details Drawer */}
                {isExpanded && (
                  <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-800 bg-gray-50/50 dark:bg-gray-800/40 -mx-6 -mb-6 p-4 rounded-b-xl space-y-3">
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      <div className="p-2.5 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="text-xs text-gray-500">Relevance (R)</div>
                        <div className="text-base font-bold text-blue-600">
                          {arc?.relevanceScore != null ? arc.relevanceScore.toFixed(1) : '—'} / 100
                        </div>
                        <div className="text-[11px] text-gray-400 mt-0.5">
                          Quality: {arc?.qualityScore?.toFixed(1) ?? '—'}, Diversity: {arc?.diversityScore?.toFixed(1) ?? '—'}
                        </div>
                      </div>

                      <div className="p-2.5 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="text-xs text-gray-500">Resource (S)</div>
                        <div className="text-base font-bold text-green-600">
                          {arc?.resourceScore != null ? arc.resourceScore.toFixed(1) : '—'} / 100
                        </div>
                        <div className="text-[11px] text-gray-400 mt-0.5">
                          RAM: {arc?.memoryScore?.toFixed(1) ?? '—'}, CPU: {arc?.processingScore?.toFixed(1) ?? '—'}
                        </div>
                      </div>

                      <div className="p-2.5 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                        <div className="text-xs text-gray-500">Communication (C)</div>
                        <div className="text-base font-bold text-purple-600">
                          {arc?.communicationScore != null ? arc.communicationScore.toFixed(1) : '—'} / 100
                        </div>
                        <div className="text-[11px] text-gray-400 mt-0.5">
                          Bandwidth: {arc?.bandwidthScore?.toFixed(1) ?? '—'}, Latency: {arc?.latencyScore?.toFixed(1) ?? '—'}
                        </div>
                      </div>
                    </div>

                    <div className="text-xs text-gray-500 space-y-1">
                      <div>
                        <strong>Hyperparameters:</strong> Epochs: {sub.epochs || 2}, Batch Size: {sub.batchSize || 128}, LR: {sub.learningRate || 0.001}
                      </div>
                      <div>
                        <strong>Training Duration:</strong> {sub.trainingDurationSeconds ? `${sub.trainingDurationSeconds} seconds` : 'N/A'}
                      </div>
                      {sub.validationMessage && (
                        <div className="text-amber-600 dark:text-amber-400">
                          <strong>Admin Feedback:</strong> {sub.validationMessage}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
