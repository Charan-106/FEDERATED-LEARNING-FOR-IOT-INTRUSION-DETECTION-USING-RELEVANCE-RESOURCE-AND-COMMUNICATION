// Admin Submissions Page - Fleet-wide client submission review
// Section 6 & 9: Vertical cards for client model updates with review and inspection actions

import { useEffect, useState } from 'react'
import {
  FileStack,
  CheckCircle,
  XCircle,
  Clock,
  Award,
  RefreshCw,
  Search,
  Filter,
  Check,
  X,
  ChevronDown,
  ChevronUp
} from 'lucide-react'
import { adminApi } from '../../services/api'

export default function AdminSubmissionsPage() {
  const [submissions, setSubmissions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [actionLoading, setActionLoading] = useState('')
  const [expandedId, setExpandedId] = useState(null)
  const [statusFilter, setStatusFilter] = useState('ALL')

  useEffect(() => {
    loadSubmissions()
  }, [statusFilter])

  function loadSubmissions() {
    setLoading(true)
    const params = statusFilter !== 'ALL' ? { status: statusFilter } : {}
    adminApi.getAllSubmissions(params)
      .then((data) => setSubmissions(data.submissions || []))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }

  async function handleReview(submissionId, nextStatus) {
    const reason = nextStatus === 'REJECTED'
      ? window.prompt('Enter rejection reason for client feedback:', 'Model parameters or loss failed acceptance criteria.')
      : 'Submission verified and accepted for federated aggregation.'

    if (nextStatus === 'REJECTED' && reason === null) return // User cancelled prompt

    setActionLoading(submissionId)
    try {
      await adminApi.reviewSubmission(submissionId, nextStatus, reason)
      await loadSubmissions()
    } catch (err) {
      alert(`Review failed: ${err.message}`)
    } finally {
      setActionLoading('')
    }
  }

  if (loading) {
    return <div className="card text-center py-16 text-gray-500">Loading submissions...</div>
  }

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <FileStack className="text-arc-600" size={26} />
            Client Submissions Management
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Review and qualify incoming edge parameter state dicts before federated aggregation.
          </p>
        </div>

        {/* Filter and Refresh */}
        <div className="flex items-center gap-2">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value)}
            className="text-xs p-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
          >
            <option value="ALL">All Statuses</option>
            <option value="PENDING">Pending Review</option>
            <option value="ACCEPTED">Accepted</option>
            <option value="REJECTED">Rejected</option>
            <option value="AGGREGATED">Aggregated</option>
          </select>

          <button
            onClick={loadSubmissions}
            className="btn-secondary flex items-center gap-1.5 text-xs py-2 px-3"
          >
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      {error && <div className="card bg-red-50 text-red-600 p-4 text-sm">{error}</div>}

      {/* Vertical Cards for Submissions (Mandatory layout) */}
      <div className="space-y-4">
        {submissions.length === 0 ? (
          <div className="card text-center py-16 text-gray-500">
            <FileStack size={40} className="mx-auto text-gray-300 dark:text-gray-600 mb-2" />
            <p className="font-semibold text-gray-700 dark:text-gray-200">No Submissions Found</p>
            <p className="text-xs text-gray-400 mt-1">Incoming client updates will appear here.</p>
          </div>
        ) : (
          submissions.map((sub) => {
            const isExpanded = expandedId === sub.id
            const arc = sub.arcScores?.[0]
            const isBusy = actionLoading === sub.id

            return (
              <div
                key={sub.id}
                className="card border border-gray-200 dark:border-gray-700 hover:border-arc-300 dark:hover:border-arc-700 transition-all shadow-sm"
              >
                {/* Header */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-gray-100 dark:border-gray-800 pb-3">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-bold text-gray-900 dark:text-white text-base">
                      {sub.client?.name || sub.clientId}
                    </span>
                    <span className="font-mono text-xs text-arc-600 bg-arc-50 dark:bg-arc-900/40 px-2 py-0.5 rounded">
                      {sub.clientId}
                    </span>
                    <span className="badge-blue text-xs">
                      Round #{sub.round?.roundNumber || '1'}
                    </span>
                    <span className={`badge-${sub.status === 'ACCEPTED' ? 'green' : sub.status === 'REJECTED' ? 'red' : 'yellow'} text-xs font-semibold`}>
                      {sub.status}
                    </span>
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2 self-end sm:self-auto">
                    {sub.status === 'PENDING' && (
                      <>
                        <button
                          onClick={() => handleReview(sub.id, 'ACCEPTED')}
                          disabled={isBusy}
                          className="px-3 py-1.5 bg-green-50 text-green-700 hover:bg-green-100 dark:bg-green-900/30 dark:text-green-300 rounded-lg text-xs font-semibold flex items-center gap-1 transition-colors"
                        >
                          <Check size={14} /> Accept
                        </button>
                        <button
                          onClick={() => handleReview(sub.id, 'REJECTED')}
                          disabled={isBusy}
                          className="px-3 py-1.5 bg-red-50 text-red-700 hover:bg-red-100 dark:bg-red-900/30 dark:text-red-300 rounded-lg text-xs font-semibold flex items-center gap-1 transition-colors"
                        >
                          <X size={14} /> Reject
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {/* Metrics */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 py-3">
                  <div>
                    <span className="text-xs text-gray-500">Local Accuracy</span>
                    <p className="text-lg font-bold text-green-600 mt-0.5">
                      {sub.localAccuracy != null ? `${sub.localAccuracy}%` : '—'}
                    </p>
                  </div>

                  <div>
                    <span className="text-xs text-gray-500">Local Loss</span>
                    <p className="text-lg font-bold text-gray-800 dark:text-gray-200 mt-0.5">
                      {sub.localLoss != null ? sub.localLoss.toFixed(4) : '—'}
                    </p>
                  </div>

                  <div>
                    <span className="text-xs text-gray-500">Sample Count</span>
                    <p className="text-lg font-bold text-purple-600 mt-0.5">
                      {sub.samplesTrained != null ? sub.samplesTrained.toLocaleString() : '—'}
                    </p>
                  </div>

                  <div>
                    <span className="text-xs text-gray-500">Overall ARC Score</span>
                    <p className="text-lg font-bold text-arc-600 flex items-center gap-1 mt-0.5">
                      <Award size={16} />
                      {sub.arcScore != null ? sub.arcScore.toFixed(1) : arc?.overallArcScore?.toFixed(1) || 'Pending'}
                    </p>
                  </div>
                </div>

                {/* Footer and Drawer Toggle */}
                <div className="pt-2 border-t border-gray-100 dark:border-gray-800 flex justify-between items-center text-xs">
                  <button
                    onClick={() => setExpandedId(isExpanded ? null : sub.id)}
                    className="text-arc-600 dark:text-arc-400 font-medium hover:underline flex items-center gap-1"
                  >
                    {isExpanded ? (
                      <>
                        <ChevronUp size={14} /> Hide Component ARC & Details
                      </>
                    ) : (
                      <>
                        <ChevronDown size={14} /> View Component ARC & Details
                      </>
                    )}
                  </button>

                  <span className="text-gray-400">
                    Submitted: {new Date(sub.submittedAt).toLocaleString()}
                  </span>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-800 bg-gray-50/60 dark:bg-gray-800/40 -mx-6 -mb-6 p-4 rounded-b-xl space-y-3">
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                        <span className="text-xs text-blue-600 font-semibold">Relevance (R)</span>
                        <div className="text-base font-bold text-gray-900 dark:text-white mt-0.5">
                          {arc?.relevanceScore?.toFixed(1) || '—'} / 100
                        </div>
                      </div>
                      <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                        <span className="text-xs text-green-600 font-semibold">Resource (S)</span>
                        <div className="text-base font-bold text-gray-900 dark:text-white mt-0.5">
                          {arc?.resourceScore?.toFixed(1) || '—'} / 100
                        </div>
                      </div>
                      <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                        <span className="text-xs text-purple-600 font-semibold">Communication (C)</span>
                        <div className="text-base font-bold text-gray-900 dark:text-white mt-0.5">
                          {arc?.communicationScore?.toFixed(1) || '—'} / 100
                        </div>
                      </div>
                    </div>

                    <div className="text-xs text-gray-500 font-mono space-y-1">
                      <div>Storage Path: {sub.storagePath}</div>
                      <div>SHA-256 Checksum: {sub.checksumSha256}</div>
                      {sub.validationMessage && (
                        <div className="text-amber-600 font-sans">
                          <strong>Admin Feedback:</strong> {sub.validationMessage}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
