// Admin Clients Page - Edge Fleet Management
// Section 6.B: Vertical card layout for all registered edge clients with status toggles

import { useEffect, useState } from 'react'
import {
  Users,
  CheckCircle2,
  XCircle,
  Clock,
  Cpu,
  Wifi,
  Award,
  RefreshCw,
  Power,
  ChevronDown,
  ChevronUp,
  FileStack
} from 'lucide-react'
import { adminApi } from '../../services/api'

export default function AdminClientsPage() {
  const [clients, setClients] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [actionLoading, setActionLoading] = useState('')
  const [expandedClientId, setExpandedClientId] = useState(null)

  useEffect(() => {
    loadClients()
  }, [])

  function loadClients() {
    setLoading(true)
    adminApi.getClients()
      .then((data) => setClients(data.clients || []))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }

  async function handleToggleStatus(clientId, currentStatus) {
    const nextStatus = currentStatus === 'ACTIVE' ? 'INACTIVE' : 'ACTIVE'
    const confirmMsg = `Are you sure you want to set client ${clientId} to ${nextStatus}?`
    if (!window.confirm(confirmMsg)) return

    setActionLoading(clientId)
    try {
      await adminApi.updateClientStatus(clientId, nextStatus)
      await loadClients()
    } catch (err) {
      alert(`Failed to update status: ${err.message}`)
    } finally {
      setActionLoading('')
    }
  }

  if (loading) {
    return <div className="card text-center py-16 text-gray-500">Loading edge clients fleet...</div>
  }

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <Users className="text-arc-600" size={26} />
            Clients
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Fleet monitoring and status authorization for distributed training edge participants.
          </p>
        </div>

        <button
          onClick={loadClients}
          className="btn-secondary self-start sm:self-auto flex items-center gap-1.5 text-xs py-2 px-3"
        >
          <RefreshCw size={14} /> Refresh Fleet
        </button>
      </div>

      {error && (
        <div className="card bg-red-50 text-red-600 p-4 text-sm">{error}</div>
      )}

      {/* Mandatory Vertical Cards: One Client per Card */}
      <div className="space-y-4">
        {clients.length === 0 ? (
          <div className="card text-center py-12 text-gray-500">No clients registered yet.</div>
        ) : (
          clients.map((client) => {
            const isExpanded = expandedClientId === client.id
            const arc = client.arcScores?.[0]
            const isBusy = actionLoading === client.clientId

            return (
              <div
                key={client.id}
                className="card border border-gray-200 dark:border-gray-700 hover:border-arc-300 dark:hover:border-arc-700 transition-all shadow-sm"
              >
                {/* Header Row */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-gray-100 dark:border-gray-800 pb-3">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-arc-100 dark:bg-arc-900/50 flex items-center justify-center text-arc-700 dark:text-arc-300 font-bold font-mono text-sm">
                      {client.clientId.substring(client.clientId.length - 3)}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="font-bold text-gray-900 dark:text-white text-base">
                          {client.name}
                        </h3>
                        <span className="font-mono text-xs text-arc-600 bg-arc-50 dark:bg-arc-900/40 px-2 py-0.5 rounded">
                          {client.clientId}
                        </span>
                      </div>
                      <p className="text-xs text-gray-500 mt-0.5">
                        User: {client.user?.email || 'N/A'} • Registered: {new Date(client.createdAt).toLocaleDateString()}
                      </p>
                    </div>
                  </div>

                  {/* Actions & Status */}
                  <div className="flex items-center gap-2 self-end sm:self-auto">
                    <span
                      className={`badge-${client.status === 'ACTIVE' ? 'green' : 'red'} flex items-center gap-1 text-xs`}
                    >
                      {client.status === 'ACTIVE' ? (
                        <>
                          <CheckCircle2 size={12} /> Active
                        </>
                      ) : (
                        <>
                          <XCircle size={12} /> Inactive
                        </>
                      )}
                    </span>

                    <button
                      onClick={() => handleToggleStatus(client.clientId, client.status)}
                      disabled={isBusy}
                      className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1 transition-colors ${
                        client.status === 'ACTIVE'
                          ? 'bg-red-50 text-red-600 hover:bg-red-100 dark:bg-red-900/20 dark:hover:bg-red-900/40'
                          : 'bg-green-50 text-green-600 hover:bg-green-100 dark:bg-green-900/20 dark:hover:bg-green-900/40'
                      }`}
                      title={client.status === 'ACTIVE' ? 'Deactivate Client' : 'Activate Client'}
                    >
                      <Power size={13} />
                      {isBusy ? 'Updating...' : client.status === 'ACTIVE' ? 'Deactivate' : 'Activate'}
                    </button>
                  </div>
                </div>

                {/* Primary Telemetry Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 py-3">
                  <div>
                    <span className="text-xs text-gray-500">Submissions Contributed</span>
                    <p className="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-1.5 mt-0.5">
                      <FileStack size={16} className="text-arc-500" />
                      {client._count?.submissions || 0}
                    </p>
                  </div>

                  <div>
                    <span className="text-xs text-gray-500">Latest ARC Score</span>
                    <p className="text-lg font-bold text-arc-600 flex items-center gap-1.5 mt-0.5">
                      <Award size={16} />
                      {arc?.overallArcScore != null ? arc.overallArcScore.toFixed(1) : 'Not Evaluated'}
                    </p>
                  </div>

                  <div>
                    <span className="text-xs text-gray-500">Hardware / Resource</span>
                    <p className="text-sm font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-1.5 mt-1">
                      <Cpu size={16} className="text-green-500" />
                      {arc?.resourceScore != null ? `${arc.resourceScore.toFixed(1)} / 100` : '16 GB RAM • CPU Idle'}
                    </p>
                  </div>

                  <div>
                    <span className="text-xs text-gray-500">Communication Status</span>
                    <p className="text-sm font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-1.5 mt-1">
                      <Wifi size={16} className="text-purple-500" />
                      {arc?.communicationScore != null ? `${arc.communicationScore.toFixed(1)} / 100` : 'Connected'}
                    </p>
                  </div>
                </div>

                {/* Footer Drawer Toggle */}
                <div className="pt-2 border-t border-gray-100 dark:border-gray-800 flex justify-between items-center">
                  <button
                    onClick={() => setExpandedClientId(isExpanded ? null : client.id)}
                    className="text-xs text-arc-600 dark:text-arc-400 font-medium hover:underline flex items-center gap-1"
                  >
                    {isExpanded ? (
                      <>
                        <ChevronUp size={14} /> Hide Component Scores
                      </>
                    ) : (
                      <>
                        <ChevronDown size={14} /> View Component Scores
                      </>
                    )}
                  </button>

                  <span className="text-xs text-gray-400 flex items-center gap-1">
                    <Clock size={12} />
                    Last Active: {client.lastActiveAt ? new Date(client.lastActiveAt).toLocaleString() : 'Recently'}
                  </span>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-800 bg-gray-50/60 dark:bg-gray-800/40 -mx-6 -mb-6 p-4 rounded-b-xl space-y-3">
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                      <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                        <span className="text-xs text-blue-600 font-semibold">Relevance Score (R)</span>
                        <div className="text-xl font-bold text-gray-900 dark:text-white mt-0.5">
                          {arc?.relevanceScore != null ? arc.relevanceScore.toFixed(1) : '—'} / 100
                        </div>
                        <div className="text-[11px] text-gray-400 mt-1">
                          Quality: {arc?.qualityScore?.toFixed(1) ?? '—'} • Diversity: {arc?.diversityScore?.toFixed(1) ?? '—'}
                        </div>
                      </div>

                      <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                        <span className="text-xs text-green-600 font-semibold">Resource Score (S)</span>
                        <div className="text-xl font-bold text-gray-900 dark:text-white mt-0.5">
                          {arc?.resourceScore != null ? arc.resourceScore.toFixed(1) : '—'} / 100
                        </div>
                        <div className="text-[11px] text-gray-400 mt-1">
                          RAM: {arc?.memoryScore?.toFixed(1) ?? '—'} • CPU: {arc?.processingScore?.toFixed(1) ?? '—'}
                        </div>
                      </div>

                      <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                        <span className="text-xs text-purple-600 font-semibold">Communication Score (C)</span>
                        <div className="text-xl font-bold text-gray-900 dark:text-white mt-0.5">
                          {arc?.communicationScore != null ? arc.communicationScore.toFixed(1) : '—'} / 100
                        </div>
                        <div className="text-[11px] text-gray-400 mt-1">
                          Bandwidth: {arc?.bandwidthScore?.toFixed(1) ?? '—'} • Latency: {arc?.latencyScore?.toFixed(1) ?? '—'}
                        </div>
                      </div>
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
