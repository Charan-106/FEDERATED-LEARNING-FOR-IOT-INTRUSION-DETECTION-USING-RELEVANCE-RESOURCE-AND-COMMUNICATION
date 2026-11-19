// Admin Training Monitor Page - Multi-Client Side-by-Side Training Observatory
// Fulfills Requirements:
// 1. Real-time SSE stream for Admin Observatory (/api/admin/training-progress)
// 2. Clear distinction between Live Training Progress (mini-batch events) and Model Updates Received
// 3. Dedicated Model Updates Received counter card (X / N) and details list (checksum, size, filename)
// 4. Side-by-side comparative telemetry, parameter deltas, and lifecycle progression

import { useState, useEffect, useRef } from 'react'
import {
  Activity,
  Users,
  Cpu,
  Database,
  Layers,
  ShieldCheck,
  CheckCircle,
  Clock,
  ArrowRight,
  RefreshCw,
  Copy,
  Check,
  AlertTriangle,
  Server,
  GitMerge,
  BarChart2,
  Calendar,
  ChevronRight,
  Info,
  Radio,
  FileCheck,
  Terminal,
  Zap
} from 'lucide-react'
import { adminApi } from '../../services/api'

export default function AdminTrainingMonitorPage() {
  const [data, setData] = useState(null)
  const [rounds, setRounds] = useState([])
  const [selectedRoundId, setSelectedRoundId] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [copiedHash, setCopiedHash] = useState('')

  // SSE & Live Streaming State
  const [connectionStatus, setConnectionStatus] = useState('CONNECTING') // 'CONNECTING' | 'CONNECTED' | 'DISCONNECTED' | 'ERROR'
  const [batchEventsCount, setBatchEventsCount] = useState(0)
  const [adminEventsLog, setAdminEventsLog] = useState([])

  const sseAbortRef = useRef(null)
  const seenAdminBatchesRef = useRef(new Set())

  useEffect(() => {
    fetchRounds()
  }, [])

  useEffect(() => {
    if (selectedRoundId) {
      fetchMonitorData(selectedRoundId)
    }
  }, [selectedRoundId])

  useEffect(() => {
    setupAdminSSE()
    return () => {
      if (sseAbortRef.current) {
        sseAbortRef.current.abort()
      }
    }
  }, [])

  async function fetchRounds() {
    try {
      const res = await adminApi.getRounds()
      const roundsList = res.rounds || []
      setRounds(roundsList)
      if (roundsList.length > 0) {
        setSelectedRoundId(roundsList[0].id)
      } else {
        setLoading(false)
      }
    } catch (err) {
      setError(err.message || 'Failed to fetch training rounds.')
      setLoading(false)
    }
  }

  async function fetchMonitorData(roundId) {
    setLoading(true)
    setError('')
    try {
      const res = await adminApi.getTrainingMonitor(roundId)
      setData(res)
    } catch (err) {
      setError(err.message || 'Failed to load admin training monitor data.')
    } finally {
      setLoading(false)
    }
  }

  function setupAdminSSE() {
    const token = localStorage.getItem('arc_fl_token')
    if (!token) return

    if (sseAbortRef.current) {
      sseAbortRef.current.abort()
    }
    const abortController = new AbortController()
    sseAbortRef.current = abortController

    connectAdminSSE(token, abortController.signal)
  }

  async function connectAdminSSE(token, signal) {
    setConnectionStatus('CONNECTING')
    try {
      const response = await fetch('/api/admin/training-progress', {
        headers: {
          Authorization: `Bearer ${token}`,
        },
        signal,
      })

      if (!response.ok) {
        setConnectionStatus('ERROR')
        return
      }

      setConnectionStatus('CONNECTED')
      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop()

        for (const line of lines) {
          const trimmed = line.trim()
          if (trimmed.startsWith('data: ')) {
            try {
              const eventData = JSON.parse(trimmed.substring(6))
              handleAdminStreamEvent(eventData)
            } catch (e) {
              // Ignore ping or heartbeat parse issues
            }
          }
        }
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.warn('[Admin SSE] Stream error:', err.message)
        setConnectionStatus('DISCONNECTED')
        // Attempt reconnect after 5 seconds
        setTimeout(() => {
          if (!signal.aborted) setupAdminSSE()
        }, 5000)
      }
    }
  }

  function handleAdminStreamEvent(event) {
    if (event.type === 'CONNECTED') {
      setConnectionStatus('CONNECTED')
      if (event.activeSessions && Object.keys(event.activeSessions).length > 0) {
        setData((prev) => {
          if (!prev || !prev.clientMatrix) return prev
          const updatedMatrix = prev.clientMatrix.map((c) => {
            const live = event.activeSessions[c.clientId]
            if (live && live.status === 'RUNNING') {
              return {
                ...c,
                isCurrentlyTraining: true,
                status: 'TRAINING_IN_PROGRESS',
                liveProgress: {
                  currentStage: live.currentStage,
                  progress: live.progress,
                  currentEpoch: live.currentEpoch,
                  totalEpochs: live.totalEpochs,
                  currentBatch: live.currentBatch,
                  totalBatches: live.totalBatches,
                  batchLoss: live.batchLoss,
                  elapsedSeconds: live.elapsedSeconds,
                },
              }
            }
            return c
          })
          return { ...prev, clientMatrix: updatedMatrix }
        })
      }
    } else if (event.type === 'CLIENT_PROGRESS') {
      // 1. Increment Mini-Batch Events Counter ONLY for distinct mini-batch events (NEVER increment model updates)
      const isBatch = Boolean(
        event.currentBatch &&
        event.totalBatches &&
        (event.batchLoss !== undefined || event.isBatchEvent)
      )

      if (isBatch) {
        const batchKey = `${event.clientId}_e${event.currentEpoch || 1}_b${event.currentBatch}`
        if (!seenAdminBatchesRef.current.has(batchKey)) {
          seenAdminBatchesRef.current.add(batchKey)
          setBatchEventsCount(seenAdminBatchesRef.current.size)
        }
      }

      // 2. Update client in matrix
      setData((prev) => {
        if (!prev || !prev.clientMatrix) return prev
        const updatedMatrix = prev.clientMatrix.map((c) => {
          if (c.clientId === event.clientId) {
            return {
              ...c,
              isCurrentlyTraining: event.status === 'RUNNING',
              status: event.status === 'RUNNING' ? 'TRAINING_IN_PROGRESS' : c.status,
              liveProgress: {
                currentStage: event.currentStage,
                progress: event.progress,
                currentEpoch: event.currentEpoch,
                totalEpochs: event.totalEpochs,
                currentBatch: event.currentBatch,
                totalBatches: event.totalBatches,
                batchLoss: event.batchLoss,
                elapsedSeconds: event.elapsedSeconds,
                message: event.message,
              },
            }
          }
          return c
        })
        return { ...prev, clientMatrix: updatedMatrix }
      })

      // 3. Log event
      setAdminEventsLog((prev) => [
        {
          timestamp: new Date().toLocaleTimeString(),
          type: 'PROGRESS',
          clientId: event.clientId,
          message: event.message || `Epoch ${event.currentEpoch || 1}, Batch ${event.currentBatch || 0}/${event.totalBatches || 0} (Loss: ${event.batchLoss ?? 'N/A'})`,
        },
        ...prev.slice(0, 49),
      ])
    } else if (event.type === 'UPDATE_REGISTERED') {
      // 1. Increment Model Updates Received Counter and Register Update
      setData((prev) => {
        if (!prev) return prev
        const existingSubs = prev.summary?.registeredSubmissions || []
        const alreadyExists = existingSubs.some((s) => s.clientId === event.clientId)
        const updatedSubs = alreadyExists
          ? existingSubs.map((s) => (s.clientId === event.clientId ? { ...s, ...event } : s))
          : [...existingSubs, event]

        const updatesCount = updatedSubs.length
        const targetCount = prev.summary?.targetClientsCount || 2

        // Also update client matrix
        const updatedMatrix = prev.clientMatrix?.map((c) => {
          if (c.clientId === event.clientId) {
            return {
              ...c,
              status: 'VALIDATED_OK',
              isCurrentlyTraining: false,
              submittedUpdate: {
                checksumSha256: event.checksumSha256,
                fileSizeBytes: event.fileSizeBytes,
                fileSizeMb: event.fileSizeMb,
                submittedAt: event.submittedAt,
                validationStatus: 'VALIDATED_OK',
              },
              trainingMetrics: {
                ...c.trainingMetrics,
                localLoss: event.localLoss ?? c.trainingMetrics?.localLoss,
                localAccuracy: event.localAccuracy ?? c.trainingMetrics?.localAccuracy,
              },
            }
          }
          return c
        })

        return {
          ...prev,
          summary: {
            ...prev.summary,
            updatesReceivedCount: updatesCount,
            allUpdatesReceived: updatesCount >= targetCount,
            registeredSubmissions: updatedSubs,
          },
          clientMatrix: updatedMatrix,
        }
      })

      // 2. Log event
      setAdminEventsLog((prev) => [
        {
          timestamp: new Date().toLocaleTimeString(),
          type: 'SUBMISSION',
          clientId: event.clientId,
          message: `Model update registered by server: SHA-256 ${event.checksumSha256?.substring(0, 16)}... (${event.fileSizeMb || 'N/A'} MB)`,
        },
        ...prev.slice(0, 49),
      ])
    }
  }

  function copyText(text, key) {
    if (!text) return
    navigator.clipboard.writeText(text)
    setCopiedHash(key)
    setTimeout(() => setCopiedHash(''), 2000)
  }

  const updatesCount = data?.summary?.updatesReceivedCount ?? 0
  const targetClientsCount = data?.summary?.targetClientsCount ?? 2
  const allReceived = updatesCount >= targetClientsCount
  const registeredSubmissions = data?.summary?.registeredSubmissions || []

  return (
    <div className="space-y-6">
      {/* ------------------------------------------------------------- */}
      {/* Header & Controls                                             */}
      {/* ------------------------------------------------------------- */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-gray-200 dark:border-gray-700 pb-4">
        <div>
          <div className="flex items-center gap-3">
            <h1 className="text-2xl font-bold text-gray-900 dark:text-white">Admin Training Monitor</h1>
            <span className="badge-purple font-mono font-bold">Round #{data?.round?.roundNumber || ''}</span>
            <span className={`px-2 py-0.5 text-xs font-semibold rounded-full ${
              data?.round?.status === 'COMPLETED' ? 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300' :
              data?.round?.status === 'ACTIVE' ? 'bg-blue-100 text-blue-800 dark:bg-blue-900/40 dark:text-blue-300' :
              'bg-gray-100 text-gray-800 dark:bg-gray-800 dark:text-gray-300'
            }`}>
              {data?.round?.status || 'UNKNOWN'}
            </span>
            {/* Live SSE Connection Pill */}
            <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-mono border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
              <span className={`w-2 h-2 rounded-full ${
                connectionStatus === 'CONNECTED' ? 'bg-green-500 animate-pulse' :
                connectionStatus === 'CONNECTING' ? 'bg-yellow-500' : 'bg-red-500'
              }`} />
              <span className="text-gray-600 dark:text-gray-400 font-semibold">{connectionStatus}</span>
            </div>
          </div>
          <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
            Real-time cross-client comparative telemetry, delta verification, and round lifecycle progression
          </p>
        </div>

        <div className="flex items-center gap-3">
          <select
            value={selectedRoundId}
            onChange={(e) => setSelectedRoundId(e.target.value)}
            className="input text-xs font-medium py-1.5 px-3 min-w-[180px]"
          >
            {rounds.map((r) => (
              <option key={r.id} value={r.id}>
                Round #{r.roundNumber} ({r.status})
              </option>
            ))}
          </select>

          <button
            onClick={() => fetchMonitorData(selectedRoundId)}
            disabled={loading}
            className="btn-secondary flex items-center gap-1.5 text-xs"
          >
            <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
            Refresh
          </button>
        </div>
      </div>

      {loading && !data && (
        <div className="flex flex-col items-center justify-center min-h-[350px] space-y-4">
          <RefreshCw size={32} className="animate-spin text-arc-600" />
          <p className="text-sm text-gray-500">Loading cross-client telemetry...</p>
        </div>
      )}

      {error && (
        <div className="card border-red-200 bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300 p-4 text-xs">
          {error}
        </div>
      )}

      {data && (
        <>
          {/* ------------------------------------------------------------- */}
          {/* TOP DUAL-MONITOR SECTION: MODEL UPDATES vs LIVE MINI-BATCHES  */}
          {/* ------------------------------------------------------------- */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* CARD 1: Model Updates Received Counter (CRITICAL REQUIREMENT) */}
            <div className={`card p-6 border-2 transition-all ${
              allReceived
                ? 'border-green-300 dark:border-green-800 bg-green-50/20 dark:bg-green-900/10'
                : 'border-blue-200 dark:border-blue-800 bg-white dark:bg-gray-800'
            }`}>
              <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-3">
                <div className="flex items-center gap-2.5">
                  <FileCheck className={allReceived ? 'text-green-600 dark:text-green-400' : 'text-arc-600'} size={22} />
                  <div>
                    <h2 className="text-sm font-bold text-gray-900 dark:text-white uppercase tracking-wider">
                      Model Updates Received
                    </h2>
                    <p className="text-[11px] text-gray-500">
                      Validated and registered client parameter checkpoints in PostgreSQL
                    </p>
                  </div>
                </div>
                <span className={`px-2.5 py-1 text-xs font-bold rounded-full ${
                  allReceived
                    ? 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300'
                    : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300'
                }`}>
                  {allReceived ? 'ALL UPDATES RECEIVED' : 'AWAITING UPDATES'}
                </span>
              </div>

              {/* Large Updates Counter */}
              <div className="my-4 flex items-baseline gap-3">
                <span className="text-4xl font-extrabold font-mono text-gray-900 dark:text-white">
                  {updatesCount}
                </span>
                <span className="text-xl font-bold font-mono text-gray-400">
                  / {targetClientsCount}
                </span>
                <span className="text-xs text-gray-500 font-medium ml-2">
                  Client updates registered for Round #{data.round?.roundNumber}
                </span>
              </div>

              {/* Submissions List / Details */}
              <div className="space-y-2 mt-4">
                <span className="text-[11px] uppercase font-bold text-gray-500 tracking-wider">
                  Registered Client Updates:
                </span>
                {registeredSubmissions.length === 0 ? (
                  <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800/60 border border-dashed border-gray-300 dark:border-gray-700 text-center text-xs text-gray-400">
                    No client model updates registered yet for this round.
                  </div>
                ) : (
                  <div className="space-y-2 max-h-48 overflow-y-auto">
                    {registeredSubmissions.map((sub, idx) => (
                      <div
                        key={sub.submissionId || idx}
                        className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800/80 border border-gray-200 dark:border-gray-700 text-xs space-y-1.5"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-gray-900 dark:text-white">{sub.clientId}</span>
                            <span className="badge-green text-[10px] font-mono">VALIDATED_OK</span>
                          </div>
                          <span className="font-mono text-[11px] text-gray-500">
                            {sub.fileSizeMb ? `${sub.fileSizeMb} MB` : 'Size N/A'}
                          </span>
                        </div>
                        <div className="flex items-center justify-between font-mono text-[10px] text-gray-500">
                          <span>SHA-256: {sub.checksumSha256 ? `${sub.checksumSha256.substring(0, 20)}...` : 'N/A'}</span>
                          {sub.checksumSha256 && (
                            <button
                              onClick={() => copyText(sub.checksumSha256, sub.submissionId || idx)}
                              className="text-arc-600 hover:text-arc-700 flex items-center gap-1"
                            >
                              {copiedHash === (sub.submissionId || idx) ? <Check size={10} /> : <Copy size={10} />}
                              {copiedHash === (sub.submissionId || idx) ? 'Copied' : 'Copy'}
                            </button>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            {/* CARD 2: Live Mini-Batch Progress & Fleet Activity */}
            <div className="card p-6 border-blue-200 dark:border-blue-800 bg-white dark:bg-gray-800 space-y-4">
              <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-3">
                <div className="flex items-center gap-2.5">
                  <Zap className="text-amber-500" size={22} />
                  <div>
                    <h2 className="text-sm font-bold text-gray-900 dark:text-white uppercase tracking-wider">
                      Live Edge Training Fleet Activity
                    </h2>
                    <p className="text-[11px] text-gray-500">
                      Real-time PyTorch mini-batch events emitted from local edge loops
                    </p>
                  </div>
                </div>
                <span className="badge-blue font-mono text-xs">
                  {batchEventsCount} Mini-Batch Events Streamed
                </span>
              </div>

              {/* Invariant Explanation Banner */}
              <div className="p-3 rounded-lg bg-blue-50/70 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 text-[11px] text-blue-900 dark:text-blue-200 flex items-start gap-2">
                <Info size={14} className="mt-0.5 flex-shrink-0 text-blue-600 dark:text-blue-400" />
                <span>
                  <strong>Strict Invariant:</strong> Mini-batch progress events stream PyTorch edge training telemetry and do <em>not</em> increment the completed model-update counter. Completed model updates increment the counter only after backend validation in PostgreSQL.
                </span>
              </div>

              {/* Recent Live Events Console */}
              <div className="space-y-1.5">
                <span className="text-[11px] uppercase font-bold text-gray-500 tracking-wider flex items-center gap-1.5">
                  <Terminal size={12} />
                  Live SSE Event Stream:
                </span>
                <div className="bg-gray-900 text-gray-200 rounded-lg p-3 font-mono text-[10px] h-36 overflow-y-auto space-y-1">
                  {adminEventsLog.length === 0 ? (
                    <span className="text-gray-500">Connected to SSE stream. Waiting for edge training activity...</span>
                  ) : (
                    adminEventsLog.map((log, idx) => (
                      <div key={idx} className="flex items-start gap-2">
                        <span className="text-gray-500">[{log.timestamp}]</span>
                        <span className={log.type === 'SUBMISSION' ? 'text-green-400 font-bold' : 'text-blue-300'}>
                          [{log.clientId}]
                        </span>
                        <span className="text-gray-300">{log.message}</span>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* ------------------------------------------------------------- */}
          {/* Side-by-Side Participating Clients Comparison Matrix          */}
          {/* ------------------------------------------------------------- */}
          <div className="card p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-3">
              <div>
                <h2 className="text-sm font-bold text-gray-900 dark:text-white uppercase tracking-wider flex items-center gap-2">
                  <Users size={16} className="text-arc-600" />
                  Side-by-Side Client Edge Training Matrix
                </h2>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                  Comparative analysis of training convergence, parameter deltas, and ARC contribution scores
                </p>
              </div>
              <span className="text-xs font-mono font-semibold text-gray-500">
                {data.clientMatrix?.length || 0} Clients Registered
              </span>
            </div>

            {/* Side-by-Side Responsive Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {data.clientMatrix?.map((client) => {
                const isAggregated = client.status === 'AGGREGATED'
                const isValidated = client.status === 'VALIDATED_OK'
                const isPending = client.status === 'PENDING'

                return (
                  <div
                    key={client.clientId}
                    className={`rounded-xl border p-5 space-y-4 transition-all ${
                      isAggregated || isValidated
                        ? 'border-green-200 dark:border-green-800 bg-green-50/20 dark:bg-green-900/10'
                        : isPending
                        ? 'border-gray-200 dark:border-gray-700 bg-gray-50/50 dark:bg-gray-800/40'
                        : 'border-blue-200 dark:border-blue-800 bg-blue-50/20 dark:bg-blue-900/10'
                    }`}
                  >
                    {/* Client Header */}
                    <div className="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 pb-3">
                      <div>
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-gray-900 dark:text-white text-sm">{client.clientName}</span>
                          <span className="badge-blue font-mono text-[10px]">{client.clientId}</span>
                        </div>
                        <span className="text-[10px] text-gray-500">{client.executionMode}</span>
                      </div>
                      <span className={`px-2 py-0.5 text-[10px] font-bold rounded-full ${
                        isAggregated || isValidated ? 'bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300' :
                        isPending ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300' :
                        'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
                      }`}>
                        {client.status}
                      </span>
                    </div>

                    {/* Live Training Status (if in progress) */}
                    {client.isCurrentlyTraining && client.liveProgress && (
                      <div className="p-2.5 rounded-lg bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 space-y-1 text-xs">
                        <div className="flex items-center justify-between text-amber-800 dark:text-amber-200 font-bold">
                          <span className="flex items-center gap-1">
                            <RefreshCw size={12} className="animate-spin" />
                            Live Edge Training
                          </span>
                          <span>{client.liveProgress.progress}%</span>
                        </div>
                        <div className="text-[11px] text-amber-700 dark:text-amber-300">
                          Epoch {client.liveProgress.currentEpoch}/{client.liveProgress.totalEpochs} &bull; Batch {client.liveProgress.currentBatch}/{client.liveProgress.totalBatches}
                        </div>
                        {client.liveProgress.batchLoss !== undefined && (
                          <div className="text-[10px] font-mono text-amber-800 dark:text-amber-200">
                            Batch Loss: {client.liveProgress.batchLoss} ({client.liveProgress.elapsedSeconds}s)
                          </div>
                        )}
                      </div>
                    )}

                    {/* Private Dataset Partition */}
                    <div className="space-y-1 text-xs">
                      <div className="flex items-center justify-between text-gray-500">
                        <span>Assigned Partition:</span>
                        <span className="font-mono font-semibold text-gray-900 dark:text-white">{client.assignedDataset?.name}</span>
                      </div>
                      <div className="flex items-center justify-between text-gray-500">
                        <span>Partition Samples:</span>
                        <span className="font-semibold text-gray-900 dark:text-white">{client.assignedDataset?.sampleCount} rows ({client.assignedDataset?.sizeKb} KB)</span>
                      </div>
                    </div>

                    {/* Model Received */}
                    <div className="p-2.5 rounded-lg bg-gray-100/70 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700 space-y-1 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] uppercase font-bold text-gray-500">Model Received:</span>
                        <span className="font-mono text-purple-600 font-semibold">{client.modelReceived?.version}</span>
                      </div>
                      <div className="font-mono text-[9px] text-gray-600 dark:text-gray-400 break-all">
                        {client.modelReceived?.checksumSha256?.substring(0, 24)}...
                      </div>
                    </div>

                    {/* Training Metrics */}
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="p-2.5 rounded-lg bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700">
                        <span className="text-[10px] text-gray-400">Local Loss:</span>
                        <div className="font-mono font-bold text-sm text-gray-900 dark:text-white mt-0.5">
                          {client.trainingMetrics?.localLoss ?? 'N/A'}
                        </div>
                      </div>
                      <div className="p-2.5 rounded-lg bg-white dark:bg-gray-800 border border-gray-100 dark:border-gray-700">
                        <span className="text-[10px] text-gray-400">Local Accuracy:</span>
                        <div className="font-mono font-bold text-sm text-green-600 dark:text-green-400 mt-0.5">
                          {client.trainingMetrics?.localAccuracy ? `${client.trainingMetrics.localAccuracy}%` : 'N/A'}
                        </div>
                      </div>
                    </div>

                    {/* ARC Contribution & Weight */}
                    <div className="p-3 rounded-lg bg-purple-50/50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 space-y-1 text-xs">
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-purple-900 dark:text-purple-200">ARC Score:</span>
                        <span className="font-mono font-bold text-purple-700 dark:text-purple-300 text-sm">
                          {client.arcScore ? `${Number(client.arcScore).toFixed(4)} pts` : 'Pending'}
                        </span>
                      </div>
                      {client.arcBreakdown && (
                        <div className="text-[10px] text-purple-700 dark:text-purple-300 flex items-center justify-between">
                          <span>R: {client.arcBreakdown.relevance?.toFixed(1)}</span>
                          <span>C: {client.arcBreakdown.resource?.toFixed(1)}</span>
                          <span>M: {client.arcBreakdown.communication?.toFixed(1)}</span>
                        </div>
                      )}
                    </div>

                    {/* Submitted Parameter Delta */}
                    <div className="space-y-1.5 text-xs pt-1 border-t border-gray-200 dark:border-gray-700">
                      <div className="flex items-center justify-between text-gray-500">
                        <span>Delta Checksum:</span>
                        {client.submittedUpdate?.checksumSha256 && (
                          <button
                            onClick={() => copyText(client.submittedUpdate.checksumSha256, client.clientId)}
                            className="text-arc-600 hover:text-arc-700 text-[10px] flex items-center gap-1"
                          >
                            {copiedHash === client.clientId ? <Check size={10} /> : <Copy size={10} />}
                            {copiedHash === client.clientId ? 'Copied' : 'Copy'}
                          </button>
                        )}
                      </div>
                      <div className="font-mono text-[9px] text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 p-1.5 rounded border border-gray-200 dark:border-gray-700 break-all">
                        {client.submittedUpdate?.checksumSha256 || 'Awaiting Submission'}
                      </div>
                      <div className="flex items-center justify-between text-[10px] text-gray-400">
                        <span>Artifact Size: {client.submittedUpdate?.fileSizeMb ? `${client.submittedUpdate.fileSizeMb} MB` : 'N/A'}</span>
                        <span className={`font-semibold ${client.submittedUpdate?.validationStatus === 'VALIDATED_OK' ? 'text-green-600' : 'text-gray-400'}`}>
                          {client.submittedUpdate?.validationStatus}
                        </span>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* ------------------------------------------------------------- */}
          {/* Round Lifecycle Visual Timeline                               */}
          {/* ------------------------------------------------------------- */}
          <div className="card p-6">
            <h2 className="text-sm font-bold text-gray-900 dark:text-white uppercase tracking-wider mb-4 flex items-center gap-2">
              <Clock size={16} className="text-arc-600" />
              Round #{data.round?.roundNumber} Full Lifecycle Timeline
            </h2>

            <div className="relative border-l border-gray-200 dark:border-gray-700 ml-4 space-y-6 py-2">
              {data.timeline?.map((item, idx) => (
                <div key={idx} className="relative pl-6">
                  <span className={`absolute -left-3 top-0.5 w-6 h-6 rounded-full flex items-center justify-center border-2 ${
                    item.status === 'COMPLETED'
                      ? 'bg-green-100 border-green-500 text-green-700 dark:bg-green-900/60 dark:border-green-400'
                      : 'bg-gray-100 border-gray-300 text-gray-400 dark:bg-gray-800 dark:border-gray-600'
                  }`}>
                    {item.status === 'COMPLETED' ? <CheckCircle size={14} /> : <span className="w-2 h-2 rounded-full bg-gray-400" />}
                  </span>

                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-1">
                    <span className="font-bold text-sm text-gray-900 dark:text-white">{item.label}</span>
                    {item.timestamp && (
                      <span className="text-[11px] font-mono text-gray-400">
                        {new Date(item.timestamp).toLocaleString()}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">{item.details}</p>
                </div>
              ))}
            </div>
          </div>

          {/* ------------------------------------------------------------- */}
          {/* Aggregated Candidate Model Summary (If Completed)             */}
          {/* ------------------------------------------------------------- */}
          {data.candidate && (
            <div className="card p-6 border-orange-200 dark:border-orange-800 bg-orange-50/20 dark:bg-orange-900/10 space-y-3">
              <div className="flex items-center justify-between border-b border-orange-200 dark:border-orange-800 pb-3">
                <div className="flex items-center gap-2">
                  <GitMerge className="text-orange-600" size={18} />
                  <h3 className="font-bold text-sm text-gray-900 dark:text-white uppercase tracking-wider">
                    Aggregated Candidate Evaluation Outcome
                  </h3>
                </div>
                <span className="badge-orange font-mono text-xs">{data.candidate.strategy}</span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
                <div>
                  <span className="text-gray-500">Benchmark A Accuracy:</span>
                  <div className="font-mono font-bold text-base text-green-600 dark:text-green-400 mt-0.5">
                    {data.candidate.evaluation?.accuracy ? `${data.candidate.evaluation.accuracy}%` : 'Evaluation Pending'}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">Cross-Entropy Loss:</span>
                  <div className="font-mono font-bold text-base text-gray-900 dark:text-white mt-0.5">
                    {data.candidate.evaluation?.loss || 'Evaluation Pending'}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">Candidate SHA-256:</span>
                  <div className="font-mono text-[10px] text-gray-700 dark:text-gray-300 break-all mt-0.5">
                    {data.candidate.candidateChecksum}
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
