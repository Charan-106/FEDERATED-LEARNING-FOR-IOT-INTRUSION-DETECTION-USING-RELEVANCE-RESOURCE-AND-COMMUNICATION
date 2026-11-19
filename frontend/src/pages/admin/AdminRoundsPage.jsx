// Admin Rounds Page - Complete Federated Round Management & Lifecycle Controls
// Supports configurable round creation, full round details, editing pending rounds,
// starting, cancelling, and safe deletion with historical preservation.

import { useEffect, useState } from 'react'
import {
  RefreshCw,
  Plus,
  Play,
  CheckCircle2,
  Clock,
  Layers,
  Award,
  ChevronRight,
  AlertCircle,
  X,
  Trash2,
  Edit3,
  Ban,
  Search,
  Sliders,
  ShieldCheck,
  Cpu,
  Eye,
  Check,
  Filter,
  Info,
  Archive,
  FolderArchive
} from 'lucide-react'
import { adminApi, modelApi } from '../../services/api'

export default function AdminRoundsPage() {
  const [rounds, setRounds] = useState([])
  const [activeModel, setActiveModel] = useState(null)
  const [modelsList, setModelsList] = useState([])
  const [clientsList, setClientsList] = useState([])
  const [loading, setLoading] = useState(true)
  const [filterStatus, setFilterStatus] = useState('ALL')
  const [searchQuery, setSearchQuery] = useState('')
  const [error, setError] = useState('')
  const [successMsg, setSuccessMsg] = useState('')

  // Modals state
  const [createModalOpen, setCreateModalOpen] = useState(false)
  const [editModalOpen, setEditModalOpen] = useState(false)
  const [detailsModalOpen, setDetailsModalOpen] = useState(false)
  const [activeRoundDetails, setActiveRoundDetails] = useState(null)
  const [actionLoading, setActionLoading] = useState(false)

  // Manage / Archive / Safe Delete Modal State
  const [manageModalOpen, setManageModalOpen] = useState(false)
  const [manageRoundData, setManageRoundData] = useState(null)
  const [manageDependencies, setManageDependencies] = useState(null)
  const [archiveReasonInput, setArchiveReasonInput] = useState('')

  // Create / Edit Form State
  const [formState, setFormState] = useState({
    id: null,
    name: '',
    description: '',
    baseModelId: '',
    clientIds: ['client-001'],
    epochs: 2,
    batchSize: 32,
    learningRate: 0.0001,
    timeoutSeconds: 300,
    lambdaRelevance: 0.40,
    lambdaResource: 0.30,
    lambdaCommunication: 0.30,
    aggregationStrategy: 'ARC_WEIGHTED',
  })

  // Review step inside modal
  const [isReviewStep, setIsReviewStep] = useState(false)

  useEffect(() => {
    loadData()
  }, [])

  async function loadData() {
    setLoading(true)
    setError('')
    try {
      const [rData, mData, allModels, cData] = await Promise.all([
        adminApi.getRounds(),
        modelApi.getCurrent(),
        modelApi.getVersions(),
        adminApi.getClients(),
      ])
      setRounds(rData.rounds || [])
      setActiveModel(mData.model || null)
      const models = allModels?.models || []
      const active = mData?.model || null
      setModelsList(models.length > 0 ? models : (active ? [active] : []))
      setClientsList(cData.clients || [])
    } catch (err) {
      setError(`Failed to load rounds, models, or clients: ${err.message}`)
    } finally {
      setLoading(false)
    }
  }

  function openCreateModal() {
    const nextRoundNum = rounds.length > 0 ? Math.max(...rounds.map((r) => r.roundNumber)) + 1 : 1
    const availableClients = clientsList.map((c) => c.clientId)
    const activeBaseId = activeModel?.id || modelsList.find((m) => m.isActive)?.id || modelsList[0]?.id || ''
    setFormState({
      id: null,
      name: `Federated Round #${nextRoundNum}`,
      description: 'CICIoT2023 Intrusion Detection Trial with ResMLP',
      baseModelId: activeBaseId,
      clientIds: availableClients.length > 0 ? availableClients : ['client-001'],
      epochs: 2,
      batchSize: 32,
      learningRate: 0.0001,
      timeoutSeconds: 300,
      lambdaRelevance: 0.40,
      lambdaResource: 0.30,
      lambdaCommunication: 0.30,
      aggregationStrategy: 'ARC_WEIGHTED',
    })
    setIsReviewStep(false)
    setError('')
    setCreateModalOpen(true)
  }

  function openEditModal(round) {
    if (round.status !== 'CREATED') {
      alert(`Cannot edit round with status '${round.status}'. Only CREATED (pending) rounds can be edited.`)
      return
    }
    const hp = round.hyperparameters || {}
    const arc = round.arcWeights || {}
    setFormState({
      id: round.id,
      name: round.name || `Round #${round.roundNumber}`,
      description: round.description || '',
      baseModelId: round.baseModelId || activeModel?.id || '',
      clientIds: round.targetClients || (round.clientCount > 0 ? ['client-001'] : []),
      epochs: hp.epochs !== undefined ? Number(hp.epochs) : 2,
      batchSize: hp.batchSize !== undefined ? Number(hp.batchSize) : 32,
      learningRate: hp.learningRate !== undefined ? Number(hp.learningRate) : 0.0001,
      timeoutSeconds: hp.timeoutSeconds !== undefined ? Number(hp.timeoutSeconds) : 300,
      lambdaRelevance: arc.lambdaRelevance ?? 0.40,
      lambdaResource: arc.lambdaResource ?? 0.30,
      lambdaCommunication: arc.lambdaCommunication ?? 0.30,
      aggregationStrategy: round.aggregationStrategy || 'ARC_WEIGHTED',
    })
    setError('')
    setEditModalOpen(true)
  }

  async function openDetailsModal(roundId) {
    setActionLoading(true)
    setError('')
    try {
      const res = await adminApi.getRoundDetails(roundId)
      setActiveRoundDetails(res.round)
      setDetailsModalOpen(true)
    } catch (err) {
      setError(`Failed to load round details: ${err.message}`)
    } finally {
      setActionLoading(false)
    }
  }

  // ARC weight sum validation
  const arcSum = Number((Number(formState.lambdaRelevance) + Number(formState.lambdaResource) + Number(formState.lambdaCommunication)).toFixed(4))
  const isArcValid = Math.abs(arcSum - 1.0) < 0.001

  function handleClientCheckbox(clientId) {
    if (formState.clientIds.includes(clientId)) {
      if (formState.clientIds.length === 1) {
        alert('At least one client must be selected for the federated round.')
        return
      }
      setFormState({ ...formState, clientIds: formState.clientIds.filter((id) => id !== clientId) })
    } else {
      setFormState({ ...formState, clientIds: [...formState.clientIds, clientId] })
    }
  }

  async function handleConfirmCreate() {
    if (!isArcValid) {
      alert(`ARC weights must sum to exactly 1.0 (current sum: ${arcSum})`)
      return
    }
    if (formState.clientIds.length === 0) {
      alert('Please select at least one participating client.')
      return
    }

    setActionLoading(true)
    setError('')
    try {
      const payload = {
        name: formState.name,
        description: formState.description,
        baseModelId: formState.baseModelId,
        clientIds: formState.clientIds,
        hyperparameters: {
          epochs: Number(formState.epochs),
          batchSize: Number(formState.batchSize),
          learningRate: Number(formState.learningRate),
          timeoutSeconds: Number(formState.timeoutSeconds),
        },
        arcWeights: {
          lambdaRelevance: Number(formState.lambdaRelevance),
          lambdaResource: Number(formState.lambdaResource),
          lambdaCommunication: Number(formState.lambdaCommunication),
        },
        aggregationStrategy: formState.aggregationStrategy,
      }
      const res = await adminApi.createRound(payload)
      setCreateModalOpen(false)
      setSuccessMsg(`Round #${res.round.roundNumber} created successfully! (Status: CREATED / Pending)`)
      await loadData()
    } catch (err) {
      setError(`Failed to create round: ${err.message}`)
    } finally {
      setActionLoading(false)
    }
  }

  async function handleConfirmEdit() {
    if (!isArcValid) {
      alert(`ARC weights must sum to exactly 1.0 (current sum: ${arcSum})`)
      return
    }
    setActionLoading(true)
    setError('')
    try {
      const payload = {
        name: formState.name,
        description: formState.description,
        baseModelId: formState.baseModelId,
        clientIds: formState.clientIds,
        hyperparameters: {
          epochs: Number(formState.epochs),
          batchSize: Number(formState.batchSize),
          learningRate: Number(formState.learningRate),
          timeoutSeconds: Number(formState.timeoutSeconds),
        },
        arcWeights: {
          lambdaRelevance: Number(formState.lambdaRelevance),
          lambdaResource: Number(formState.lambdaResource),
          lambdaCommunication: Number(formState.lambdaCommunication),
        },
        aggregationStrategy: formState.aggregationStrategy,
      }
      await adminApi.updateRound(formState.id, payload)
      setEditModalOpen(false)
      setSuccessMsg(`Round #${formState.id} updated successfully.`)
      await loadData()
    } catch (err) {
      setError(`Failed to update round: ${err.message}`)
    } finally {
      setActionLoading(false)
    }
  }

  async function handleStartRound(round) {
    if (!window.confirm(`Start Round #${round.roundNumber} (${round.name || 'Untitled'})? Participating clients will be notified to begin training.`)) return
    setActionLoading(true)
    setError('')
    try {
      await adminApi.startRound(round.id)
      setSuccessMsg(`Round #${round.roundNumber} is now ACTIVE!`)
      await loadData()
    } catch (err) {
      setError(`Failed to start round: ${err.message}`)
    } finally {
      setActionLoading(false)
    }
  }

  async function handleCancelRound(round) {
    const reason = window.prompt(`Cancel Round #${round.roundNumber}? Provide cancellation reason:`, 'Administrative cancellation')
    if (reason === null) return
    setActionLoading(true)
    setError('')
    try {
      await adminApi.cancelRound(round.id, reason)
      setSuccessMsg(`Round #${round.roundNumber} cancelled.`)
      await loadData()
    } catch (err) {
      setError(`Failed to cancel round: ${err.message}`)
    } finally {
      setActionLoading(false)
    }
  }

  async function handleDeleteRound(round) {
    handleOpenManageModal(round)
  }

  async function handleOpenManageModal(round) {
    setManageRoundData(round)
    setManageDependencies(null)
    setArchiveReasonInput('')
    setManageModalOpen(true)
    try {
      const deps = await adminApi.getRoundDependencies(round.id)
      setManageDependencies(deps)
    } catch (err) {
      setError(`Failed to inspect round dependencies: ${err.message}`)
    }
  }

  async function handleConfirmArchive() {
    if (!manageRoundData) return
    setActionLoading(true)
    setError('')
    try {
      await adminApi.archiveRound(manageRoundData.id, archiveReasonInput)
      setSuccessMsg(`Round #${manageRoundData.roundNumber} safely archived to research records.`)
      setManageModalOpen(false)
      await loadData()
    } catch (err) {
      setError(`Archival failed: ${err.message}`)
    } finally {
      setActionLoading(false)
    }
  }

  async function handleConfirmDelete() {
    if (!manageRoundData) return
    setActionLoading(true)
    setError('')
    try {
      await adminApi.deleteRound(manageRoundData.id)
      setSuccessMsg(`Round #${manageRoundData.roundNumber} safely deleted.`)
      setManageModalOpen(false)
      await loadData()
    } catch (err) {
      setError(`Deletion failed: ${err.message}`)
    } finally {
      setActionLoading(false)
    }
  }

  // Filtered rounds
  const filteredRounds = rounds.filter((r) => {
    if (filterStatus === 'CREATED' && r.status !== 'CREATED') return false
    if (filterStatus === 'ACTIVE' && r.status !== 'ACTIVE') return false
    if (filterStatus === 'COMPLETED' && r.status !== 'COMPLETED') return false
    if (filterStatus === 'ARCHIVED' && r.status !== 'ARCHIVED') return false
    if (filterStatus === 'CANCELLED' && r.status !== 'CANCELLED') return false
    if (filterStatus === 'FAILED' && r.status !== 'FAILED') return false

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase()
      const matchNum = `round #${r.roundNumber}`.includes(q)
      const matchName = (r.name || '').toLowerCase().includes(q)
      const matchModel = (r.baseModel?.version || '').toLowerCase().includes(q)
      return matchNum || matchName || matchModel
    }
    return true
  })

  // Status counts
  const statusCounts = {
    ALL: rounds.length,
    CREATED: rounds.filter((r) => r.status === 'CREATED').length,
    ACTIVE: rounds.filter((r) => r.status === 'ACTIVE').length,
    COMPLETED: rounds.filter((r) => r.status === 'COMPLETED').length,
    ARCHIVED: rounds.filter((r) => r.status === 'ARCHIVED').length,
    CANCELLED: rounds.filter((r) => r.status === 'CANCELLED').length,
    FAILED: rounds.filter((r) => r.status === 'FAILED').length,
  }

  if (loading) return <div className="card text-center py-16 text-gray-500">Loading training rounds...</div>

  return (
    <div className="space-y-6 max-w-6xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <RefreshCw className="text-arc-600" size={26} />
            Federated Round Management
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Configure round parameters, select participating edge clients, and monitor the training-to-aggregation lifecycle.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={loadData}
            disabled={actionLoading}
            className="btn-secondary flex items-center gap-1.5 text-xs py-2 px-3"
          >
            <RefreshCw size={14} className={actionLoading ? 'animate-spin' : ''} /> Refresh
          </button>

          <button
            onClick={openCreateModal}
            className="btn-primary flex items-center gap-1.5 text-sm py-2 px-4 shadow-sm font-semibold"
          >
            <Plus size={16} /> Create New Round
          </button>
        </div>
      </div>

      {error && (
        <div className="card bg-red-50 text-red-600 p-4 text-sm flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
          <button onClick={() => setError('')}><X size={16} /></button>
        </div>
      )}

      {successMsg && (
        <div className="card bg-green-50 text-green-700 p-4 text-sm flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 size={18} />
            <span>{successMsg}</span>
          </div>
          <button onClick={() => setSuccessMsg('')}><X size={16} /></button>
        </div>
      )}

      {/* Filter Tabs & Search Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-1 bg-gray-100 dark:bg-gray-800 p-1 rounded-xl overflow-x-auto">
          {[
            { id: 'ALL', label: 'All' },
            { id: 'CREATED', label: 'Pending / Created' },
            { id: 'ACTIVE', label: 'Active / Running' },
            { id: 'COMPLETED', label: 'Completed' },
            { id: 'ARCHIVED', label: 'Archived' },
            { id: 'CANCELLED', label: 'Cancelled' },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => setFilterStatus(tab.id)}
              className={`px-3 py-1.5 text-xs font-semibold rounded-lg transition-all whitespace-nowrap flex items-center gap-1.5 ${
                filterStatus === tab.id
                  ? 'bg-white dark:bg-gray-700 text-arc-600 dark:text-arc-300 shadow-sm'
                  : 'text-gray-500 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              <span>{tab.label}</span>
              <span className="text-[10px] px-1.5 py-0.2 rounded-full bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300">
                {statusCounts[tab.id] || 0}
              </span>
            </button>
          ))}
        </div>

        <div className="relative w-full sm:w-64">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
          <input
            type="text"
            placeholder="Search rounds or models..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="input pl-9 text-xs py-2 w-full"
          />
        </div>
      </div>

      {/* Rounds Progression Timeline / Cards */}
      <div className="space-y-4">
        {filteredRounds.length === 0 ? (
          <div className="card text-center py-16 text-gray-500">
            <Layers size={40} className="mx-auto text-gray-300 dark:text-gray-600 mb-2" />
            <p className="font-semibold text-gray-700 dark:text-gray-200">No Rounds Found</p>
            <p className="text-xs text-gray-400 mt-1">
              {rounds.length === 0 ? 'Click "Create New Round" to configure and initialize a training round.' : 'Try adjusting your search or status filter.'}
            </p>
          </div>
        ) : (
          filteredRounds.map((round) => {
            const isCreated = round.status === 'CREATED'
            const isActive = round.status === 'ACTIVE'
            const isCompleted = round.status === 'COMPLETED'
            const isArchived = round.status === 'ARCHIVED'
            const isCancelled = round.status === 'CANCELLED'
            const isFailed = round.status === 'FAILED'

            const hp = round.hyperparameters || {}
            const arc = round.arcWeights || {}

            return (
              <div
                key={round.id}
                className={`card border transition-all shadow-sm ${
                  isActive
                    ? 'border-blue-300 dark:border-blue-700 bg-blue-50/10'
                    : isCreated
                    ? 'border-amber-300 dark:border-amber-700 bg-amber-50/10'
                    : isArchived
                    ? 'border-purple-200 dark:border-purple-900 bg-purple-50/5 opacity-80'
                    : 'border-gray-200 dark:border-gray-700'
                }`}
              >
                {/* Header */}
                <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-gray-100 dark:border-gray-800 pb-3">
                  <div className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center font-bold text-sm ${
                      isCompleted
                        ? 'bg-green-100 text-green-700 dark:bg-green-900/40'
                        : isArchived
                        ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/40'
                        : isActive
                        ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/40 animate-pulse'
                        : isCancelled
                        ? 'bg-gray-100 text-gray-500 dark:bg-gray-800'
                        : 'bg-amber-100 text-amber-700 dark:bg-amber-900/40'
                    }`}>
                      #{round.roundNumber}
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="font-bold text-gray-900 dark:text-white text-base">
                          {round.name || `Training Round #${round.roundNumber}`}
                        </h3>
                        <span className={`text-xs font-semibold px-2.5 py-0.5 rounded-full ${
                          isCompleted
                            ? 'bg-green-100 text-green-700 dark:bg-green-900/40'
                            : isArchived
                            ? 'bg-purple-100 text-purple-700 dark:bg-purple-900/40'
                            : isActive
                            ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/40'
                            : isCreated
                            ? 'bg-amber-100 text-amber-700 dark:bg-amber-900/40'
                            : isCancelled
                            ? 'bg-gray-100 text-gray-600 dark:bg-gray-800'
                            : 'bg-red-100 text-red-700'
                        }`}>
                          {round.status === 'CREATED' ? 'PENDING' : round.status}
                        </span>
                      </div>
                      <p className="text-xs text-gray-500 mt-0.5">
                        Base: <span className="font-mono text-arc-600 font-semibold">{round.baseModel?.version || 'initial-v1'}</span>
                        {round.description && <span> • {round.description}</span>}
                      </p>
                    </div>
                  </div>

                  {/* Top Right Timestamps & Quick Actions */}
                  <div className="flex items-center gap-2">
                    <span className="text-xs text-gray-400">
                      {round.startedAt ? `Started: ${new Date(round.startedAt).toLocaleString()}` : 'Not started yet'}
                    </span>
                    <button
                      onClick={() => openDetailsModal(round.id)}
                      className="btn-secondary text-xs py-1.5 px-2.5 flex items-center gap-1"
                    >
                      <Eye size={12} /> Details
                    </button>
                  </div>
                </div>

                {/* Grid of parameters */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 py-3 text-sm">
                  <div>
                    <span className="text-xs text-gray-500">Participating Clients</span>
                    <p className="font-bold text-gray-900 dark:text-white mt-0.5 flex items-center gap-1.5">
                      <Cpu size={14} className="text-arc-600" />
                      {round.targetClients && Array.isArray(round.targetClients)
                        ? `${round.targetClients.length} clients (${round.targetClients.join(', ')})`
                        : `${round.clientCount || 0} clients`}
                    </p>
                  </div>

                  <div>
                    <span className="text-xs text-gray-500">Local Hyperparameters</span>
                    <p className="font-mono text-xs font-semibold text-gray-800 dark:text-gray-200 mt-1">
                      {hp.epochs || 2} epochs • b={hp.batchSize || 32} • lr={hp.learningRate || 0.001}
                    </p>
                  </div>

                  <div>
                    <span className="text-xs text-gray-500">ARC Weights (α, β, γ)</span>
                    <p className="font-mono text-xs font-semibold text-arc-600 mt-1">
                      α:{arc.lambdaRelevance ?? 0.40} • β:{arc.lambdaResource ?? 0.30} • γ:{arc.lambdaCommunication ?? 0.30}
                    </p>
                  </div>

                  <div>
                    <span className="text-xs text-gray-500">Aggregation Strategy</span>
                    <p className="font-mono text-xs font-bold text-indigo-600 dark:text-indigo-400 mt-1">
                      {round.aggregationStrategy || 'ARC_WEIGHTED'}
                    </p>
                  </div>
                </div>

                {/* Action Buttons Row */}
                <div className="flex flex-wrap items-center justify-between gap-2 pt-3 border-t border-gray-100 dark:border-gray-800">
                  <div className="flex items-center gap-2 text-xs text-gray-500">
                    <span>Submissions: <strong className="text-gray-900 dark:text-white">{round._count?.submissions || 0}</strong></span>
                    {round.resultingModel && (
                      <span className="flex items-center gap-1 text-green-600 font-semibold">
                        <CheckCircle2 size={12} /> Published as {round.resultingModel.version}
                      </span>
                    )}
                  </div>

                  <div className="flex items-center gap-2">
                    {/* START ROUND (Pending rounds only) */}
                    {isCreated && (
                      <button
                        onClick={() => handleStartRound(round)}
                        disabled={actionLoading}
                        className="btn-primary text-xs py-1.5 px-3 flex items-center gap-1 shadow-sm font-bold bg-green-600 hover:bg-green-700"
                      >
                        <Play size={13} /> Start Round
                      </button>
                    )}

                    {/* EDIT ROUND (Pending rounds only) */}
                    {isCreated && (
                      <button
                        onClick={() => openEditModal(round)}
                        disabled={actionLoading}
                        className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1 text-amber-600 hover:text-amber-700"
                      >
                        <Edit3 size={13} /> Edit Config
                      </button>
                    )}

                    {/* CANCEL ROUND (Created or Active rounds) */}
                    {(isCreated || isActive) && (
                      <button
                        onClick={() => handleCancelRound(round)}
                        disabled={actionLoading}
                        className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1 text-red-600 hover:text-red-700"
                      >
                        <Ban size={13} /> Cancel Round
                      </button>
                    )}

                    {/* DELETE / MANAGE ROUND (Created, Cancelled, or Failed rounds) */}
                    {(isCreated || isCancelled || isFailed) && (
                      <button
                        onClick={() => handleOpenManageModal(round)}
                        disabled={actionLoading}
                        className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1 text-red-600 hover:bg-red-50"
                        title="Inspect dependencies and delete or manage round"
                      >
                        <Trash2 size={13} /> Delete / Manage
                      </button>
                    )}

                    {/* ARCHIVE ROUND (Completed rounds) */}
                    {isCompleted && (
                      <button
                        onClick={() => handleOpenManageModal(round)}
                        disabled={actionLoading}
                        className="btn-secondary text-xs py-1.5 px-3 flex items-center gap-1 text-purple-600 hover:bg-purple-50"
                        title="Archive this completed round to historical research records"
                      >
                        <Archive size={13} /> Archive Round
                      </button>
                    )}

                    {/* PROTECTED HISTORY BADGE (Completed rounds) */}
                    {isCompleted && (
                      <span className="inline-flex items-center gap-1 text-xs text-gray-400 font-medium px-2.5 py-1 bg-gray-100 dark:bg-gray-800 rounded-md">
                        <ShieldCheck size={13} className="text-green-600" /> Research Record
                      </span>
                    )}

                    {/* ARCHIVED BADGE */}
                    {isArchived && (
                      <span className="inline-flex items-center gap-1 text-xs text-purple-600 font-medium px-2.5 py-1 bg-purple-50 dark:bg-purple-950/40 rounded-md border border-purple-200 dark:border-purple-800">
                        <Archive size={13} /> Archived Record
                      </span>
                    )}
                  </div>
                </div>
              </div>
            )
          })
        )}
      </div>

      {/* CREATE ROUND MODAL */}
      {createModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm overflow-y-auto">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 max-w-2xl w-full p-6 space-y-5 my-8">
            <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-3">
              <div>
                <h3 className="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
                  <Sliders className="text-arc-600" size={20} />
                  {isReviewStep ? 'Review & Confirm Round Configuration' : 'Configure New Federated Training Round'}
                </h3>
                <p className="text-xs text-gray-500 mt-0.5">
                  {isReviewStep ? 'Verify round settings before saving as Pending.' : 'Customize participants, ARC contribution coefficients, and hyperparameters.'}
                </p>
              </div>
              <button onClick={() => setCreateModalOpen(false)} className="text-gray-400 hover:text-gray-600">
                <X size={20} />
              </button>
            </div>

            {!isReviewStep ? (
              <div className="space-y-4 max-h-[65vh] overflow-y-auto pr-1">
                {/* 1. Basic Info */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Round Name *</label>
                    <input
                      type="text"
                      value={formState.name}
                      onChange={(e) => setFormState({ ...formState, name: e.target.value })}
                      className="input text-xs w-full"
                      placeholder="e.g. Federated Round #2"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Base Global Model *</label>
                    <select
                      value={formState.baseModelId}
                      onChange={(e) => setFormState({ ...formState, baseModelId: e.target.value })}
                      className="input text-xs w-full font-mono"
                      disabled={modelsList.length === 0}
                    >
                      {modelsList.length === 0 ? (
                        <option value="" disabled>No registered global models available</option>
                      ) : (
                        modelsList.map((m) => (
                          <option key={m.id} value={m.id}>
                            {m.version} — {m.name || 'ResMLP CICIoT2023'} {m.isActive ? '★ [Active Base]' : '[Archived]'}
                          </option>
                        ))
                      )}
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Description (Optional)</label>
                  <input
                    type="text"
                    value={formState.description}
                    onChange={(e) => setFormState({ ...formState, description: e.target.value })}
                    className="input text-xs w-full"
                    placeholder="Brief research purpose or edge scenario"
                  />
                </div>

                {/* 2. Participating Clients */}
                <div>
                  <div className="flex items-center justify-between mb-1.5">
                    <label className="text-xs font-semibold text-gray-700 dark:text-gray-300">
                      Participating Edge Clients * ({formState.clientIds.length} selected)
                    </label>
                    <span className="text-[11px] text-arc-600 font-medium">Auto-bound to assigned edge partitions</span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2">
                    {clientsList.map((c) => {
                      const isSelected = formState.clientIds.includes(c.clientId)
                      return (
                        <div
                          key={c.clientId}
                          onClick={() => handleClientCheckbox(c.clientId)}
                          className={`p-2.5 rounded-xl border cursor-pointer transition-all flex items-center justify-between ${
                            isSelected
                              ? 'border-arc-500 bg-arc-50/40 dark:bg-arc-900/20 text-arc-700 dark:text-arc-300 font-semibold'
                              : 'border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-400'
                          }`}
                        >
                          <div className="flex items-center gap-2">
                            <input
                              type="checkbox"
                              checked={isSelected}
                              onChange={() => {}}
                              className="rounded text-arc-600 pointer-events-none"
                            />
                            <div>
                              <p className="text-xs font-bold">{c.name}</p>
                              <p className="text-[10px] font-mono opacity-70">{c.clientId}</p>
                            </div>
                          </div>
                          <span className={`text-[10px] px-1.5 py-0.5 rounded ${c.status === 'ACTIVE' ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'}`}>
                            {c.status}
                          </span>
                        </div>
                      )
                    })}
                  </div>
                </div>

                {/* 3. ARC Weights Configuration */}
                <div className="p-4 bg-gray-50 dark:bg-gray-900/50 rounded-xl border border-gray-200 dark:border-gray-700 space-y-3">
                  <div className="flex items-center justify-between">
                    <div>
                      <h4 className="text-xs font-bold text-gray-900 dark:text-white flex items-center gap-1.5">
                        <Award size={14} className="text-arc-600" />
                        ARC Contribution Formula Weights
                      </h4>
                      <p className="text-[11px] text-gray-500 font-mono mt-0.5">ARC_i = α·Relevance + β·Resource + γ·Communication</p>
                    </div>

                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-semibold text-gray-600 dark:text-gray-300">Sum:</span>
                      <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded ${
                        isArcValid ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                      }`}>
                        {arcSum.toFixed(2)} / 1.00 {isArcValid ? '✓' : '⚠️'}
                      </span>
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-3">
                    <div>
                      <label className="block text-[11px] font-semibold text-gray-600 dark:text-gray-400 mb-1">α (Relevance)</label>
                      <input
                        type="number"
                        step="0.05"
                        min="0"
                        max="1"
                        value={formState.lambdaRelevance}
                        onChange={(e) => setFormState({ ...formState, lambdaRelevance: parseFloat(e.target.value) || 0 })}
                        className="input text-xs w-full font-mono text-center"
                      />
                    </div>
                    <div>
                      <label className="block text-[11px] font-semibold text-gray-600 dark:text-gray-400 mb-1">β (Resource)</label>
                      <input
                        type="number"
                        step="0.05"
                        min="0"
                        max="1"
                        value={formState.lambdaResource}
                        onChange={(e) => setFormState({ ...formState, lambdaResource: parseFloat(e.target.value) || 0 })}
                        className="input text-xs w-full font-mono text-center"
                      />
                    </div>
                    <div>
                      <label className="block text-[11px] font-semibold text-gray-600 dark:text-gray-400 mb-1">γ (Communication)</label>
                      <input
                        type="number"
                        step="0.05"
                        min="0"
                        max="1"
                        value={formState.lambdaCommunication}
                        onChange={(e) => setFormState({ ...formState, lambdaCommunication: parseFloat(e.target.value) || 0 })}
                        className="input text-xs w-full font-mono text-center"
                      />
                    </div>
                  </div>
                </div>

                {/* 4. Hyperparameters & Strategy */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div>
                    <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Local Epochs</label>
                    <input
                      type="number"
                      min="1"
                      max="50"
                      value={formState.epochs}
                      onChange={(e) => setFormState({ ...formState, epochs: parseInt(e.target.value) || 1 })}
                      className="input text-xs w-full font-mono"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Batch Size</label>
                    <input
                      type="number"
                      min="16"
                      max="1024"
                      step="16"
                      value={formState.batchSize}
                      onChange={(e) => setFormState({ ...formState, batchSize: parseInt(e.target.value) || 16 })}
                      className="input text-xs w-full font-mono"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Learning Rate</label>
                    <input
                      type="number"
                      step="0.0001"
                      min="0.00001"
                      max="1.0"
                      value={formState.learningRate}
                      onChange={(e) => setFormState({ ...formState, learningRate: parseFloat(e.target.value) || 0.0001 })}
                      className="input text-xs w-full font-mono"
                    />
                  </div>

                  <div>
                    <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Strategy</label>
                    <select
                      value={formState.aggregationStrategy}
                      onChange={(e) => setFormState({ ...formState, aggregationStrategy: e.target.value })}
                      className="input text-xs w-full font-bold text-arc-600"
                    >
                      <option value="ARC_WEIGHTED">ARC-Weighted</option>
                      <option value="FEDAVG">Standard FedAvg</option>
                      <option value="EQUAL">Equal-Weight</option>
                    </select>
                  </div>
                </div>
              </div>
            ) : (
              /* Review Step */
              <div className="space-y-4 max-h-[65vh] overflow-y-auto">
                <div className="p-4 bg-arc-50/50 dark:bg-arc-900/20 rounded-xl border border-arc-200 dark:border-arc-800 text-xs space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Round Name:</span>
                    <strong className="text-gray-900 dark:text-white">{formState.name}</strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Base Model:</span>
                    <strong className="font-mono text-arc-600">
                      {modelsList.find((m) => m.id === formState.baseModelId)?.version || 'initial-v1'}
                    </strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Participating Clients ({formState.clientIds.length}):</span>
                    <strong className="text-gray-900 dark:text-white font-mono">{formState.clientIds.join(', ')}</strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">ARC Weights:</span>
                    <strong className="font-mono text-arc-700">
                      α:{formState.lambdaRelevance} + β:{formState.lambdaResource} + γ:{formState.lambdaCommunication} = 1.00
                    </strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Hyperparameters:</span>
                    <strong className="font-mono text-gray-800 dark:text-gray-200">
                      {formState.epochs} epochs • batch={formState.batchSize} • lr={formState.learningRate}
                    </strong>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Aggregation Strategy:</span>
                    <strong className="font-bold text-indigo-600">{formState.aggregationStrategy}</strong>
                  </div>
                </div>

                <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg text-xs text-amber-800 dark:text-amber-200 flex items-start gap-2">
                  <Info size={16} className="flex-shrink-0 mt-0.5" />
                  <span>
                    Confirming will save this round as <strong>PENDING (CREATED)</strong>. Edge training will not begin until you click <strong>Start Round</strong>.
                  </span>
                </div>
              </div>
            )}

            {/* Modal Footer Controls */}
            <div className="flex items-center justify-between border-t border-gray-100 dark:border-gray-700 pt-3">
              <button
                type="button"
                onClick={() => {
                  if (isReviewStep) setIsReviewStep(false)
                  else setCreateModalOpen(false)
                }}
                className="btn-secondary text-xs py-2 px-4"
              >
                {isReviewStep ? '← Back to Config' : 'Cancel'}
              </button>

              {!isReviewStep ? (
                <button
                  type="button"
                  onClick={() => setIsReviewStep(true)}
                  disabled={!isArcValid || formState.clientIds.length === 0}
                  className="btn-primary text-xs py-2 px-4 flex items-center gap-1.5 font-bold shadow-sm"
                >
                  Review Configuration →
                </button>
              ) : (
                <button
                  type="button"
                  onClick={handleConfirmCreate}
                  disabled={actionLoading}
                  className="btn-primary text-xs py-2 px-5 flex items-center gap-1.5 font-bold shadow-sm bg-green-600 hover:bg-green-700"
                >
                  <Check size={14} /> Confirm & Save Round
                </button>
              )}
            </div>
          </div>
        </div>
      )}

      {/* EDIT ROUND MODAL */}
      {editModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm overflow-y-auto">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 max-w-2xl w-full p-6 space-y-4 my-8">
            <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-3">
              <h3 className="text-base font-bold text-gray-900 dark:text-white flex items-center gap-2">
                <Edit3 className="text-amber-600" size={18} />
                Edit Pending Round Configuration
              </h3>
              <button onClick={() => setEditModalOpen(false)}><X size={18} /></button>
            </div>

            <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-1">
              <div>
                <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Round Name</label>
                <input
                  type="text"
                  value={formState.name}
                  onChange={(e) => setFormState({ ...formState, name: e.target.value })}
                  className="input text-xs w-full"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Description</label>
                <input
                  type="text"
                  value={formState.description}
                  onChange={(e) => setFormState({ ...formState, description: e.target.value })}
                  className="input text-xs w-full"
                />
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Epochs</label>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    value={formState.epochs}
                    onChange={(e) => setFormState({ ...formState, epochs: parseInt(e.target.value) || 1 })}
                    className="input text-xs w-full font-mono"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Batch Size</label>
                  <input
                    type="number"
                    min="16"
                    max="1024"
                    value={formState.batchSize}
                    onChange={(e) => setFormState({ ...formState, batchSize: parseInt(e.target.value) || 16 })}
                    className="input text-xs w-full font-mono"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">Learning Rate</label>
                  <input
                    type="number"
                    step="0.0001"
                    min="0.00001"
                    max="1.0"
                    value={formState.learningRate}
                    onChange={(e) => setFormState({ ...formState, learningRate: parseFloat(e.target.value) || 0.0001 })}
                    className="input text-xs w-full font-mono"
                  />
                </div>
              </div>

              <div className="p-3 bg-gray-50 dark:bg-gray-900/50 rounded-xl border space-y-2">
                <div className="flex justify-between items-center text-xs font-bold">
                  <span>ARC Weights (α + β + γ = 1.0)</span>
                  <span className={`font-mono ${isArcValid ? 'text-green-600' : 'text-red-600'}`}>{arcSum.toFixed(2)}</span>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  <input
                    type="number"
                    step="0.05"
                    value={formState.lambdaRelevance}
                    onChange={(e) => setFormState({ ...formState, lambdaRelevance: parseFloat(e.target.value) || 0 })}
                    className="input text-xs font-mono text-center"
                    placeholder="α"
                  />
                  <input
                    type="number"
                    step="0.05"
                    value={formState.lambdaResource}
                    onChange={(e) => setFormState({ ...formState, lambdaResource: parseFloat(e.target.value) || 0 })}
                    className="input text-xs font-mono text-center"
                    placeholder="β"
                  />
                  <input
                    type="number"
                    step="0.05"
                    value={formState.lambdaCommunication}
                    onChange={(e) => setFormState({ ...formState, lambdaCommunication: parseFloat(e.target.value) || 0 })}
                    className="input text-xs font-mono text-center"
                    placeholder="γ"
                  />
                </div>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 border-t border-gray-100 dark:border-gray-700 pt-3">
              <button onClick={() => setEditModalOpen(false)} className="btn-secondary text-xs py-2 px-4">Cancel</button>
              <button
                onClick={handleConfirmEdit}
                disabled={actionLoading || !isArcValid}
                className="btn-primary text-xs py-2 px-4 font-bold"
              >
                Save Changes
              </button>
            </div>
          </div>
        </div>
      )}

      {/* FULL ROUND DETAILS MODAL */}
      {detailsModalOpen && activeRoundDetails && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm overflow-y-auto">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 max-w-3xl w-full p-6 space-y-4 my-8">
            <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-3">
              <div>
                <h3 className="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
                  <Info className="text-arc-600" size={20} />
                  Round #{activeRoundDetails.roundNumber}: {activeRoundDetails.name || 'Federated Training Round'}
                </h3>
                <p className="text-xs font-mono text-gray-400 mt-0.5">ID: {activeRoundDetails.id}</p>
              </div>
              <button onClick={() => setDetailsModalOpen(false)}><X size={20} /></button>
            </div>

            <div className="space-y-4 max-h-[70vh] overflow-y-auto pr-1 text-xs">
              {/* Metadata strip */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 p-3 bg-gray-50 dark:bg-gray-900/50 rounded-xl">
                <div>
                  <span className="text-gray-400">Status</span>
                  <p className="font-bold text-gray-900 dark:text-white mt-0.5">{activeRoundDetails.status}</p>
                </div>
                <div>
                  <span className="text-gray-400">Base Model</span>
                  <p className="font-mono font-bold text-arc-600 mt-0.5">{activeRoundDetails.baseModel?.version}</p>
                </div>
                <div>
                  <span className="text-gray-400">Created At</span>
                  <p className="mt-0.5">{new Date(activeRoundDetails.createdAt).toLocaleString()}</p>
                </div>
                <div>
                  <span className="text-gray-400">Started At</span>
                  <p className="mt-0.5">{activeRoundDetails.startedAt ? new Date(activeRoundDetails.startedAt).toLocaleString() : 'Pending'}</p>
                </div>
              </div>

              {/* Base Model Integrity */}
              <div className="p-3 bg-arc-50/50 dark:bg-arc-900/20 rounded-xl border border-arc-200 dark:border-arc-800">
                <span className="text-gray-500 font-semibold block mb-1">Base Model Architecture & Checksum:</span>
                <p className="font-mono text-[11px] text-gray-800 dark:text-gray-200 break-all">
                  SHA-256: {activeRoundDetails.baseModel?.checksumSha256 || 'Verified'}
                </p>
                <p className="text-gray-500 mt-1">
                  Architecture: <span className="font-mono font-bold">{activeRoundDetails.baseModel?.architecture || 'preact_resmlp_v3'}</span> • Baseline Accuracy: <span className="font-bold text-green-600">{activeRoundDetails.baseModel?.baselineAccuracy}%</span>
                </p>
              </div>

              {/* Submissions Section */}
              <div>
                <h4 className="font-bold text-gray-900 dark:text-white text-sm mb-2 flex items-center justify-between">
                  <span>Client Submissions ({activeRoundDetails.submissions?.length || 0})</span>
                  <span className="text-xs text-gray-400 font-normal">Edge Parameter Deltas</span>
                </h4>
                {activeRoundDetails.submissions?.length === 0 ? (
                  <div className="p-4 text-center text-gray-400 border rounded-xl">No submissions received for this round yet.</div>
                ) : (
                  <div className="space-y-2">
                    {activeRoundDetails.submissions.map((sub) => {
                      const arc = sub.arcScores?.[0]
                      return (
                        <div key={sub.id} className="p-3 border rounded-xl flex items-center justify-between bg-white dark:bg-gray-800">
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="font-bold text-gray-900 dark:text-white">{sub.client?.name || sub.clientId}</span>
                              <span className="font-mono text-[10px] px-1.5 py-0.5 bg-gray-100 dark:bg-gray-700 rounded">{sub.clientId}</span>
                              <span className={`text-[10px] px-1.5 py-0.5 rounded font-semibold ${sub.status === 'AGGREGATED' ? 'bg-indigo-100 text-indigo-700' : 'bg-green-100 text-green-700'}`}>
                                {sub.status}
                              </span>
                            </div>
                            <p className="text-[11px] text-gray-500 mt-0.5 font-mono">
                              Acc: <strong className="text-green-600">{sub.localAccuracy?.toFixed(2)}%</strong> • Loss: {sub.localLoss?.toFixed(4)} • Samples: {sub.samplesTrained} • Time: {sub.trainingDurationSeconds}s
                            </p>
                          </div>

                          <div className="text-right">
                            <span className="text-[10px] text-gray-400">ARC Score</span>
                            <p className="text-sm font-mono font-bold text-arc-600">
                              {arc ? arc.overallArcScore?.toFixed(2) : sub.arcScore?.toFixed(2) || 'N/A'}
                            </p>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            </div>

            <div className="border-t border-gray-100 dark:border-gray-700 pt-3 flex justify-end">
              <button onClick={() => setDetailsModalOpen(false)} className="btn-secondary text-xs py-2 px-4">Close</button>
            </div>
          </div>
        </div>
      )}

      {/* MANAGE / ARCHIVE / SAFE DELETE MODAL WITH DEPENDENCY AUDIT */}
      {manageModalOpen && manageRoundData && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-gray-900 rounded-2xl max-w-xl w-full p-6 shadow-2xl border border-gray-200 dark:border-gray-800 space-y-4">
            <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-800 pb-3">
              <div className="flex items-center gap-2">
                <FolderArchive className="text-arc-600" size={22} />
                <h3 className="text-lg font-black text-gray-900 dark:text-white">
                  Round Lifecycle Management & Dependencies
                </h3>
              </div>
              <button onClick={() => setManageModalOpen(false)} className="text-gray-400 hover:text-gray-600">
                <X size={18} />
              </button>
            </div>

            <div className="p-3 bg-gray-50 dark:bg-gray-800/60 rounded-xl space-y-1.5 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-gray-500">Target Round:</span>
                <span className="font-bold text-gray-900 dark:text-white">Round #{manageRoundData.roundNumber} ({manageRoundData.name || 'Untitled'})</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500">Current Status:</span>
                <span className="font-bold font-mono text-arc-600">{manageRoundData.status}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-500">Base Model:</span>
                <span className="font-mono text-gray-700 dark:text-gray-300">{manageRoundData.baseModel?.version || 'initial-v1'}</span>
              </div>
            </div>

            {/* Dependency Audit Summary */}
            {manageDependencies ? (
              <div className="space-y-3">
                <h4 className="text-xs font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider">
                  Dependency Audit Results
                </h4>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="p-2.5 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
                    <span className="text-gray-400 block text-[10px] uppercase">Client Submissions</span>
                    <strong className="text-sm font-mono text-gray-900 dark:text-white">
                      {manageDependencies.dependencies.submissionCount} submissions
                    </strong>
                  </div>

                  <div className="p-2.5 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
                    <span className="text-gray-400 block text-[10px] uppercase">Aggregation Jobs</span>
                    <strong className="text-sm font-mono text-gray-900 dark:text-white">
                      {manageDependencies.dependencies.aggregationCount} jobs
                    </strong>
                  </div>
                </div>

                {manageDependencies.dependencies.hasResultingModel && (
                  <div className="p-3 bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-800 rounded-lg text-xs text-blue-700">
                    <strong>Resulting Global Model:</strong> Generated production model <code className="font-bold">{manageDependencies.dependencies.resultingModelVersion}</code>
                    {manageDependencies.dependencies.isResultingModelActive && (
                      <span className="block text-red-600 font-bold mt-1">
                        ⚠️ This is currently the active fleet production model. Physical deletion is strictly blocked.
                      </span>
                    )}
                  </div>
                )}

                {/* Permissions & Recommendations */}
                {manageDependencies.permissions.blockReason && (
                  <div className="p-3 bg-red-50 dark:bg-red-950/30 border border-red-200 dark:border-red-800 rounded-lg text-xs text-red-600">
                    {manageDependencies.permissions.blockReason}
                  </div>
                )}

                {/* Archival Reason Input */}
                {manageDependencies.permissions.canArchive && (
                  <div className="space-y-1.5 pt-2">
                    <label className="text-xs font-bold text-gray-700 dark:text-gray-300">
                      Archival Note / Reason (Optional):
                    </label>
                    <input
                      type="text"
                      placeholder="e.g. Preserved for academic benchmark comparison..."
                      value={archiveReasonInput}
                      onChange={(e) => setArchiveReasonInput(e.target.value)}
                      className="w-full text-xs p-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
                    />
                  </div>
                )}
              </div>
            ) : (
              <div className="py-6 text-center text-xs text-gray-500">Checking round dependencies and active relations...</div>
            )}

            {/* Modal Actions */}
            <div className="flex items-center justify-end gap-2 pt-3 border-t border-gray-100 dark:border-gray-800">
              <button
                type="button"
                onClick={() => setManageModalOpen(false)}
                className="btn-secondary text-xs px-4 py-2"
              >
                Cancel
              </button>

              {manageDependencies?.permissions.canArchive && (
                <button
                  type="button"
                  disabled={actionLoading}
                  onClick={handleConfirmArchive}
                  className="btn-primary text-xs px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white font-bold flex items-center gap-1.5"
                >
                  <Archive size={14} />
                  {actionLoading ? 'Archiving...' : 'Archive to Historical Records'}
                </button>
              )}

              {manageDependencies?.permissions.canDelete && (
                <button
                  type="button"
                  disabled={actionLoading}
                  onClick={handleConfirmDelete}
                  className="btn-primary text-xs px-4 py-2 bg-red-600 hover:bg-red-700 text-white font-bold flex items-center gap-1.5"
                >
                  <Trash2 size={14} />
                  {actionLoading ? 'Deleting...' : 'Permanently Delete Round'}
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
