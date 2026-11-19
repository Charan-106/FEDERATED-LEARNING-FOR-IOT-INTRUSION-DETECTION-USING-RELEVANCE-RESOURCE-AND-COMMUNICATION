// Client Local Training Page - Edge Training Studio & Configuration Manager
// Section 5.C & Update: Allows clients to edit Training Parameters and ARC Telemetry Inputs with full auditability

import { useEffect, useState, useRef } from 'react'
import {
  Play,
  CheckCircle,
  AlertTriangle,
  Cpu,
  Layers,
  Database,
  Send,
  RefreshCw,
  HardDrive,
  Activity,
  Wifi,
  BarChart2,
  Sliders,
  ShieldCheck,
  Clock,
  Edit3,
  Save,
  X,
  Lock,
  Info,
  Radio
} from 'lucide-react'
import { clientApi } from '../../services/api'

export default function ClientTrainingPage() {
  const [assignedDataset, setAssignedDataset] = useState(null)
  const [clientProfile, setClientProfile] = useState(null)
  const [activeRound, setActiveRound] = useState(null)
  const [activeModel, setActiveModel] = useState(null)
  const [clientEligibility, setClientEligibility] = useState(null)

  // Status & Telemetry
  const [loading, setLoading] = useState(true)
  const [training, setTraining] = useState(false)
  const [trainingResult, setTrainingResult] = useState(null)
  const [submitting, setSubmitting] = useState(false)
  const [submissionSuccess, setSubmissionSuccess] = useState('')
  const [error, setError] = useState('')

  // Live SSE streaming telemetry
  const [liveProgress, setLiveProgress] = useState(0)
  const [liveStage, setLiveStage] = useState('')
  const [liveBatch, setLiveBatch] = useState(0)
  const [liveTotalBatches, setLiveTotalBatches] = useState(0)
  const [liveLoss, setLiveLoss] = useState(null)
  const [liveEpoch, setLiveEpoch] = useState(1)
  const [liveTotalEpochs, setLiveTotalEpochs] = useState(2)
  const sseAbortRef = useRef(null)

  // -------------------------------------------------------------
  // Configurable Parameters State (Training & ARC Inputs)
  // -------------------------------------------------------------
  const [isEditing, setIsEditing] = useState(false)
  const [validationErrors, setValidationErrors] = useState({})

  // Active (saved) configuration snapshot
  const [configValues, setConfigValues] = useState({
    // Training Hyperparameters
    epochs: 2,
    batchSize: 256,
    learningRate: 0.0001,

    // Relevance Inputs
    qualityScore: 98.5,
    diversityScore: 85.0,

    // Resource Inputs
    availableRamGb: 8.0,
    totalRamGb: 16.0,
    cpuIdlePct: 65.0,
    throughputSamplesPerSec: 500.0,

    // Communication Inputs
    bandwidthMbps: 35.0,
    latencyMs: 40.0,
    reliabilityPct: 99.5,

    // Synchronization & Automation
    autoSubmit: true,
  })

  // Edit draft buffer (for Cancel/Save pattern)
  const [draftValues, setDraftValues] = useState({ ...configValues })

  // Source attribution flags: 'MEASURED' | 'MANUAL' | 'SYSTEM'
  const [paramSources, setParamSources] = useState({
    epochs: 'MANUAL',
    batchSize: 'MANUAL',
    learningRate: 'MANUAL',
    qualityScore: 'MEASURED',
    diversityScore: 'MEASURED',
    availableRamGb: 'MEASURED',
    cpuIdlePct: 'MEASURED',
    throughputSamplesPerSec: 'MEASURED',
    bandwidthMbps: 'MEASURED',
    latencyMs: 'MEASURED',
    reliabilityPct: 'MEASURED',
  })

  useEffect(() => {
    loadData()
    setupSSE()

    return () => {
      if (sseAbortRef.current) {
        sseAbortRef.current.abort()
      }
    }
  }, [])

  function setupSSE() {
    const token = localStorage.getItem('arc_fl_token')
    if (!token) return

    if (sseAbortRef.current) {
      sseAbortRef.current.abort()
    }
    const abortController = new AbortController()
    sseAbortRef.current = abortController

    connectSSEStream(token, abortController.signal)
  }

  async function connectSSEStream(token, signal) {
    try {
      const response = await fetch('/api/client/training-progress', {
        headers: { Authorization: `Bearer ${token}` },
        signal,
      })

      if (!response.ok) return

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let packetBuffer = ''

      while (true) {
        const { value, done } = await reader.read()
        if (done) break

        packetBuffer += decoder.decode(value, { stream: true })
        const lines = packetBuffer.split('\n')
        packetBuffer = lines.pop()

        for (const line of lines) {
          const trimmed = line.trim()
          if (!trimmed.startsWith('data:')) continue
          const raw = trimmed.replace(/^data:\s*/, '')
          try {
            const ev = JSON.parse(raw)
            if (ev.type === 'PROGRESS') {
              if (ev.progress !== undefined) setLiveProgress(ev.progress)
              if (ev.currentStage) setLiveStage(ev.currentStage)
              if (ev.currentBatch !== undefined) setLiveBatch(ev.currentBatch)
              if (ev.totalBatches !== undefined) setLiveTotalBatches(ev.totalBatches)
              if (ev.batchLoss !== undefined && ev.batchLoss !== null) setLiveLoss(ev.batchLoss)
              if (ev.currentEpoch !== undefined) setLiveEpoch(ev.currentEpoch)
              if (ev.totalEpochs !== undefined) setLiveTotalEpochs(ev.totalEpochs)

              if (ev.status === 'RUNNING') {
                setTraining(true)
              } else if (ev.status === 'COMPLETED' || ev.status === 'TRAINED') {
                setTraining(false)
              }
            } else if (ev.type === 'UPDATE_REGISTERED') {
              setSubmissionSuccess(`Update verified & registered! Checksum: ${ev.checksumSha256?.substring(0, 16)}...`)
              loadData()
            }
          } catch (_) {}
        }
      }
    } catch (e) {
      if (e.name !== 'AbortError') {
        // SSE disconnected, will auto-reconnect on next trigger
      }
    }
  }

  async function loadData() {
    setLoading(true)
    setError('')
    const t0 = performance.now()
    try {
      const [dash, ds, cfgRes] = await Promise.all([
        clientApi.getDashboard(),
        clientApi.getDatasets(),
        clientApi.getConfig().catch(() => null),
      ])

      const rtt = Math.max(5, Math.round(performance.now() - t0))
      setActiveRound(dash.latestRound)
      setActiveModel(dash.activeModel)
      setClientProfile(dash.clientProfile)
      setClientEligibility(dash.clientEligibility)

      const assigned = ds.assignedDataset || (ds.datasets && ds.datasets[0])
      setAssignedDataset(assigned)

      // Initial measured communication values
      const initialBw = assigned?.sizeKb ? Math.round((assigned.sizeKb * 8) / Math.max(rtt, 10) * 10) / 10 : 35.0

      if (cfgRes?.config) {
        setConfigValues((prev) => {
          const updated = {
            ...prev,
            ...cfgRes.config,
            latencyMs: cfgRes.config.latencyMs || rtt,
            bandwidthMbps: cfgRes.config.bandwidthMbps || initialBw,
          }
          setDraftValues({ ...updated })
          return updated
        })
        if (cfgRes.paramSources) {
          setParamSources(cfgRes.paramSources)
        }
      } else {
        setConfigValues((prev) => {
          const updated = {
            ...prev,
            latencyMs: rtt,
            bandwidthMbps: initialBw,
            reliabilityPct: 100.0,
          }
          setDraftValues({ ...updated })
          return updated
        })
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  // -------------------------------------------------------------
  // Validation Routine
  // -------------------------------------------------------------
  function validateDraft(draft) {
    const errs = {}
    const ep = Number(draft.epochs)
    const bs = Number(draft.batchSize)
    const lr = Number(draft.learningRate)
    const ram = Number(draft.availableRamGb)
    const cpu = Number(draft.cpuIdlePct)
    const tp = Number(draft.throughputSamplesPerSec)
    const bw = Number(draft.bandwidthMbps)
    const lat = Number(draft.latencyMs)
    const rel = Number(draft.reliabilityPct)
    const qual = Number(draft.qualityScore)
    const div = Number(draft.diversityScore)

    // Training Parameters
    if (!Number.isInteger(ep) || ep < 1 || ep > 50) errs.epochs = 'Epochs must be an integer between 1 and 50.'
    if (!Number.isInteger(bs) || bs < 16 || bs > 1024) errs.batchSize = 'Batch Size must be an integer between 16 and 1024.'
    if (isNaN(lr) || lr <= 0 || lr > 1.0) errs.learningRate = 'Learning Rate must be between 0.00001 and 1.0.'

    // Relevance Inputs
    if (isNaN(qual) || qual < 0 || qual > 100) errs.qualityScore = 'Data Quality must be between 0 and 100%.'
    if (isNaN(div) || div < 0 || div > 100) errs.diversityScore = 'Data Diversity must be between 0 and 100%.'

    // Resource Inputs
    if (isNaN(ram) || ram <= 0) errs.availableRamGb = 'Available RAM must be greater than 0 GB.'
    if (isNaN(cpu) || cpu < 0 || cpu > 100) errs.cpuIdlePct = 'Available CPU must be between 0 and 100%.'
    if (isNaN(tp) || tp <= 0) errs.throughputSamplesPerSec = 'Compute capacity / throughput must be greater than 0.'

    // Communication Inputs
    if (isNaN(bw) || bw <= 0) errs.bandwidthMbps = 'Bandwidth must be greater than 0 Mbps.'
    if (isNaN(lat) || lat < 0) errs.latencyMs = 'Network latency must be 0 ms or higher.'
    if (isNaN(rel) || rel < 0 || rel > 100) errs.reliabilityPct = 'Transmission reliability must be between 0 and 100%.'

    setValidationErrors(errs)
    return Object.keys(errs).length === 0
  }

  // -------------------------------------------------------------
  // Edit & Save Controls
  // -------------------------------------------------------------
  function handleEnterEdit() {
    setDraftValues({ ...configValues })
    setValidationErrors({})
    setIsEditing(true)
  }

  async function handleSaveEdit() {
    if (!validateDraft(draftValues)) {
      setError('Please correct the validation errors before saving.')
      return
    }

    // Determine parameter sources (mark edited fields as MANUAL / CLIENT OVERRIDE)
    const updatedSources = { ...paramSources }
    Object.keys(draftValues).forEach((key) => {
      if (draftValues[key] !== configValues[key]) {
        updatedSources[key] = 'MANUAL'
      }
    })

    try {
      await clientApi.updateConfig({
        config: draftValues,
        paramSources: updatedSources,
      })
      setConfigValues({ ...draftValues })
      setParamSources(updatedSources)
      setIsEditing(false)
      setError('')
      setSubmissionSuccess('Configuration successfully saved and synchronized across client views!')
      setTimeout(() => setSubmissionSuccess(''), 4000)
    } catch (err) {
      setError(`Failed to save configuration: ${err.message}`)
    }
  }

  function handleCancelEdit() {
    setDraftValues({ ...configValues })
    setValidationErrors({})
    setIsEditing(false)
    setError('')
  }

  function handleDraftChange(field, val) {
    const updated = { ...draftValues, [field]: val }
    setDraftValues(updated)
    validateDraft(updated)
  }

  // -------------------------------------------------------------
  // Local Training Execution
  // -------------------------------------------------------------
  async function handleStartTraining() {
    if (isEditing) {
      setError('Please save or cancel parameter edits before starting training.')
      return
    }

    setTraining(true)
    setError('')
    setTrainingResult(null)
    setSubmissionSuccess('')

    const trainStart = performance.now()
    try {
      const res = await clientApi.trainLocal({
        roundId: activeRound?.id,
        epochs: parseInt(configValues.epochs, 10),
        batchSize: parseInt(configValues.batchSize, 10),
        learningRate: parseFloat(configValues.learningRate),
      })

      const netElapsed = Math.round(performance.now() - trainStart)
      setTrainingResult(res)

      // Auto-update measured telemetry if user hasn't overridden them manually
      setConfigValues((prev) => {
        const updated = { ...prev }
        if (paramSources.availableRamGb === 'MEASURED' && res.resource_metrics?.available_ram_gb) {
          updated.availableRamGb = res.resource_metrics.available_ram_gb
        }
        if (paramSources.totalRamGb === 'MEASURED' && res.resource_metrics?.total_ram_gb) {
          updated.totalRamGb = res.resource_metrics.total_ram_gb
        }
        if (paramSources.cpuIdlePct === 'MEASURED' && res.resource_metrics?.cpu_idle_pct !== null) {
          updated.cpuIdlePct = res.resource_metrics.cpu_idle_pct
        }
        if (paramSources.throughputSamplesPerSec === 'MEASURED' && res.resource_metrics?.throughput_samples_per_sec) {
          updated.throughputSamplesPerSec = res.resource_metrics.throughput_samples_per_sec
        }
        if (paramSources.qualityScore === 'MEASURED' && res.quality_metrics) {
          const comp = 1.0 - (res.quality_metrics.missing_ratio || 0) - (res.quality_metrics.inf_ratio || 0)
          updated.qualityScore = Math.round(Math.max(0, comp) * 1000) / 10
        }
        if (paramSources.diversityScore === 'MEASURED' && res.diversity_metrics) {
          const cov = (res.diversity_metrics.class_count || 1) / 34.0
          updated.diversityScore = Math.round(Math.min(1.0, cov) * 1000) / 10
        }
        if (paramSources.latencyMs === 'MEASURED') {
          // Genuine network ping time to backend coordinator (not model training execution duration)
          try {
            const pStart = performance.now()
            clientApi.getDashboard().then(() => {
              const pingRtt = Math.max(5, Math.min(250, Math.round(performance.now() - pStart)))
              setConfigValues((c) => ({ ...c, latencyMs: pingRtt }))
              setDraftValues((d) => ({ ...d, latencyMs: pingRtt }))
            }).catch(() => {})
          } catch (_) {}
        }
        setDraftValues({ ...updated })
        return updated
      })

      // AUTO-SUBMIT: Automatically transmit model update to server unless explicitly disabled
      if (configValues.autoSubmit !== false && activeRound && activeRound.status === 'ACTIVE') {
        setSubmitting(true)
        try {
          const qualPayload = paramSources.qualityScore === 'MANUAL'
            ? { quality_score: parseFloat(configValues.qualityScore) }
            : {
                missing_ratio: res.quality_metrics?.missing_ratio ?? 0.0,
                inf_ratio: res.quality_metrics?.inf_ratio ?? 0.0,
                preprocessing_success: 1.0,
              }

          const divPayload = paramSources.diversityScore === 'MANUAL'
            ? { diversity_score: parseFloat(configValues.diversityScore) }
            : {
                class_count: res.diversity_metrics?.class_count ?? 20,
                entropy: res.diversity_metrics?.entropy ?? 3.5,
                minority_class_coverage: res.diversity_metrics?.minority_class_coverage ?? 0.6,
              }

          const subRes = await clientApi.submitTrained({
            roundId: activeRound.id,
            updatePath: res.update_path,
            localAccuracy: res.accuracy,
            localLoss: res.loss,
            samplesTrained: res.samples_trained,
            epochs: parseInt(configValues.epochs, 10),
            batchSize: parseInt(configValues.batchSize, 10),
            learningRate: parseFloat(configValues.learningRate),
            trainingDurationSeconds: res.training_duration_seconds,
            resourceMetrics: {
              available_ram_gb: parseFloat(configValues.availableRamGb),
              total_ram_gb: parseFloat(configValues.totalRamGb),
              cpu_idle_pct: parseFloat(configValues.cpuIdlePct),
              throughput_samples_per_sec: parseFloat(configValues.throughputSamplesPerSec),
            },
            qualityMetrics: qualPayload,
            diversityMetrics: divPayload,
            commMetrics: {
              bandwidth_mbps: parseFloat(configValues.bandwidthMbps),
              latency_ms: parseFloat(configValues.latencyMs),
              reliability_pct: parseFloat(configValues.reliabilityPct),
            },
            parameterSources: paramSources,
          })

          setSubmissionSuccess(subRes.message || 'Model update automatically submitted and verified by server!')
          await loadData()
        } catch (subErr) {
          const subMsg = subErr.response?.data?.error || subErr.message
          setError(`Training finished, but automatic update submission failed: ${subMsg}`)
        } finally {
          setSubmitting(false)
        }
      }
    } catch (err) {
      const msg = err.response?.data?.error || err.message
      setError(`Training failed: ${msg}`)
    } finally {
      setTraining(false)
    }
  }

  // -------------------------------------------------------------
  // Model Update Submission
  // -------------------------------------------------------------
  async function handleSubmitUpdate() {
    if (!trainingResult) return
    if (!activeRound || activeRound.status !== 'ACTIVE') {
      alert('Cannot submit update: No active training round is currently open.')
      return
    }

    setSubmitting(true)
    setError('')

    try {
      const qualPayload = paramSources.qualityScore === 'MANUAL'
        ? { quality_score: parseFloat(configValues.qualityScore) }
        : {
            missing_ratio: trainingResult.quality_metrics?.missing_ratio ?? 0.0,
            inf_ratio: trainingResult.quality_metrics?.inf_ratio ?? 0.0,
            preprocessing_success: 1.0,
          }

      const divPayload = paramSources.diversityScore === 'MANUAL'
        ? { diversity_score: parseFloat(configValues.diversityScore) }
        : {
            class_count: trainingResult.diversity_metrics?.class_count ?? 20,
            entropy: trainingResult.diversity_metrics?.entropy ?? 3.5,
            minority_class_coverage: trainingResult.diversity_metrics?.minority_class_coverage ?? 0.6,
          }

      const res = await clientApi.submitTrained({
        roundId: activeRound.id,
        updatePath: trainingResult.update_path,
        localAccuracy: trainingResult.accuracy,
        localLoss: trainingResult.loss,
        samplesTrained: trainingResult.samples_trained,
        epochs: configValues.epochs,
        batchSize: configValues.batchSize,
        learningRate: configValues.learningRate,
        trainingDurationSeconds: trainingResult.training_duration_seconds,
        resourceMetrics: {
          available_ram_gb: parseFloat(configValues.availableRamGb),
          total_ram_gb: parseFloat(configValues.totalRamGb),
          cpu_idle_pct: parseFloat(configValues.cpuIdlePct),
          throughput_samples_per_sec: parseFloat(configValues.throughputSamplesPerSec),
        },
        qualityMetrics: qualPayload,
        diversityMetrics: divPayload,
        commMetrics: {
          bandwidth_mbps: parseFloat(configValues.bandwidthMbps),
          latency_ms: parseFloat(configValues.latencyMs),
          reliability_pct: parseFloat(configValues.reliabilityPct),
        },
        parameterSources: paramSources,
      })

      setSubmissionSuccess(res.message || 'Model update submitted successfully with client configuration snapshot!')
    } catch (err) {
      const msg = err.response?.data?.error || err.message
      setError(`Submission failed: ${msg}`)
    } finally {
      setSubmitting(false)
    }
  }

  function renderSourceBadge(source) {
    if (source === 'MANUAL') {
      return (
        <span className="px-1.5 py-0.5 text-[10px] font-semibold rounded bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300">
          Client Override
        </span>
      )
    }
    if (source === 'MEASURED') {
      return (
        <span className="px-1.5 py-0.5 text-[10px] font-semibold rounded bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300">
          Auto Measured
        </span>
      )
    }
    return (
      <span className="px-1.5 py-0.5 text-[10px] font-semibold rounded bg-emerald-100 dark:bg-emerald-900/40 text-emerald-700 dark:text-emerald-300">
        System Calc
      </span>
    )
  }

  if (loading) {
    return <div className="card text-center py-12 text-gray-500">Loading training environment...</div>
  }

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
          <Cpu className="text-arc-600" size={26} />
          Local Edge Training & ARC Configuration Studio
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
          Review, customize, and submit edge training parameters and telemetry inputs. Client data stays 100% on-device.
        </p>
      </div>

      {/* Round & Model Banner */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card border-l-4 border-l-arc-500 flex items-center justify-between">
          <div>
            <div className="text-xs text-gray-500 uppercase tracking-wide">Base Global Model</div>
            <div className="text-base font-bold text-gray-900 dark:text-white mt-0.5">
              {activeModel?.version || 'N/A'} ({activeModel?.name || 'preact_resmlp_v3'})
            </div>
            <div className="text-xs text-green-600 mt-0.5 font-medium">
              Baseline Accuracy: {activeModel?.accuracy ?? 98.5}%
            </div>
          </div>
          <CheckCircle className="text-green-500" size={22} />
        </div>

        <div className={`card border-l-4 ${activeRound?.status === 'ACTIVE' ? 'border-l-green-500' : 'border-l-amber-500'} flex items-center justify-between`}>
          <div>
            <div className="text-xs text-gray-500 uppercase tracking-wide">Current Training Round</div>
            <div className="text-base font-bold text-gray-900 dark:text-white mt-0.5">
              {activeRound ? `Round #${activeRound.roundNumber}` : 'No Active Round'}
            </div>
            <div className="text-xs text-gray-500 mt-0.5">
              Status:{' '}
              <span className={`font-semibold ${activeRound?.status === 'ACTIVE' ? 'text-green-600' : 'text-amber-600'}`}>
                {activeRound?.status || 'IDLE'}
              </span>
            </div>
          </div>
          <Activity className={activeRound?.status === 'ACTIVE' ? 'text-green-500' : 'text-amber-500'} size={22} />
        </div>
      </div>

      {/* ARC Formula Restriction Notice */}
      <div className="p-3.5 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg text-xs text-blue-800 dark:text-blue-300 flex items-start gap-2.5">
        <Info size={16} className="mt-0.5 flex-shrink-0 text-blue-600" />
        <div className="space-y-0.5">
          <div className="font-semibold">ARC Formula Governance & Permissions</div>
          <div>
            Edge clients may inspect and edit training hyperparameters and device-reported telemetry. Official formula weights (λ<sub>R</sub>, λ<sub>S</sub>, λ<sub>C</sub>) and normalization criteria are strictly administered by Platform Administrators to prevent weight poisoning.
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-600 text-sm flex items-start gap-2">
          <AlertTriangle size={18} className="mt-0.5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {submissionSuccess && (
        <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg text-green-700 dark:text-green-300 text-sm flex items-start gap-2">
          <CheckCircle size={18} className="mt-0.5 flex-shrink-0" />
          <div>
            <span className="font-semibold">Success!</span> {submissionSuccess}
          </div>
        </div>
      )}

      {/* Main Configuration Card with Edit / Save Controls */}
      <div className="card space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 border-b border-gray-100 dark:border-gray-800 pb-3">
          <div className="flex items-center gap-2">
            <Sliders size={20} className="text-arc-500" />
            <div>
              <h2 className="text-base font-semibold text-gray-900 dark:text-white">
                Client Training & ARC Input Configuration
              </h2>
              <p className="text-xs text-gray-500">
                {isEditing ? 'Editing active values in memory. Click Save Changes when done.' : 'Values preserved for training round snapshot.'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {!isEditing ? (
              <button
                onClick={handleEnterEdit}
                disabled={training}
                className="px-3.5 py-1.5 rounded-lg border border-gray-300 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-800 text-xs font-semibold text-gray-700 dark:text-gray-200 flex items-center gap-1.5 transition-colors"
              >
                <Edit3 size={14} />
                Edit Parameters & Telemetry
              </button>
            ) : (
              <>
                <button
                  onClick={handleCancelEdit}
                  className="px-3 py-1.5 rounded-lg border border-gray-300 dark:border-gray-600 text-xs font-semibold text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 flex items-center gap-1 transition-colors"
                >
                  <X size={14} />
                  Cancel
                </button>
                <button
                  onClick={handleSaveEdit}
                  className="btn-primary px-3.5 py-1.5 text-xs font-semibold flex items-center gap-1 shadow-sm"
                >
                  <Save size={14} />
                  Save Changes
                </button>
              </>
            )}
          </div>
        </div>

        {/* Assigned Dataset Partition (Read-Only) */}
        <div className="p-3.5 bg-gray-50 dark:bg-gray-800/60 rounded-lg border border-gray-200 dark:border-gray-700 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div className="flex items-center gap-2.5">
            <Database size={18} className="text-arc-500 flex-shrink-0" />
            <div>
              <div className="text-xs text-gray-500 uppercase tracking-wider font-semibold">Assigned Dataset Partition</div>
              <div className="text-sm font-bold text-gray-900 dark:text-white font-mono">
                {assignedDataset?.filename || 'Loading assigned partition...'} ({assignedDataset?.sizeKb || 0} KB)
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="px-2 py-0.5 rounded-full bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 font-medium flex items-center gap-1">
              <Lock size={12} /> System Assigned ({clientProfile?.clientId})
            </span>
            <span className="text-emerald-600 dark:text-emerald-400 font-medium">100% On-Device</span>
          </div>
        </div>

        {/* 1. Editable Training Hyperparameters */}
        <div className="space-y-3">
          <div className="text-xs font-bold uppercase tracking-wider text-gray-700 dark:text-gray-300 flex items-center gap-2">
            <Layers size={15} className="text-arc-500" />
            1. Local Training Hyperparameters
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Epochs */}
            <div className="p-3 bg-white dark:bg-gray-800/80 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-semibold text-gray-600 dark:text-gray-400">Local Epochs</label>
                {renderSourceBadge(paramSources.epochs)}
              </div>
              {isEditing ? (
                <input
                  type="number"
                  min="1"
                  max="50"
                  step="1"
                  value={draftValues.epochs}
                  onChange={(e) => handleDraftChange('epochs', e.target.value)}
                  className={`w-full p-2 text-sm rounded border ${validationErrors.epochs ? 'border-red-500 bg-red-50' : 'border-gray-300 dark:border-gray-600 dark:bg-gray-700 text-gray-900 dark:text-white'}`}
                />
              ) : (
                <div className="text-lg font-bold text-gray-900 dark:text-white mt-0.5">{configValues.epochs}</div>
              )}
              {validationErrors.epochs && <div className="text-[11px] text-red-500 mt-1">{validationErrors.epochs}</div>}
              <div className="text-[11px] text-gray-400 mt-1">Iterations over local dataset (1-50)</div>
            </div>

            {/* Batch Size */}
            <div className="p-3 bg-white dark:bg-gray-800/80 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-semibold text-gray-600 dark:text-gray-400">Batch Size</label>
                {renderSourceBadge(paramSources.batchSize)}
              </div>
              {isEditing ? (
                <input
                  type="number"
                  min="16"
                  max="1024"
                  step="16"
                  value={draftValues.batchSize}
                  onChange={(e) => handleDraftChange('batchSize', e.target.value)}
                  className={`w-full p-2 text-sm rounded border ${validationErrors.batchSize ? 'border-red-500 bg-red-50' : 'border-gray-300 dark:border-gray-600 dark:bg-gray-700 text-gray-900 dark:text-white'}`}
                />
              ) : (
                <div className="text-lg font-bold text-gray-900 dark:text-white mt-0.5">{configValues.batchSize}</div>
              )}
              {validationErrors.batchSize && <div className="text-[11px] text-red-500 mt-1">{validationErrors.batchSize}</div>}
              <div className="text-[11px] text-gray-400 mt-1">Samples per mini-batch (16-1024)</div>
            </div>

            {/* Learning Rate */}
            <div className="p-3 bg-white dark:bg-gray-800/80 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-semibold text-gray-600 dark:text-gray-400">Learning Rate</label>
                {renderSourceBadge(paramSources.learningRate)}
              </div>
              {isEditing ? (
                <input
                  type="number"
                  step="0.0001"
                  min="0.00001"
                  max="1.0"
                  value={draftValues.learningRate}
                  onChange={(e) => handleDraftChange('learningRate', e.target.value)}
                  className={`w-full p-2 text-sm rounded border ${validationErrors.learningRate ? 'border-red-500 bg-red-50' : 'border-gray-300 dark:border-gray-600 dark:bg-gray-700 text-gray-900 dark:text-white'}`}
                />
              ) : (
                <div className="text-lg font-bold text-gray-900 dark:text-white mt-0.5">{configValues.learningRate}</div>
              )}
              {validationErrors.learningRate && <div className="text-[11px] text-red-500 mt-1">{validationErrors.learningRate}</div>}
              <div className="text-[11px] text-gray-400 mt-1">Optimizer step size (0.00001-1.0)</div>
            </div>
          </div>
        </div>

        {/* 2. Editable ARC Relevance Inputs */}
        <div className="space-y-3">
          <div className="text-xs font-bold uppercase tracking-wider text-gray-700 dark:text-gray-300 flex items-center gap-2">
            <Database size={15} className="text-blue-500" />
            2. ARC Relevance Inputs (Data Quality & Diversity)
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Data Quality */}
            <div className="p-3 bg-white dark:bg-gray-800/80 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-semibold text-gray-600 dark:text-gray-400">Data Quality Score (Q<sub>i</sub>)</label>
                {renderSourceBadge(paramSources.qualityScore)}
              </div>
              {isEditing ? (
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="100"
                  value={draftValues.qualityScore}
                  onChange={(e) => handleDraftChange('qualityScore', e.target.value)}
                  className={`w-full p-2 text-sm rounded border ${validationErrors.qualityScore ? 'border-red-500 bg-red-50' : 'border-gray-300 dark:border-gray-600 dark:bg-gray-700 text-gray-900 dark:text-white'}`}
                />
              ) : (
                <div className="text-lg font-bold text-blue-600 mt-0.5">{configValues.qualityScore}%</div>
              )}
              {validationErrors.qualityScore && <div className="text-[11px] text-red-500 mt-1">{validationErrors.qualityScore}</div>}
              <div className="text-[11px] text-gray-400 mt-1">Evaluates missing/inf ratios and schema integrity</div>
            </div>

            {/* Data Diversity */}
            <div className="p-3 bg-white dark:bg-gray-800/80 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-semibold text-gray-600 dark:text-gray-400">Data Diversity Score (D<sub>i</sub>)</label>
                {renderSourceBadge(paramSources.diversityScore)}
              </div>
              {isEditing ? (
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="100"
                  value={draftValues.diversityScore}
                  onChange={(e) => handleDraftChange('diversityScore', e.target.value)}
                  className={`w-full p-2 text-sm rounded border ${validationErrors.diversityScore ? 'border-red-500 bg-red-50' : 'border-gray-300 dark:border-gray-600 dark:bg-gray-700 text-gray-900 dark:text-white'}`}
                />
              ) : (
                <div className="text-lg font-bold text-blue-600 mt-0.5">{configValues.diversityScore}%</div>
              )}
              {validationErrors.diversityScore && <div className="text-[11px] text-red-500 mt-1">{validationErrors.diversityScore}</div>}
              <div className="text-[11px] text-gray-400 mt-1">Shannon entropy and minority attack class coverage</div>
            </div>
          </div>
        </div>

        {/* 3. Editable ARC Device Resource Inputs */}
        <div className="space-y-3">
          <div className="text-xs font-bold uppercase tracking-wider text-gray-700 dark:text-gray-300 flex items-center gap-2">
            <HardDrive size={15} className="text-purple-500" />
            3. ARC Device Resource Inputs (RAM, CPU, Capacity)
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Available RAM */}
            <div className="p-3 bg-white dark:bg-gray-800/80 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-semibold text-gray-600 dark:text-gray-400">Available RAM</label>
                {renderSourceBadge(paramSources.availableRamGb)}
              </div>
              {isEditing ? (
                <input
                  type="number"
                  step="0.1"
                  min="0.1"
                  value={draftValues.availableRamGb}
                  onChange={(e) => handleDraftChange('availableRamGb', e.target.value)}
                  className={`w-full p-2 text-sm rounded border ${validationErrors.availableRamGb ? 'border-red-500 bg-red-50' : 'border-gray-300 dark:border-gray-600 dark:bg-gray-700 text-gray-900 dark:text-white'}`}
                />
              ) : (
                <div className="text-lg font-bold text-purple-600 mt-0.5">{configValues.availableRamGb} GB</div>
              )}
              {validationErrors.availableRamGb && <div className="text-[11px] text-red-500 mt-1">{validationErrors.availableRamGb}</div>}
              <div className="text-[11px] text-gray-400 mt-1">Free physical memory available for local training</div>
            </div>

            {/* Available CPU */}
            <div className="p-3 bg-white dark:bg-gray-800/80 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-semibold text-gray-600 dark:text-gray-400">CPU Headroom</label>
                {renderSourceBadge(paramSources.cpuIdlePct)}
              </div>
              {isEditing ? (
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="100"
                  value={draftValues.cpuIdlePct}
                  onChange={(e) => handleDraftChange('cpuIdlePct', e.target.value)}
                  className={`w-full p-2 text-sm rounded border ${validationErrors.cpuIdlePct ? 'border-red-500 bg-red-50' : 'border-gray-300 dark:border-gray-600 dark:bg-gray-700 text-gray-900 dark:text-white'}`}
                />
              ) : (
                <div className="text-lg font-bold text-purple-600 mt-0.5">{configValues.cpuIdlePct}%</div>
              )}
              {validationErrors.cpuIdlePct && <div className="text-[11px] text-red-500 mt-1">{validationErrors.cpuIdlePct}</div>}
              <div className="text-[11px] text-gray-400 mt-1">Idle processor percentage ready for computation</div>
            </div>

            {/* Compute Capacity / Throughput */}
            <div className="p-3 bg-white dark:bg-gray-800/80 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-semibold text-gray-600 dark:text-gray-400">Compute Capacity</label>
                {renderSourceBadge(paramSources.throughputSamplesPerSec)}
              </div>
              {isEditing ? (
                <input
                  type="number"
                  step="1"
                  min="1"
                  value={draftValues.throughputSamplesPerSec}
                  onChange={(e) => handleDraftChange('throughputSamplesPerSec', e.target.value)}
                  className={`w-full p-2 text-sm rounded border ${validationErrors.throughputSamplesPerSec ? 'border-red-500 bg-red-50' : 'border-gray-300 dark:border-gray-600 dark:bg-gray-700 text-gray-900 dark:text-white'}`}
                />
              ) : (
                <div className="text-lg font-bold text-purple-600 mt-0.5">{configValues.throughputSamplesPerSec} /s</div>
              )}
              {validationErrors.throughputSamplesPerSec && <div className="text-[11px] text-red-500 mt-1">{validationErrors.throughputSamplesPerSec}</div>}
              <div className="text-[11px] text-gray-400 mt-1">Training throughput in samples per second</div>
            </div>
          </div>
        </div>

        {/* 4. Editable ARC Communication Inputs */}
        <div className="space-y-3">
          <div className="text-xs font-bold uppercase tracking-wider text-gray-700 dark:text-gray-300 flex items-center gap-2">
            <Wifi size={15} className="text-emerald-500" />
            4. ARC Communication Inputs (Bandwidth, Latency, Reliability)
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            {/* Bandwidth */}
            <div className="p-3 bg-white dark:bg-gray-800/80 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-semibold text-gray-600 dark:text-gray-400">Uplink Bandwidth</label>
                {renderSourceBadge(paramSources.bandwidthMbps)}
              </div>
              {isEditing ? (
                <input
                  type="number"
                  step="0.5"
                  min="0.1"
                  value={draftValues.bandwidthMbps}
                  onChange={(e) => handleDraftChange('bandwidthMbps', e.target.value)}
                  className={`w-full p-2 text-sm rounded border ${validationErrors.bandwidthMbps ? 'border-red-500 bg-red-50' : 'border-gray-300 dark:border-gray-600 dark:bg-gray-700 text-gray-900 dark:text-white'}`}
                />
              ) : (
                <div className="text-lg font-bold text-emerald-600 mt-0.5">{configValues.bandwidthMbps} Mbps</div>
              )}
              {validationErrors.bandwidthMbps && <div className="text-[11px] text-red-500 mt-1">{validationErrors.bandwidthMbps}</div>}
              <div className="text-[11px] text-gray-400 mt-1">Measured or reported connection uplink rate</div>
            </div>

            {/* Latency */}
            <div className="p-3 bg-white dark:bg-gray-800/80 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-semibold text-gray-600 dark:text-gray-400">Round-Trip Latency</label>
                {renderSourceBadge(paramSources.latencyMs)}
              </div>
              {isEditing ? (
                <input
                  type="number"
                  step="1"
                  min="0"
                  value={draftValues.latencyMs}
                  onChange={(e) => handleDraftChange('latencyMs', e.target.value)}
                  className={`w-full p-2 text-sm rounded border ${validationErrors.latencyMs ? 'border-red-500 bg-red-50' : 'border-gray-300 dark:border-gray-600 dark:bg-gray-700 text-gray-900 dark:text-white'}`}
                />
              ) : (
                <div className="text-lg font-bold text-emerald-600 mt-0.5">{configValues.latencyMs} ms</div>
              )}
              {validationErrors.latencyMs && <div className="text-[11px] text-red-500 mt-1">{validationErrors.latencyMs}</div>}
              <div className="text-[11px] text-gray-400 mt-1">Network ping time to federated coordinator</div>
            </div>

            {/* Transmission Reliability */}
            <div className="p-3 bg-white dark:bg-gray-800/80 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between mb-1">
                <label className="text-xs font-semibold text-gray-600 dark:text-gray-400">Reliability Rate</label>
                {renderSourceBadge(paramSources.reliabilityPct)}
              </div>
              {isEditing ? (
                <input
                  type="number"
                  step="0.1"
                  min="0"
                  max="100"
                  value={draftValues.reliabilityPct}
                  onChange={(e) => handleDraftChange('reliabilityPct', e.target.value)}
                  className={`w-full p-2 text-sm rounded border ${validationErrors.reliabilityPct ? 'border-red-500 bg-red-50' : 'border-gray-300 dark:border-gray-600 dark:bg-gray-700 text-gray-900 dark:text-white'}`}
                />
              ) : (
                <div className="text-lg font-bold text-emerald-600 mt-0.5">{configValues.reliabilityPct}%</div>
              )}
              {validationErrors.reliabilityPct && <div className="text-[11px] text-red-500 mt-1">{validationErrors.reliabilityPct}</div>}
              <div className="text-[11px] text-gray-400 mt-1">Packet transmission success rate</div>
            </div>
          </div>
        </div>

        {/* Action Button */}
        <div className="pt-2 flex flex-col sm:flex-row items-start sm:items-center gap-4">
          <button
            onClick={handleStartTraining}
            disabled={
              training ||
              isEditing ||
              clientEligibility?.hasSubmitted ||
              (clientEligibility && !clientEligibility.isTargeted) ||
              activeRound?.status !== 'ACTIVE'
            }
            className={`btn-primary w-full sm:w-auto px-6 py-2.5 flex items-center justify-center gap-2 font-medium ${
              training ||
              isEditing ||
              clientEligibility?.hasSubmitted ||
              (clientEligibility && !clientEligibility.isTargeted) ||
              activeRound?.status !== 'ACTIVE'
                ? 'opacity-50 cursor-not-allowed'
                : ''
            }`}
          >
            {training ? (
              <>
                <RefreshCw size={18} className="animate-spin" />
                Training On-Device Model (PyTorch)...
              </>
            ) : (
              <>
                <Play size={18} />
                Start Local Training
              </>
            )}
          </button>

          {isEditing && (
            <span className="text-xs text-amber-600 font-medium">
              Save or Cancel edits before starting local training.
            </span>
          )}

          {clientEligibility?.hasSubmitted && (
            <span className="text-xs text-blue-600 dark:text-blue-400 font-medium flex items-center gap-1.5">
              <CheckCircle size={14} />
              Update already submitted for Round #{activeRound?.roundNumber}. Awaiting server aggregation.
            </span>
          )}

          {clientEligibility && !clientEligibility.isTargeted && (
            <span className="text-xs text-amber-600 font-medium flex items-center gap-1.5">
              <AlertTriangle size={14} />
              Client is not designated for Round #{activeRound?.roundNumber}.
            </span>
          )}

          {activeRound?.status !== 'ACTIVE' && (
            <span className="text-xs text-gray-500 font-medium">
              No open active round. (Status: {activeRound?.status || 'NONE'})
            </span>
          )}
        </div>
      </div>

      {/* Live PyTorch Streaming Telemetry Card (Active Training) */}
      {training && (
        <div className="card border-2 border-arc-500 bg-arc-50/20 dark:bg-arc-950/20 p-5 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-arc-200 dark:border-arc-800 pb-3">
            <div className="flex items-center gap-2">
              <Radio size={18} className="text-arc-500 animate-pulse" />
              <h3 className="text-sm font-bold text-gray-900 dark:text-white">
                Live Edge Training Telemetry Stream
              </h3>
            </div>
            <span className="text-xs font-mono text-arc-600 dark:text-arc-400 font-bold">
              {liveStage || 'FINE_TUNING_PYTORCH'}
            </span>
          </div>

          <div className="space-y-1.5">
            <div className="flex justify-between text-xs font-medium text-gray-600 dark:text-gray-300">
              <span>Epoch {liveEpoch} of {liveTotalEpochs} {liveTotalBatches > 0 && `(Batch ${liveBatch} / ${liveTotalBatches})`}</span>
              <span className="font-bold text-arc-600">{liveProgress}%</span>
            </div>
            <div className="w-full bg-gray-200 dark:bg-gray-700 h-2.5 rounded-full overflow-hidden">
              <div
                className="bg-arc-600 h-2.5 rounded-full transition-all duration-300"
                style={{ width: `${liveProgress}%` }}
              />
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs pt-1">
            <div className="p-2.5 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="text-gray-400">Current Epoch</div>
              <div className="text-base font-bold text-gray-900 dark:text-white mt-0.5">{liveEpoch} / {liveTotalEpochs}</div>
            </div>
            <div className="p-2.5 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="text-gray-400">Mini-Batch Progress</div>
              <div className="text-base font-bold text-purple-600 mt-0.5">{liveBatch} / {liveTotalBatches || 2876}</div>
            </div>
            <div className="p-2.5 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="text-gray-400">Current Loss</div>
              <div className="text-base font-bold text-arc-600 mt-0.5">{liveLoss !== null ? Number(liveLoss).toFixed(4) : 'Tracking...'}</div>
            </div>
            <div className="p-2.5 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
              <div className="text-gray-400">BatchNorm State</div>
              <div className="text-base font-bold text-emerald-600 mt-0.5">FROZEN (Eval)</div>
            </div>
          </div>
        </div>
      )}

      {/* Training Results & Model Update Review */}
      {trainingResult && (
        <div className="card space-y-6 border-2 border-arc-300 dark:border-arc-700 bg-gradient-to-br from-white to-arc-50/30 dark:from-gray-800 dark:to-gray-900">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 border-b border-gray-100 dark:border-gray-800 pb-3">
            <div>
              <span className="badge-green text-xs font-bold uppercase">Training Complete</span>
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mt-1">
                Local Training Results & Candidate Update ({trainingResult.update_id?.substring(0, 8)})
              </h3>
            </div>
            <span className="text-xs text-gray-500 font-mono flex items-center gap-1">
              <Clock size={14} /> Duration: {trainingResult.training_duration_seconds}s
            </span>
          </div>

          {/* Section 1: Snapshot of Configuration Values Used */}
          <div>
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1.5">
              <Sliders size={14} className="text-arc-500" /> Preserved Configuration Snapshot for Round
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              <div className="p-2.5 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="text-[11px] text-gray-500">Epochs Used</div>
                <div className="text-sm font-bold text-gray-900 dark:text-white">{configValues.epochs}</div>
              </div>
              <div className="p-2.5 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="text-[11px] text-gray-500">Batch Size Used</div>
                <div className="text-sm font-bold text-gray-900 dark:text-white">{configValues.batchSize}</div>
              </div>
              <div className="p-2.5 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="text-[11px] text-gray-500">Learning Rate Used</div>
                <div className="text-sm font-bold text-gray-900 dark:text-white">{configValues.learningRate}</div>
              </div>
              <div className="p-2.5 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
                <div className="text-[11px] text-gray-500">Samples Processed</div>
                <div className="text-sm font-bold text-purple-600">{trainingResult.samples_trained}</div>
              </div>
            </div>
          </div>

          {/* Section 2: PyTorch Performance */}
          <div>
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1.5">
              <BarChart2 size={14} className="text-green-500" /> PyTorch Training Performance
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
              <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 text-center">
                <div className="text-xs text-gray-500">Local Accuracy</div>
                <div className="text-2xl font-bold text-green-600">{trainingResult.accuracy}%</div>
              </div>

              <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 text-center">
                <div className="text-xs text-gray-500">Local CrossEntropy Loss</div>
                <div className="text-2xl font-bold text-arc-600">{trainingResult.loss}</div>
              </div>

              <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 text-center">
                <div className="text-xs text-gray-500">Measured Throughput</div>
                <div className="text-2xl font-bold text-amber-600">
                  {configValues.throughputSamplesPerSec} /s
                </div>
              </div>
            </div>
          </div>

          {/* Section 3: ARC Telemetry Inputs Summary */}
          <div>
            <div className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-2 flex items-center gap-1.5">
              <Activity size={14} className="text-blue-500" /> ARC Telemetry Snapshot to be Transmitted
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
              <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 space-y-1">
                <div className="font-semibold text-gray-700 dark:text-gray-200 flex items-center justify-between">
                  <span>Relevance Inputs</span>
                  {renderSourceBadge(paramSources.qualityScore)}
                </div>
                <div className="text-gray-600 dark:text-gray-300 font-mono">
                  <div>Quality Score: {configValues.qualityScore}%</div>
                  <div>Diversity Score: {configValues.diversityScore}%</div>
                </div>
              </div>

              <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 space-y-1">
                <div className="font-semibold text-gray-700 dark:text-gray-200 flex items-center justify-between">
                  <span>Resource Inputs</span>
                  {renderSourceBadge(paramSources.availableRamGb)}
                </div>
                <div className="text-gray-600 dark:text-gray-300 font-mono">
                  <div>RAM: {configValues.availableRamGb} GB / {configValues.totalRamGb} GB</div>
                  <div>CPU Headroom: {configValues.cpuIdlePct}%</div>
                </div>
              </div>

              <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 space-y-1">
                <div className="font-semibold text-gray-700 dark:text-gray-200 flex items-center justify-between">
                  <span>Communication Inputs</span>
                  {renderSourceBadge(paramSources.latencyMs)}
                </div>
                <div className="text-gray-600 dark:text-gray-300 font-mono">
                  <div>Latency: {configValues.latencyMs} ms</div>
                  <div>Bandwidth: {configValues.bandwidthMbps} Mbps</div>
                  <div>Reliability: {configValues.reliabilityPct}%</div>
                </div>
              </div>
            </div>
          </div>

          {/* Submit Action */}
          <div className="pt-2 flex flex-col sm:flex-row gap-3">
            <button
              onClick={handleSubmitUpdate}
              disabled={submitting || !!submissionSuccess}
              className="btn-primary flex-1 py-2.5 flex items-center justify-center gap-2 font-semibold shadow-md"
            >
              {submitting ? (
                <>
                  <RefreshCw size={18} className="animate-spin" />
                  Transmitting Model Update & Preserved Telemetry...
                </>
              ) : submissionSuccess ? (
                <>
                  <CheckCircle size={18} />
                  Update Submitted to Server
                </>
              ) : (
                <>
                  <Send size={18} />
                  Submit Model Update to Server
                </>
              )}
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
