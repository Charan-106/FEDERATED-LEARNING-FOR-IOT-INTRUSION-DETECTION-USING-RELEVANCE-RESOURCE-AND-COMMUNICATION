// Admin ARC Evaluation & Configuration Page
// Section 7: Allows administrators to review fleet ARC scores, dynamically configure formula weights,
// review formula version history, and trigger live recalculation across edge submissions.

import { useEffect, useState } from 'react'
import {
  BarChart3,
  Sliders,
  Award,
  Save,
  RotateCcw,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Database,
  Cpu,
  Wifi,
  History,
  TrendingUp,
  Clock,
  Layers,
  ChevronDown,
  ChevronUp
} from 'lucide-react'
import { arcApi } from '../../services/api'

export default function AdminArcPage() {
  const [config, setConfig] = useState(null)
  const [configs, setConfigs] = useState([])
  const [scores, setScores] = useState([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [recalculating, setRecalculating] = useState(false)
  const [saveSuccess, setSaveSuccess] = useState('')
  const [recalcResult, setRecalcResult] = useState(null)
  const [error, setError] = useState('')
  const [autoRecalculate, setAutoRecalculate] = useState(true)
  const [expandedSubId, setExpandedSubId] = useState(null)
  const [showHistory, setShowHistory] = useState(false)

  // Form state
  const [weights, setWeights] = useState({
    lambdaRelevance: 0.40,
    lambdaResource: 0.30,
    lambdaCommunication: 0.30,
    weightQuality: 0.60,
    weightDiversity: 0.40,
    weightMemory: 0.40,
    weightProcessing: 0.40,
    weightCapacity: 0.20,
    weightBandwidth: 0.50,
    weightLatency: 0.30,
    weightReliability: 0.20,
  })

  useEffect(() => {
    loadData()
  }, [])

  async function loadData() {
    setLoading(true)
    setError('')
    try {
      const [cfg, cfgs, sc] = await Promise.all([
        arcApi.getConfig(),
        arcApi.getConfigHistory(),
        arcApi.getScores(),
      ])

      const c = cfg.config
      setConfig(c)
      setConfigs(cfgs.configs || [])
      setScores(sc.scores || [])

      if (c) {
        setWeights({
          lambdaRelevance: c.lambdaRelevance ?? 0.40,
          lambdaResource: c.lambdaResource ?? 0.30,
          lambdaCommunication: c.lambdaCommunication ?? 0.30,
          weightQuality: c.weightQuality ?? 0.60,
          weightDiversity: c.weightDiversity ?? 0.40,
          weightMemory: c.weightMemory ?? 0.40,
          weightProcessing: c.weightProcessing ?? 0.40,
          weightCapacity: c.weightCapacity ?? 0.20,
          weightBandwidth: c.weightBandwidth ?? 0.50,
          weightLatency: c.weightLatency ?? 0.30,
          weightReliability: c.weightReliability ?? 0.20,
        })
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const topLevelSum = (
    parseFloat(weights.lambdaRelevance || 0) +
    parseFloat(weights.lambdaResource || 0) +
    parseFloat(weights.lambdaCommunication || 0)
  ).toFixed(2)

  const isNormalized = Math.abs(parseFloat(topLevelSum) - 1.0) < 0.01

  async function handleSaveConfig(e) {
    e.preventDefault()
    if (!isNormalized) {
      alert(`Top-level weights must sum to 1.00 (currently ${topLevelSum}). Please adjust weights.`)
      return
    }

    setSaving(true)
    setError('')
    setSaveSuccess('')
    setRecalcResult(null)

    try {
      const res = await arcApi.updateConfig({
        lambdaRelevance: parseFloat(weights.lambdaRelevance),
        lambdaResource: parseFloat(weights.lambdaResource),
        lambdaCommunication: parseFloat(weights.lambdaCommunication),
        weightQuality: parseFloat(weights.weightQuality),
        weightDiversity: parseFloat(weights.weightDiversity),
        weightMemory: parseFloat(weights.weightMemory),
        weightProcessing: parseFloat(weights.weightProcessing),
        weightCapacity: parseFloat(weights.weightCapacity),
        weightBandwidth: parseFloat(weights.weightBandwidth),
        weightLatency: parseFloat(weights.weightLatency),
        weightReliability: parseFloat(weights.weightReliability),
        recalculatePending: autoRecalculate,
      })

      if (res.recalculation && res.recalculation.recalculatedCount > 0) {
        setRecalcResult(res.recalculation)
        setSaveSuccess(`ARC weights updated to ${res.config?.version || 'new version'}! Recalculated ${res.recalculation.recalculatedCount} pending submission(s).`)
      } else {
        setSaveSuccess(`ARC scoring parameters updated to ${res.config?.version || 'new version'} successfully!`)
      }

      await loadData()
    } catch (err) {
      setError(`Failed to save config: ${err.message}`)
    } finally {
      setSaving(false)
    }
  }

  async function handleManualRecalculate() {
    setRecalculating(true)
    setError('')
    setSaveSuccess('')
    setRecalcResult(null)

    try {
      const res = await arcApi.recalculateScores()
      setRecalcResult(res)
      setSaveSuccess(`Recalculated ARC scores for ${res.recalculatedCount} submission(s) using formula version ${res.configVersion}.`)
      await loadData()
    } catch (err) {
      setError(`Recalculation failed: ${err.message}`)
    } finally {
      setRecalculating(false)
    }
  }

  function handleResetDefaults() {
    setWeights({
      lambdaRelevance: 0.40,
      lambdaResource: 0.30,
      lambdaCommunication: 0.30,
      weightQuality: 0.60,
      weightDiversity: 0.40,
      weightMemory: 0.40,
      weightProcessing: 0.40,
      weightCapacity: 0.20,
      weightBandwidth: 0.50,
      weightLatency: 0.30,
      weightReliability: 0.20,
    })
  }

  if (loading) return <div className="card text-center py-16 text-gray-500">Loading ARC parameters...</div>

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <BarChart3 className="text-arc-600" size={26} />
            ARC Framework Configuration & Fleet Scores
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Configure dynamic multi-criteria weights for Relevance, Resource, and Communication.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowHistory(!showHistory)}
            className="btn-secondary flex items-center gap-1.5 text-xs py-2 px-3"
          >
            <History size={14} /> {showHistory ? 'Hide Versions' : 'Version History'}
          </button>
          <button
            onClick={loadData}
            className="btn-secondary flex items-center gap-1.5 text-xs py-2 px-3"
          >
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      {error && <div className="card bg-red-50 text-red-600 p-4 text-sm">{error}</div>}
      {saveSuccess && (
        <div className="card bg-green-50 text-green-700 p-4 text-sm flex items-center gap-2">
          <CheckCircle2 size={18} />
          {saveSuccess}
        </div>
      )}

      {/* Recalculation Results Banner if available */}
      {recalcResult && recalcResult.results && recalcResult.results.length > 0 && (
        <div className="card bg-blue-50/70 border border-blue-200 dark:bg-gray-800 p-4 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-bold text-blue-700 dark:text-blue-300 uppercase tracking-wider flex items-center gap-1.5">
              <Award size={14} /> Live Recalculation Impact ({recalcResult.recalculatedCount} Submissions)
            </span>
            <span className="text-xs text-gray-500 font-mono">Config v{recalcResult.configVersion}</span>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
            {recalcResult.results.map((r) => (
              <div key={r.submissionId} className="bg-white dark:bg-gray-900 p-2.5 rounded border border-blue-100 dark:border-gray-700 text-xs">
                <div className="font-semibold text-gray-800 dark:text-gray-200">{r.clientId}</div>
                <div className="flex items-center justify-between mt-1 text-gray-500">
                  <span>Score: {r.previousScore?.toFixed(2)} → <strong className="text-arc-600">{r.newScore?.toFixed(2)}</strong></span>
                  <span className={`font-mono font-bold ${r.delta >= 0 ? 'text-green-600' : 'text-red-500'}`}>
                    {r.delta >= 0 ? `+${r.delta}` : r.delta}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Version History Drawer */}
      {showHistory && (
        <div className="card space-y-3 bg-gray-50 dark:bg-gray-800/80 border border-gray-200 dark:border-gray-700">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-gray-900 dark:text-white flex items-center gap-1.5">
              <History size={16} className="text-arc-600" />
              Formula Version History
            </h3>
            <span className="text-xs text-gray-400 font-mono">{configs.length} revisions stored</span>
          </div>
          <div className="space-y-2 max-h-60 overflow-y-auto">
            {configs.map((c) => (
              <div
                key={c.id}
                className={`p-3 rounded-lg border text-xs flex items-center justify-between ${
                  c.isActive
                    ? 'bg-arc-50/70 border-arc-300 dark:bg-gray-900'
                    : 'bg-white border-gray-200 dark:bg-gray-800 dark:border-gray-700'
                }`}
              >
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-bold font-mono text-gray-900 dark:text-white">v{c.version}</span>
                    {c.isActive && (
                      <span className="badge-green text-[10px] py-0.5 px-1.5">ACTIVE</span>
                    )}
                    <span className="text-gray-400 text-[11px] flex items-center gap-1">
                      <Clock size={11} /> {new Date(c.createdAt).toLocaleString()}
                    </span>
                  </div>
                  <div className="text-gray-600 dark:text-gray-300 font-mono mt-1 text-[11px]">
                    λ_R: {c.lambdaRelevance} | λ_S: {c.lambdaResource} | λ_C: {c.lambdaCommunication}
                  </div>
                </div>
                <div className="text-right text-[11px] text-gray-400">
                  Q={c.weightQuality} D={c.weightDiversity} | M={c.weightMemory} P={c.weightProcessing} A={c.weightCapacity}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Dynamic Weight Configuration Form */}
      <form onSubmit={handleSaveConfig} className="card space-y-6">
        <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-800 pb-3">
          <div>
            <h2 className="text-base font-bold text-gray-900 dark:text-white flex items-center gap-2">
              <Sliders size={18} className="text-arc-600" />
              ARC Formula Parameter Controls
            </h2>
            <p className="text-xs text-gray-500 mt-0.5">
              Active Version: <span className="font-mono font-bold text-arc-600">v{config?.version || '1.0'}</span>. Persisted in database table <code>ARCConfig</code>.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <span className={`text-xs px-2.5 py-1 rounded font-mono font-bold ${
              isNormalized ? 'bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300' : 'bg-red-100 text-red-700'
            }`}>
              Sum: {topLevelSum} / 1.00
            </span>
          </div>
        </div>

        {/* Top-Level Lambdas */}
        <div>
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
            Top-Level Component Weights (Must sum to 1.00)
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-3 bg-blue-50/50 dark:bg-gray-800 rounded-lg border border-blue-200 dark:border-gray-700">
              <label className="text-xs font-semibold text-blue-700 dark:text-blue-300 flex items-center gap-1.5">
                <Database size={14} /> Relevance (λ_R)
              </label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={weights.lambdaRelevance}
                onChange={(e) => setWeights({ ...weights, lambdaRelevance: e.target.value })}
                className="w-full mt-2 p-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm font-mono font-bold"
              />
              <span className="text-[11px] text-gray-500 mt-1 block">Data quality & diversity</span>
            </div>

            <div className="p-3 bg-green-50/50 dark:bg-gray-800 rounded-lg border border-green-200 dark:border-gray-700">
              <label className="text-xs font-semibold text-green-700 dark:text-green-300 flex items-center gap-1.5">
                <Cpu size={14} /> Resource (λ_S)
              </label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={weights.lambdaResource}
                onChange={(e) => setWeights({ ...weights, lambdaResource: e.target.value })}
                className="w-full mt-2 p-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm font-mono font-bold"
              />
              <span className="text-[11px] text-gray-500 mt-1 block">RAM, CPU & throughput headroom</span>
            </div>

            <div className="p-3 bg-purple-50/50 dark:bg-gray-800 rounded-lg border border-purple-200 dark:border-gray-700">
              <label className="text-xs font-semibold text-purple-700 dark:text-purple-300 flex items-center gap-1.5">
                <Wifi size={14} /> Communication (λ_C)
              </label>
              <input
                type="number"
                step="0.05"
                min="0"
                max="1"
                value={weights.lambdaCommunication}
                onChange={(e) => setWeights({ ...weights, lambdaCommunication: e.target.value })}
                className="w-full mt-2 p-2 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-sm font-mono font-bold"
              />
              <span className="text-[11px] text-gray-500 mt-1 block">Bandwidth, latency & reliability</span>
            </div>
          </div>
        </div>

        {/* Sub-Weights Configuration */}
        <div className="pt-2 border-t border-gray-100 dark:border-gray-800 space-y-4">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white">
            Component Sub-Weights Breakdown
          </h3>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Relevance sub-weights */}
            <div className="space-y-3 p-3 bg-gray-50 dark:bg-gray-800/60 rounded-lg">
              <span className="text-xs font-bold text-gray-700 dark:text-gray-300">Relevance: R = w_Q·Q + w_D·D</span>
              <div>
                <label className="text-xs text-gray-500">Quality Weight (w_Q)</label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={weights.weightQuality}
                  onChange={(e) => setWeights({ ...weights, weightQuality: e.target.value })}
                  className="w-full mt-1 p-1.5 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-xs font-mono"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500">Diversity Weight (w_D)</label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={weights.weightDiversity}
                  onChange={(e) => setWeights({ ...weights, weightDiversity: e.target.value })}
                  className="w-full mt-1 p-1.5 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-xs font-mono"
                />
              </div>
            </div>

            {/* Resource sub-weights */}
            <div className="space-y-3 p-3 bg-gray-50 dark:bg-gray-800/60 rounded-lg">
              <span className="text-xs font-bold text-gray-700 dark:text-gray-300">Resource: S = w_M·M + w_P·P + w_A·A</span>
              <div>
                <label className="text-xs text-gray-500">Memory Weight (w_M)</label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={weights.weightMemory}
                  onChange={(e) => setWeights({ ...weights, weightMemory: e.target.value })}
                  className="w-full mt-1 p-1.5 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-xs font-mono"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500">Processing Weight (w_P)</label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={weights.weightProcessing}
                  onChange={(e) => setWeights({ ...weights, weightProcessing: e.target.value })}
                  className="w-full mt-1 p-1.5 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-xs font-mono"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500">Capacity Weight (w_A)</label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={weights.weightCapacity}
                  onChange={(e) => setWeights({ ...weights, weightCapacity: e.target.value })}
                  className="w-full mt-1 p-1.5 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-xs font-mono"
                />
              </div>
            </div>

            {/* Communication sub-weights */}
            <div className="space-y-3 p-3 bg-gray-50 dark:bg-gray-800/60 rounded-lg">
              <span className="text-xs font-bold text-gray-700 dark:text-gray-300">Communication: C = w_B·B + w_L·L + w_T·T</span>
              <div>
                <label className="text-xs text-gray-500">Bandwidth Weight (w_B)</label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={weights.weightBandwidth}
                  onChange={(e) => setWeights({ ...weights, weightBandwidth: e.target.value })}
                  className="w-full mt-1 p-1.5 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-xs font-mono"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500">Latency Weight (w_L)</label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={weights.weightLatency}
                  onChange={(e) => setWeights({ ...weights, weightLatency: e.target.value })}
                  className="w-full mt-1 p-1.5 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-xs font-mono"
                />
              </div>
              <div>
                <label className="text-xs text-gray-500">Reliability Weight (w_T)</label>
                <input
                  type="number"
                  step="0.05"
                  min="0"
                  max="1"
                  value={weights.weightReliability}
                  onChange={(e) => setWeights({ ...weights, weightReliability: e.target.value })}
                  className="w-full mt-1 p-1.5 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-xs font-mono"
                />
              </div>
            </div>
          </div>
        </div>

        {/* Options and Action Buttons */}
        <div className="pt-2 border-t border-gray-100 dark:border-gray-800 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <label className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-300 cursor-pointer">
            <input
              type="checkbox"
              checked={autoRecalculate}
              onChange={(e) => setAutoRecalculate(e.target.checked)}
              className="rounded border-gray-300 text-arc-600 focus:ring-arc-500"
            />
            <span>Automatically recalculate pending submissions upon saving</span>
          </label>

          <div className="flex items-center gap-2">
            <button
              type="submit"
              disabled={saving || !isNormalized}
              className="btn-primary flex items-center gap-2 px-5 py-2 font-semibold"
            >
              <Save size={16} />
              {saving ? 'Saving & Applying...' : 'Save ARC Weights'}
            </button>

            <button
              type="button"
              onClick={handleManualRecalculate}
              disabled={recalculating}
              className="btn-secondary flex items-center gap-1.5 text-xs py-2 px-3"
            >
              <RefreshCw size={14} className={recalculating ? 'animate-spin' : ''} />
              {recalculating ? 'Recalculating...' : 'Recalculate Submissions'}
            </button>

            <button
              type="button"
              onClick={handleResetDefaults}
              className="btn-secondary flex items-center gap-1.5 text-xs py-2 px-3"
            >
              <RotateCcw size={14} /> Defaults
            </button>
          </div>
        </div>
      </form>

      {/* Fleet Submissions & ARC Scores Section (Strict Vertical Cards) */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-bold text-gray-900 dark:text-white flex items-center gap-2">
              <Award className="text-arc-600" size={20} />
              Fleet ARC Submissions & Evaluations
            </h2>
            <p className="text-xs text-gray-500">
              Evaluated submissions sorted by newest. Showing individual Relevance, Resource, and Communication ratings.
            </p>
          </div>
          <span className="text-xs text-gray-400 font-mono">{scores.length} evaluation(s) recorded</span>
        </div>

        {scores.length === 0 ? (
          <div className="card text-center py-12 text-gray-400">
            No submissions have been evaluated yet. Run a training round to generate scores.
          </div>
        ) : (
          <div className="space-y-3">
            {scores.map((sc) => {
              const isExpanded = expandedSubId === sc.id
              return (
                <div
                  key={sc.id}
                  className="card p-4 transition-all hover:border-arc-300 dark:hover:border-arc-700 space-y-3"
                >
                  {/* Top Bar */}
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-arc-50 dark:bg-arc-900/30 flex items-center justify-center font-bold text-arc-600 text-sm">
                        {sc.client?.clientId?.slice(-2) || '01'}
                      </div>
                      <div>
                        <div className="flex items-center gap-2">
                          <h4 className="font-bold text-sm text-gray-900 dark:text-white">
                            {sc.client?.name || sc.clientId}
                          </h4>
                          <span className="text-xs text-gray-400 font-mono">({sc.clientId})</span>
                          <span className="badge-purple text-[10px] py-0.5">Round #{sc.round?.roundNumber || '1'}</span>
                          <span className="text-[10px] font-mono text-gray-400">v{sc.formulaVersion}</span>
                        </div>
                        <div className="text-xs text-gray-400 mt-0.5 flex items-center gap-3">
                          <span>Submission: #{sc.submissionId?.slice(0, 8)}</span>
                          <span>Samples: {sc.submission?.samplesTrained || 600}</span>
                          <span>Accuracy: {sc.submission?.localAccuracy ? `${sc.submission.localAccuracy}%` : 'N/A'}</span>
                          <span>{new Date(sc.calculatedAt).toLocaleTimeString()}</span>
                        </div>
                      </div>
                    </div>

                    {/* Overall Score Badge */}
                    <div className="flex items-center gap-4">
                      <div className="text-right">
                        <div className="text-[11px] uppercase font-bold text-gray-400">ARC Score</div>
                        <div className="text-2xl font-black text-arc-600">
                          {sc.overallArcScore?.toFixed(2)}
                        </div>
                      </div>

                      <button
                        onClick={() => setExpandedSubId(isExpanded ? null : sc.id)}
                        className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-white rounded"
                        title="Toggle details"
                      >
                        {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                      </button>
                    </div>
                  </div>

                  {/* Component Breakdown Bars */}
                  <div className="grid grid-cols-3 gap-3 pt-2 border-t border-gray-100 dark:border-gray-800 text-xs">
                    <div className="space-y-1">
                      <div className="flex justify-between text-gray-600 dark:text-gray-300">
                        <span className="flex items-center gap-1 text-blue-600 font-semibold">
                          <Database size={12} /> Relevance (R)
                        </span>
                        <span className="font-mono font-bold">{sc.relevanceScore?.toFixed(1)}</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-blue-500 h-full rounded-full"
                          style={{ width: `${Math.min(sc.relevanceScore || 0, 100)}%` }}
                        />
                      </div>
                    </div>

                    <div className="space-y-1">
                      <div className="flex justify-between text-gray-600 dark:text-gray-300">
                        <span className="flex items-center gap-1 text-green-600 font-semibold">
                          <Cpu size={12} /> Resource (S)
                        </span>
                        <span className="font-mono font-bold">{sc.resourceScore?.toFixed(1)}</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-green-500 h-full rounded-full"
                          style={{ width: `${Math.min(sc.resourceScore || 0, 100)}%` }}
                        />
                      </div>
                    </div>

                    <div className="space-y-1">
                      <div className="flex justify-between text-gray-600 dark:text-gray-300">
                        <span className="flex items-center gap-1 text-purple-600 font-semibold">
                          <Wifi size={12} /> Communication (C)
                        </span>
                        <span className="font-mono font-bold">{sc.communicationScore?.toFixed(1)}</span>
                      </div>
                      <div className="w-full bg-gray-200 dark:bg-gray-700 h-1.5 rounded-full overflow-hidden">
                        <div
                          className="bg-purple-500 h-full rounded-full"
                          style={{ width: `${Math.min(sc.communicationScore || 0, 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>

                  {/* Expanded Sub-Component Details Drawer */}
                  {isExpanded && (
                    <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-800 bg-gray-50/50 dark:bg-gray-900/50 p-3 rounded-lg text-xs space-y-2">
                      <div className="font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wider text-[11px]">
                        Detailed Metrics & Sub-Scores
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px]">
                        <div className="bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700">
                          <span className="text-gray-400 block text-[10px]">Quality (Q)</span>
                          <span className="font-bold text-blue-600">{sc.qualityScore ?? 'N/A'}</span>
                        </div>
                        <div className="bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700">
                          <span className="text-gray-400 block text-[10px]">Diversity (D)</span>
                          <span className="font-bold text-blue-600">{sc.diversityScore ?? 'N/A'}</span>
                        </div>
                        <div className="bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700">
                          <span className="text-gray-400 block text-[10px]">Memory (M)</span>
                          <span className="font-bold text-green-600">{sc.memoryScore ?? 'N/A'}</span>
                        </div>
                        <div className="bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700">
                          <span className="text-gray-400 block text-[10px]">Processing (P)</span>
                          <span className="font-bold text-green-600">{sc.processingScore ?? 'N/A'}</span>
                        </div>
                        <div className="bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700">
                          <span className="text-gray-400 block text-[10px]">Capacity (A)</span>
                          <span className="font-bold text-green-600">{sc.capacityScore ?? 'N/A'}</span>
                        </div>
                        <div className="bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700">
                          <span className="text-gray-400 block text-[10px]">Bandwidth (B)</span>
                          <span className="font-bold text-purple-600">{sc.bandwidthScore ?? 'N/A'}</span>
                        </div>
                        <div className="bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700">
                          <span className="text-gray-400 block text-[10px]">Latency (L)</span>
                          <span className="font-bold text-purple-600">{sc.latencyScore ?? 'N/A'}</span>
                        </div>
                        <div className="bg-white dark:bg-gray-800 p-2 rounded border border-gray-200 dark:border-gray-700">
                          <span className="text-gray-400 block text-[10px]">Reliability (T)</span>
                          <span className="font-bold text-purple-600">{sc.reliabilityScore ?? 'N/A'}</span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
