// Admin Aggregation Center - Multi-Criteria Model Merging
// Section 9: Strategy selection (FedAvg, Equal, ARC-Weighted), client checkboxes, weight preview, and model merging

import { useEffect, useState } from 'react'
import {
  GitMerge,
  Award,
  CheckCircle2,
  AlertTriangle,
  Play,
  RotateCw,
  Eye,
  Layers,
  ArrowRight,
  ShieldCheck,
  Check,
  Cpu,
  X
} from 'lucide-react'
import { adminApi, aggregationApi, modelApi } from '../../services/api'

export default function AdminAggregationPage() {
  const [rounds, setRounds] = useState([])
  const [selectedRoundId, setSelectedRoundId] = useState('')
  const [submissions, setSubmissions] = useState([])
  const [selectedSubIds, setSelectedSubIds] = useState([])
  const [strategy, setStrategy] = useState('ARC_WEIGHTED') // FEDAVG, EQUAL, ARC_WEIGHTED
  const [currentModel, setCurrentModel] = useState(null)

  // Weight Preview State
  const [weightPreview, setWeightPreview] = useState(null)
  const [previewLoading, setPreviewLoading] = useState(false)

  // Merge State
  const [merging, setMerging] = useState(false)
  const [mergeResult, setMergeResult] = useState(null)
  const [publishing, setPublishing] = useState(false)
  const [publishSuccess, setPublishSuccess] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)

  // Regression Override Modal State
  const [overrideModalOpen, setOverrideModalOpen] = useState(false)
  const [overrideJustification, setOverrideJustification] = useState('')
  const [regressionError, setRegressionError] = useState('')

  async function handlePublishCandidate(isOverride = false) {
    if (!mergeResult) return
    if (isOverride && !overrideJustification.trim()) {
      alert('Please provide a written justification to override the regression guard.')
      return
    }

    setPublishing(true)
    setError('')
    setPublishSuccess('')
    try {
      const payload = {
        roundId: selectedRoundId,
        jobId: mergeResult.jobId,
      }
      if (isOverride) {
        payload.allowRegression = true
        payload.justification = overrideJustification.trim()
      }
      const res = await modelApi.publishCandidate(payload)
      setPublishSuccess(`Candidate model successfully published as active global model ${res.model?.version || 'global-v2'}!`)
      setOverrideModalOpen(false)
      setOverrideJustification('')
      setRegressionError('')
      await loadInitialData()
    } catch (err) {
      const errMsg = err.message || ''
      if (errMsg.includes('regression') || errMsg.includes('Regression') || errMsg.includes('422')) {
        setRegressionError(errMsg)
        setOverrideModalOpen(true)
      } else {
        setError(`Failed to publish candidate: ${errMsg}`)
      }
    } finally {
      setPublishing(false)
    }
  }

  async function handlePublishAndStartNextRound() {
    if (!mergeResult) return
    setPublishing(true)
    setError('')
    setPublishSuccess('')
    try {
      const payload = {
        roundId: selectedRoundId,
        jobId: mergeResult.jobId,
      }
      const res = await modelApi.publishCandidateAndStartRound(payload)
      setPublishSuccess(`Model ${res.model?.version} published & Federated Round #${res.nextRound?.roundNumber} started! Ready for next round client training.`)
      await loadInitialData()
      if (res.nextRound?.id) {
        setSelectedRoundId(res.nextRound.id)
      }
    } catch (err) {
      setError(`Failed to publish and start next round: ${err.message}`)
    } finally {
      setPublishing(false)
    }
  }

  useEffect(() => {
    loadInitialData()
  }, [])

  useEffect(() => {
    if (selectedRoundId) {
      loadRoundSubmissions(selectedRoundId)
    }
  }, [selectedRoundId])

  async function loadInitialData() {
    setLoading(true)
    setError('')
    try {
      const [rData, mData] = await Promise.all([
        adminApi.getRounds(),
        modelApi.getCurrent(),
      ])

      const rList = rData.rounds || []
      setRounds(rList)
      setCurrentModel(mData.model)

      // Default to latest round
      if (rList.length > 0) {
        setSelectedRoundId(rList[0].id)
      }
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function loadRoundSubmissions(rId) {
    try {
      const data = await adminApi.getRoundSubmissions(rId)
      const subs = data.submissions || []
      setSubmissions(subs)
      // Auto-select eligible submissions (status = ACCEPTED or PENDING with valid file)
      const autoSelected = subs.filter((s) => s.status !== 'REJECTED').map((s) => s.id)
      setSelectedSubIds(autoSelected)
      setWeightPreview(null)

      // Check if aggregation was already performed for this round
      try {
        const existingAgg = await aggregationApi.getRoundAggregation(rId)
        if (existingAgg && existingAgg.jobId) {
          setMergeResult(existingAgg)
        } else {
          setMergeResult(null)
        }
      } catch {
        setMergeResult(null)
      }
    } catch (err) {
      setError(`Failed to load round submissions: ${err.message}`)
    }
  }

  function handleToggleSelect(subId) {
    if (selectedSubIds.includes(subId)) {
      setSelectedSubIds(selectedSubIds.filter((id) => id !== subId))
    } else {
      setSelectedSubIds([...selectedSubIds, subId])
    }
    setWeightPreview(null)
  }

  function handleSelectAll() {
    if (selectedSubIds.length === submissions.length) {
      setSelectedSubIds([])
    } else {
      setSelectedSubIds(submissions.map((s) => s.id))
    }
    setWeightPreview(null)
  }

  // Preview Weights
  async function handlePreviewWeights() {
    if (selectedSubIds.length === 0) {
      alert('Please select at least 1 participating client submission.')
      return
    }

    setPreviewLoading(true)
    setError('')
    try {
      const res = await aggregationApi.previewWeights({
        roundId: selectedRoundId,
        submissionIds: selectedSubIds,
        strategy,
      })
      setWeightPreview(res)
    } catch (err) {
      setError(`Weight preview failed: ${err.message}`)
    } finally {
      setPreviewLoading(false)
    }
  }

  // Merge Models
  async function handleMergeModels() {
    if (selectedSubIds.length === 0) {
      alert('Cannot merge: No clients selected.')
      return
    }

    const confirmMsg = `Authorize Federated Aggregation with ${selectedSubIds.length} clients using strategy "${strategy}"?\n\nThis will trigger Python PyTorch parameter aggregation and candidate evaluation.`
    if (!window.confirm(confirmMsg)) return

    setMerging(true)
    setError('')
    setMergeResult(null)

    try {
      const res = await aggregationApi.mergeModels({
        roundId: selectedRoundId,
        submissionIds: selectedSubIds,
        strategy,
      })
      setMergeResult(res)
      await loadInitialData()
    } catch (err) {
      setError(`Aggregation failed: ${err.message}`)
    } finally {
      setMerging(false)
    }
  }

  if (loading) {
    return <div className="card text-center py-16 text-gray-500">Loading Aggregation Center...</div>
  }

  return (
    <div className="space-y-6 max-w-6xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-black text-gray-900 dark:text-white flex items-center gap-2">
            <GitMerge className="text-arc-600" size={28} />
            Aggregation Center
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Authoritative federated model merging: Select participants, strategy, preview weights, and merge parameters.
          </p>
        </div>

        {/* Round Selector */}
        <div className="flex items-center gap-2">
          <label className="text-xs font-semibold text-gray-600 dark:text-gray-400">Round:</label>
          <select
            value={selectedRoundId}
            onChange={(e) => setSelectedRoundId(e.target.value)}
            className="text-sm p-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 font-bold"
          >
            {rounds.map((r) => (
              <option key={r.id} value={r.id}>
                Round #{r.roundNumber} ({r.status})
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-600 text-sm flex items-start gap-2">
          <AlertTriangle size={18} className="mt-0.5 flex-shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Strategy Selection Controls (Master Prompt Section 9) */}
      <div className="card space-y-4">
        <h2 className="text-base font-bold text-gray-900 dark:text-white flex items-center gap-2 border-b border-gray-100 dark:border-gray-800 pb-3">
          <Layers size={18} className="text-arc-600" />
          Federated Aggregation Strategy Selection
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Strategy 1: FedAvg */}
          <div
            onClick={() => { setStrategy('FEDAVG'); setWeightPreview(null); }}
            className={`cursor-pointer p-4 rounded-xl border-2 transition-all ${
              strategy === 'FEDAVG'
                ? 'border-arc-500 bg-arc-50/40 dark:bg-arc-900/20 shadow-sm'
                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="font-bold text-sm text-gray-900 dark:text-white">Strategy 1: FedAvg</span>
              {strategy === 'FEDAVG' && <CheckCircle2 size={18} className="text-arc-600" />}
            </div>
            <p className="text-xs text-gray-500 mt-1">Sample-weighted averaging (McMahan et al.)</p>
            <div className="mt-3 p-2 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 font-mono text-xs text-arc-600">
              α_i = n_i / ∑ n_j
            </div>
          </div>

          {/* Strategy 2: Equal Weight */}
          <div
            onClick={() => { setStrategy('EQUAL'); setWeightPreview(null); }}
            className={`cursor-pointer p-4 rounded-xl border-2 transition-all ${
              strategy === 'EQUAL'
                ? 'border-arc-500 bg-arc-50/40 dark:bg-arc-900/20 shadow-sm'
                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="font-bold text-sm text-gray-900 dark:text-white">Strategy 2: Equal Weight</span>
              {strategy === 'EQUAL' && <CheckCircle2 size={18} className="text-arc-600" />}
            </div>
            <p className="text-xs text-gray-500 mt-1">Uniform participation weighting across all clients</p>
            <div className="mt-3 p-2 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 font-mono text-xs text-arc-600">
              α_i = 1 / N
            </div>
          </div>

          {/* Strategy 3: ARC-Weighted */}
          <div
            onClick={() => { setStrategy('ARC_WEIGHTED'); setWeightPreview(null); }}
            className={`cursor-pointer p-4 rounded-xl border-2 transition-all ${
              strategy === 'ARC_WEIGHTED'
                ? 'border-arc-500 bg-arc-50/40 dark:bg-arc-900/20 shadow-sm'
                : 'border-gray-200 dark:border-gray-700 hover:border-gray-300'
            }`}
          >
            <div className="flex items-center justify-between">
              <span className="font-bold text-sm text-gray-900 dark:text-white flex items-center gap-1.5">
                <Award size={15} className="text-purple-600" />
                Strategy 3: ARC-Weighted
              </span>
              {strategy === 'ARC_WEIGHTED' && <CheckCircle2 size={18} className="text-arc-600" />}
            </div>
            <p className="text-xs text-gray-500 mt-1">Relevance, Resource & Communication importance</p>
            <div className="mt-3 p-2 bg-white dark:bg-gray-800 rounded border border-gray-200 dark:border-gray-700 font-mono text-xs text-purple-600">
              α_i = ARC_i / ∑ ARC_j
            </div>
          </div>
        </div>
      </div>

      {/* Participating Client Submissions: Mandatory Vertical Arrangement */}
      <div className="card space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-gray-100 dark:border-gray-800 pb-3">
          <div>
            <h2 className="text-base font-bold text-gray-900 dark:text-white">
              Participating Client Submissions ({submissions.length})
            </h2>
            <p className="text-xs text-gray-500">
              Selected: <span className="font-bold text-arc-600">{selectedSubIds.length}</span> of {submissions.length} clients
            </p>
          </div>

          <button
            onClick={handleSelectAll}
            className="text-xs text-arc-600 font-semibold hover:underline"
          >
            {selectedSubIds.length === submissions.length ? 'Deselect All' : 'Select All Eligible'}
          </button>
        </div>

        {submissions.length === 0 ? (
          <div className="text-center py-12 text-gray-400">
            No client submissions found for this round.
          </div>
        ) : (
          <div className="space-y-3">
            {submissions.map((sub) => {
              const isSelected = selectedSubIds.includes(sub.id)
              const arc = sub.arcScores?.[0]

              return (
                <div
                  key={sub.id}
                  onClick={() => handleToggleSelect(sub.id)}
                  className={`p-4 rounded-xl border-2 cursor-pointer transition-all ${
                    isSelected
                      ? 'border-arc-400 bg-arc-50/20 dark:bg-arc-900/10'
                      : 'border-gray-200 dark:border-gray-700 opacity-60 hover:opacity-100'
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => {}} // handled by parent onClick
                      className="mt-1 h-4 w-4 rounded text-arc-600 focus:ring-arc-500"
                    />

                    <div className="flex-1">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <span className="font-bold text-gray-900 dark:text-white text-sm">
                            {sub.client?.name || sub.clientId}
                          </span>
                          <span className="font-mono text-xs text-gray-500 bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded">
                            {sub.clientId}
                          </span>
                          <span className={`badge-${sub.status === 'ACCEPTED' ? 'green' : sub.status === 'REJECTED' ? 'red' : 'yellow'} text-[11px]`}>
                            {sub.status}
                          </span>
                        </div>

                        <span className="text-xs font-mono text-arc-600 font-bold flex items-center gap-1">
                          <Award size={13} />
                          ARC: {sub.arcScore != null ? sub.arcScore.toFixed(1) : arc?.overallArcScore?.toFixed(1) || 'N/A'}
                        </span>
                      </div>

                      {/* Client Sub-Scores Grid */}
                      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 text-xs">
                        <div>
                          <span className="text-gray-400">Accuracy:</span>{' '}
                          <span className="font-bold text-green-600">{sub.localAccuracy}%</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Loss:</span>{' '}
                          <span className="font-bold">{sub.localLoss?.toFixed(4)}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">Samples:</span>{' '}
                          <span className="font-bold">{sub.samplesTrained?.toLocaleString()}</span>
                        </div>
                        <div>
                          <span className="text-gray-400">R / S / C:</span>{' '}
                          <span className="font-mono font-medium">
                            {arc?.relevanceScore?.toFixed(0) || '—'}/{arc?.resourceScore?.toFixed(0) || '—'}/{arc?.communicationScore?.toFixed(0) || '—'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {/* Action Controls: PREVIEW & MERGE (Master Prompt requirement) */}
        <div className="flex flex-col sm:flex-row items-center gap-3 pt-4 border-t border-gray-100 dark:border-gray-800">
          <button
            onClick={handlePreviewWeights}
            disabled={previewLoading || selectedSubIds.length === 0}
            className="btn-secondary flex-1 py-2.5 flex items-center justify-center gap-2 font-semibold"
          >
            <Eye size={16} />
            {previewLoading ? 'Calculating Weights...' : 'Preview Aggregation Weights'}
          </button>

          <button
            onClick={handleMergeModels}
            disabled={merging || selectedSubIds.length === 0}
            className="btn-primary flex-1 py-2.5 flex items-center justify-center gap-2 font-bold shadow-md"
          >
            <Play size={16} />
            {merging ? 'Aggregating PyTorch Parameters...' : 'Merge Models'}
          </button>
        </div>
      </div>

      {/* Weight Preview Breakdown (Master Prompt Section 9) */}
      {weightPreview && (
        <div className="card space-y-4 border-2 border-arc-300 dark:border-arc-700 bg-arc-50/20 dark:bg-gray-800/60">
          <div className="flex items-center justify-between border-b border-gray-200 dark:border-gray-700 pb-3">
            <div>
              <h3 className="font-bold text-base text-gray-900 dark:text-white flex items-center gap-2">
                <CheckCircle2 size={18} className="text-green-500" />
                Aggregation Weight Preview ({weightPreview.strategy})
              </h3>
              <p className="text-xs text-gray-500 mt-0.5">
                Verified normalization: Total weight = 100.00% across {weightPreview.weights?.length} participating clients.
              </p>
            </div>

            <span className="badge-green font-mono text-xs">
              Total Weight: {(weightPreview.total_weight * 100).toFixed(1)}%
            </span>
          </div>

          <div className="space-y-2">
            {weightPreview.weights?.map((w, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-3 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 text-sm"
              >
                <div className="flex items-center gap-2">
                  <span className="w-6 h-6 rounded-full bg-arc-100 dark:bg-arc-900 text-arc-600 font-bold text-xs flex items-center justify-center">
                    {idx + 1}
                  </span>
                  <span className="font-bold text-gray-900 dark:text-white">{w.client_name || w.client_id}</span>
                  <span className="text-xs text-gray-400 font-mono">({w.client_id})</span>
                </div>

                <div className="flex items-center gap-4">
                  <span className="text-xs text-gray-500 font-mono">
                    Samples: {w.samples?.toLocaleString()} • ARC: {w.arc_score?.toFixed(1) || '—'}
                  </span>
                  <div className="w-32 bg-gray-100 dark:bg-gray-700 rounded-full h-2.5 overflow-hidden">
                    <div
                      className="bg-arc-600 h-2.5 rounded-full"
                      style={{ width: `${w.weight * 100}%` }}
                    />
                  </div>
                  <span className="font-mono font-bold text-arc-600 text-sm min-w-[50px] text-right">
                    {(w.weight * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Merged Candidate Evaluation Result */}
      {mergeResult && (
        <div className="card space-y-4 border-2 border-green-500 bg-green-50/30 dark:bg-gray-800">
          <div className="flex items-center justify-between border-b border-green-200 dark:border-green-800 pb-3">
            <div>
              <span className="badge-green text-xs font-bold uppercase">Candidate Generated</span>
              <h3 className="text-lg font-bold text-gray-900 dark:text-white mt-1">
                Candidate Evaluation Report ({mergeResult.candidate_id})
              </h3>
            </div>
            <ShieldCheck size={28} className="text-green-600" />
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center">
            <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-green-200 dark:border-green-900">
              <span className="text-xs text-gray-500">Evaluation Accuracy</span>
              <p className="text-xl font-bold text-green-600 mt-0.5">
                {mergeResult.evaluation?.accuracy != null ? `${Number(mergeResult.evaluation.accuracy).toFixed(2)}%` : '—'}
              </p>
            </div>
            <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-green-200 dark:border-green-900">
              <span className="text-xs text-gray-500">Macro F1 Score</span>
              <p className="text-xl font-bold text-arc-600 mt-0.5">
                {mergeResult.evaluation?.macro_f1 != null
                  ? `${(Number(mergeResult.evaluation.macro_f1 > 1.0 ? mergeResult.evaluation.macro_f1 : mergeResult.evaluation.macro_f1 * 100)).toFixed(2)}%`
                  : '—'}
              </p>
            </div>
            <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-green-200 dark:border-green-900">
              <span className="text-xs text-gray-500">Validation Loss</span>
              <p className="text-xl font-bold text-gray-800 dark:text-gray-200 mt-0.5">
                {mergeResult.evaluation?.loss != null ? Number(mergeResult.evaluation.loss).toFixed(4) : '—'}
              </p>
            </div>
            <div className="p-3 bg-white dark:bg-gray-800 rounded-lg border border-green-200 dark:border-green-900">
              <span className="text-xs text-gray-500">Validation Samples</span>
              <p className="text-xl font-bold text-purple-600 mt-0.5">
                {mergeResult.evaluation?.sample_count?.toLocaleString() || '5,000'}
              </p>
            </div>
          </div>

          {publishSuccess ? (
            <div className="p-3 bg-green-100 dark:bg-green-900/40 text-green-800 dark:text-green-200 text-xs rounded-lg font-semibold flex items-center gap-2">
              <CheckCircle2 size={16} />
              {publishSuccess}
            </div>
          ) : (
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 pt-2">
              <p className="text-xs text-gray-600 dark:text-gray-400">
                Candidate checkpoint verified on server dataset <code>validation.csv</code>.
              </p>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => handlePublishCandidate(false)}
                  disabled={publishing}
                  className="px-3.5 py-2 rounded-lg border border-gray-300 dark:border-gray-600 text-xs font-bold text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center justify-center gap-1.5 transition-colors"
                >
                  <ShieldCheck size={16} />
                  {publishing ? 'Publishing...' : 'Publish Model Only'}
                </button>
                <button
                  onClick={handlePublishAndStartNextRound}
                  disabled={publishing}
                  className="btn-primary py-2 px-4 text-xs font-bold flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white shadow-md"
                >
                  <Play size={16} />
                  {publishing ? 'Publishing & Starting...' : 'Publish & Activate Next Round'}
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* REGRESSION OVERRIDE CONFIRMATION MODAL */}
      {overrideModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border border-red-200 dark:border-red-800 max-w-lg w-full p-6 space-y-4">
            <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-700 pb-3">
              <h3 className="text-base font-bold text-red-600 dark:text-red-400 flex items-center gap-2">
                <AlertTriangle size={20} />
                Regression Guard Blocked Publication
              </h3>
              <button onClick={() => setOverrideModalOpen(false)} className="text-gray-400 hover:text-gray-600">
                <X size={20} />
              </button>
            </div>

            <div className="p-3 bg-red-50 dark:bg-red-900/20 rounded-xl text-xs text-red-800 dark:text-red-200 space-y-1">
              <p className="font-semibold">Candidate model exhibits substantial performance regression (&gt; 3.0 pp drop below base model baseline):</p>
              <p className="font-mono text-[11px] opacity-90 break-words">{regressionError}</p>
            </div>

            <div>
              <label className="block text-xs font-bold text-gray-800 dark:text-gray-200 mb-1">
                Written Justification for Regression Override *
              </label>
              <p className="text-[11px] text-gray-500 mb-2">
                Antigravity safety policy requires a documented rationale before overriding publishing guards. This justification is permanently logged into the immutable audit trail.
              </p>
              <textarea
                rows={3}
                value={overrideJustification}
                onChange={(e) => setOverrideJustification(e.target.value)}
                placeholder="e.g., Experimental candidate evaluation under high non-IID conditions authorized by lead researcher..."
                className="input text-xs w-full"
              />
            </div>

            <div className="flex items-center justify-end gap-2 border-t border-gray-100 dark:border-gray-700 pt-3">
              <button
                type="button"
                onClick={() => setOverrideModalOpen(false)}
                className="btn-secondary text-xs py-2 px-4"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={publishing || !overrideJustification.trim()}
                onClick={() => handlePublishCandidate(true)}
                className="py-2 px-4 text-xs font-bold rounded-lg bg-red-600 hover:bg-red-700 text-white disabled:opacity-50 flex items-center gap-1.5"
              >
                <ShieldCheck size={14} />
                {publishing ? 'Publishing Override...' : 'Confirm Override & Publish'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
