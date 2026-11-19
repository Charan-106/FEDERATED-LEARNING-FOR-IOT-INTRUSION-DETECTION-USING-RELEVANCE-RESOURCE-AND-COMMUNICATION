// Admin Model Page - Global Model Registry, 3-Reference Architecture & Version Management
// Features:
// 1. Authoritative 3-Reference Architecture (INITIAL_BASELINE, PREVIOUS_BEST, ACTIVE_GLOBAL)
// 2. Comprehensive Model Version Registry with roles, lineages, and evaluation stats
// 3. Side-by-side Model Comparison with deltas and benchmark equivalence checks
// 4. Inspect modal with evaluation history and architecture specifications
// 5. Rollback modal with required administrator justification

import { useEffect, useState } from 'react'
import {
  Brain,
  Download,
  RotateCcw,
  CheckCircle2,
  Cpu,
  Layers,
  HardDrive,
  Clock,
  ShieldAlert,
  ShieldCheck,
  RefreshCw,
  FileCheck,
  Search,
  Scale,
  Eye,
  X,
  AlertTriangle,
  ArrowRight,
  Sparkles,
  GitCommit,
  Check
} from 'lucide-react'
import { modelApi } from '../../services/api'

export default function AdminModelPage() {
  const [currentModel, setCurrentModel] = useState(null)
  const [references, setReferences] = useState(null)
  const [versions, setVersions] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [actionSuccess, setActionSuccess] = useState('')
  const [searchQuery, setSearchQuery] = useState('')

  // Modal States
  const [inspectModalOpen, setInspectModalOpen] = useState(false)
  const [inspectModel, setInspectModel] = useState(null)

  const [compareModalOpen, setCompareModalOpen] = useState(false)
  const [compareModelA, setCompareModelA] = useState('')
  const [compareModelB, setCompareModelB] = useState('')
  const [compareResult, setCompareResult] = useState(null)
  const [compareLoading, setCompareLoading] = useState(false)

  const [rollbackModalOpen, setRollbackModalOpen] = useState(false)
  const [rollbackTarget, setRollbackTarget] = useState(null)
  const [rollbackReason, setRollbackReason] = useState('')
  const [actionLoading, setActionLoading] = useState(false)

  useEffect(() => {
    loadModels()
  }, [])

  async function loadModels() {
    setLoading(true)
    setError('')
    try {
      const [curr, versData, refs] = await Promise.all([
        modelApi.getCurrent(),
        modelApi.getVersions(),
        modelApi.getReferences().catch(() => null),
      ])
      setCurrentModel(curr.model)
      setVersions(versData.models || [])
      setReferences(refs || versData.references || null)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleDownload(modelId, version) {
    const token = localStorage.getItem('arc_fl_token')
    try {
      const res = await fetch(`/api/models/${modelId}/download`, {
        headers: { Authorization: `Bearer ${token}` },
      })
      if (!res.ok) throw new Error('Download failed')
      const blob = await res.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `global-model-${version}.pt`
      document.body.appendChild(a)
      a.click()
      a.remove()
      window.URL.revokeObjectURL(url)
    } catch (err) {
      alert(`Download failed: ${err.message}`)
    }
  }

  function openInspectModal(model) {
    setInspectModel(model)
    setInspectModalOpen(true)
  }

  function openCompareModal(modelAId, modelBId) {
    const activeId = currentModel?.id
    const baselineId = references?.initialModel?.id || versions.find((v) => v.version === 'initial-v1')?.id
    const aId = modelAId || baselineId || versions[0]?.id
    const bId = modelBId || activeId || versions[1]?.id || aId

    setCompareModelA(aId)
    setCompareModelB(bId)
    setCompareModalOpen(true)
    runComparison(aId, bId)
  }

  async function runComparison(aId, bId) {
    if (!aId || !bId) return
    setCompareLoading(true)
    try {
      const result = await modelApi.compare(aId, bId)
      setCompareResult(result)
    } catch (err) {
      setCompareResult(null)
      setError(`Model comparison error: ${err.message}`)
    } finally {
      setCompareLoading(false)
    }
  }

  function openRollbackModal(model) {
    setRollbackTarget(model)
    setRollbackReason('')
    setRollbackModalOpen(true)
  }

  async function handleConfirmRollback() {
    if (!rollbackTarget) return
    if (!rollbackReason.trim()) {
      alert('Administrator justification is required to perform a model rollback.')
      return
    }

    setActionLoading(true)
    setError('')
    try {
      await modelApi.rollback(rollbackTarget.id, rollbackReason.trim())
      setActionSuccess(`Active production model successfully rolled back to ${rollbackTarget.version}!`)
      setRollbackModalOpen(false)
      setRollbackTarget(null)
      setRollbackReason('')
      await loadModels()
    } catch (err) {
      setError(`Rollback failed: ${err.message}`)
    } finally {
      setActionLoading(false)
    }
  }

  // Filtered versions based on search
  const filteredVersions = versions.filter((ver) => {
    if (!searchQuery) return true
    const q = searchQuery.toLowerCase()
    return (
      ver.version?.toLowerCase().includes(q) ||
      ver.name?.toLowerCase().includes(q) ||
      ver.role?.toLowerCase().includes(q) ||
      ver.checksumSha256?.toLowerCase().includes(q)
    )
  })

  if (loading) {
    return <div className="card text-center py-16 text-gray-500">Loading model registry and references...</div>
  }

  return (
    <div className="space-y-6 max-w-7xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-black text-gray-900 dark:text-white flex items-center gap-2">
            <Brain className="text-arc-600" size={28} />
            Global Model Lifecycle & Version Registry
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Authoritative neural network checkpoints, 3-reference tracking, benchmark equivalence verification, and zero-downtime rollback.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => openCompareModal()}
            className="btn-secondary flex items-center gap-1.5 text-xs py-2 px-3 font-semibold"
          >
            <Scale size={14} className="text-arc-600" /> Compare Models
          </button>
          <button
            onClick={loadModels}
            className="btn-secondary flex items-center gap-1.5 text-xs py-2 px-3"
          >
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg text-red-600 text-sm flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
          <button onClick={() => setError('')}><X size={16} /></button>
        </div>
      )}

      {actionSuccess && (
        <div className="p-4 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg text-green-700 text-sm flex items-center justify-between">
          <div className="flex items-center gap-2">
            <CheckCircle2 size={18} />
            <span>{actionSuccess}</span>
          </div>
          <button onClick={() => setActionSuccess('')}><X size={16} /></button>
        </div>
      )}

      {/* ============================================================ */}
      {/* 3 AUTHORITATIVE REFERENCE PANELS                             */}
      {/* ============================================================ */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Panel 1: INITIAL_BASELINE */}
        <div className="card border-2 border-blue-400 bg-gradient-to-br from-blue-50/50 via-white to-blue-50/20 dark:from-gray-800 dark:via-gray-800 dark:to-gray-900 shadow-sm relative overflow-hidden flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="badge-blue text-[11px] font-bold uppercase tracking-wider flex items-center gap-1">
                <ShieldCheck size={13} /> Panel 1 • Baseline
              </span>
              <span className="text-[10px] font-mono text-blue-600 font-semibold bg-blue-100 dark:bg-blue-900/40 px-2 py-0.5 rounded">
                PERMANENT
              </span>
            </div>

            <div>
              <h3 className="text-lg font-black text-gray-900 dark:text-white">
                {references?.initialModel?.name || 'PreAct ResMLP v3 Baseline'}
              </h3>
              <p className="text-xs text-gray-500 font-mono mt-0.5">
                Version: <span className="font-bold text-blue-600">{references?.initialModel?.version || 'initial-v1'}</span>
              </p>
            </div>

            <div className="grid grid-cols-3 gap-2 py-2 border-y border-blue-100 dark:border-gray-700 text-center">
              <div>
                <span className="text-[10px] text-gray-400 uppercase">Accuracy</span>
                <p className="text-sm font-black text-blue-700 dark:text-blue-400">
                  {references?.initialModel?.accuracy != null ? `${Number(references.initialModel.accuracy).toFixed(2)}%` : '95.20%'}
                </p>
              </div>
              <div>
                <span className="text-[10px] text-gray-400 uppercase">Macro F1</span>
                <p className="text-sm font-black text-blue-700 dark:text-blue-400">
                  {references?.initialModel?.macroF1 != null ? `${Number(references.initialModel.macroF1).toFixed(2)}%` : '80.73%'}
                </p>
              </div>
              <div>
                <span className="text-[10px] text-gray-400 uppercase">Val Loss</span>
                <p className="text-sm font-black text-gray-700 dark:text-gray-300">
                  {references?.initialModel?.loss != null ? Number(references.initialModel.loss).toFixed(4) : '0.1286'}
                </p>
              </div>
            </div>

            <p className="text-[11px] text-gray-600 dark:text-gray-300 line-clamp-2">
              Authoritative initial production checkpoint. Immutable safety anchor for regression detection.
            </p>
          </div>

          <div className="pt-3 mt-3 border-t border-blue-100 dark:border-gray-700 flex items-center justify-between text-xs">
            <span className="font-mono text-[10px] text-gray-400 truncate max-w-[150px]" title={references?.initialModel?.checksum}>
              SHA: {references?.initialModel?.checksum?.slice(0, 12)}...
            </span>
            {references?.initialModel?.id && (
              <button
                onClick={() => handleDownload(references.initialModel.id, references.initialModel.version)}
                className="text-blue-600 hover:text-blue-800 font-semibold flex items-center gap-1 text-xs"
              >
                <Download size={13} /> .pt
              </button>
            )}
          </div>
        </div>

        {/* Panel 2: PREVIOUS_BEST */}
        <div className="card border-2 border-purple-400 bg-gradient-to-br from-purple-50/50 via-white to-purple-50/20 dark:from-gray-800 dark:via-gray-800 dark:to-gray-900 shadow-sm relative overflow-hidden flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="badge-purple text-[11px] font-bold uppercase tracking-wider flex items-center gap-1">
                <Sparkles size={13} /> Panel 2 • Previous Best
              </span>
              <span className="text-[10px] font-mono text-purple-600 font-semibold bg-purple-100 dark:bg-purple-900/40 px-2 py-0.5 rounded">
                TOP BENCHMARK
              </span>
            </div>

            <div>
              <h3 className="text-lg font-black text-gray-900 dark:text-white">
                {references?.previousBestModel?.name || 'Validated Historical Performer'}
              </h3>
              <p className="text-xs text-gray-500 font-mono mt-0.5">
                Version: <span className="font-bold text-purple-600">{references?.previousBestModel?.version || 'global-v8'}</span>
              </p>
            </div>

            <div className="grid grid-cols-3 gap-2 py-2 border-y border-purple-100 dark:border-gray-700 text-center">
              <div>
                <span className="text-[10px] text-gray-400 uppercase">Accuracy</span>
                <p className="text-sm font-black text-purple-700 dark:text-purple-400">
                  {references?.previousBestModel?.accuracy != null ? `${Number(references.previousBestModel.accuracy).toFixed(2)}%` : '95.22%'}
                </p>
              </div>
              <div>
                <span className="text-[10px] text-gray-400 uppercase">Macro F1</span>
                <p className="text-sm font-black text-purple-700 dark:text-purple-400">
                  {references?.previousBestModel?.macroF1 != null ? `${Number(references.previousBestModel.macroF1).toFixed(2)}%` : '81.20%'}
                </p>
              </div>
              <div>
                <span className="text-[10px] text-gray-400 uppercase">Val Loss</span>
                <p className="text-sm font-black text-gray-700 dark:text-gray-300">
                  {references?.previousBestModel?.loss != null ? Number(references.previousBestModel.loss).toFixed(4) : '0.1238'}
                </p>
              </div>
            </div>

            <p className="text-[11px] text-gray-600 dark:text-gray-300 line-clamp-2" title={references?.previousBestModel?.selectionReason}>
              {references?.previousBestModel?.selectionReason || 'Highest multi-metric benchmark performance validated on test dataset.'}
            </p>
          </div>

          <div className="pt-3 mt-3 border-t border-purple-100 dark:border-gray-700 flex items-center justify-between text-xs">
            <span className="font-mono text-[10px] text-gray-400 truncate max-w-[150px]" title={references?.previousBestModel?.checksum}>
              SHA: {references?.previousBestModel?.checksum?.slice(0, 12)}...
            </span>
            {references?.previousBestModel?.id && (
              <button
                onClick={() => handleDownload(references.previousBestModel.id, references.previousBestModel.version)}
                className="text-purple-600 hover:text-purple-800 font-semibold flex items-center gap-1 text-xs"
              >
                <Download size={13} /> .pt
              </button>
            )}
          </div>
        </div>

        {/* Panel 3: ACTIVE_GLOBAL */}
        <div className="card border-2 border-green-500 bg-gradient-to-br from-green-50/50 via-white to-green-50/20 dark:from-gray-800 dark:via-gray-800 dark:to-gray-900 shadow-md relative overflow-hidden flex flex-col justify-between">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="badge-green text-[11px] font-bold uppercase tracking-wider flex items-center gap-1">
                <CheckCircle2 size={13} /> Panel 3 • Current Global
              </span>
              <span className="text-[10px] font-mono text-green-700 font-bold bg-green-100 dark:bg-green-900/40 px-2 py-0.5 rounded animate-pulse">
                ACTIVE FLEET
              </span>
            </div>

            <div>
              <h3 className="text-lg font-black text-gray-900 dark:text-white">
                {currentModel?.name || 'Active Production Model'}
              </h3>
              <p className="text-xs text-gray-500 font-mono mt-0.5">
                Version: <span className="font-bold text-green-700 dark:text-green-400">{currentModel?.version}</span>
              </p>
            </div>

            <div className="grid grid-cols-3 gap-2 py-2 border-y border-green-100 dark:border-gray-700 text-center">
              <div>
                <span className="text-[10px] text-gray-400 uppercase">Accuracy</span>
                <p className="text-sm font-black text-green-700 dark:text-green-400">
                  {currentModel?.baselineAccuracy != null ? `${Number(currentModel.baselineAccuracy).toFixed(2)}%` : '95.20%'}
                </p>
              </div>
              <div>
                <span className="text-[10px] text-gray-400 uppercase">Macro F1</span>
                <p className="text-sm font-black text-green-700 dark:text-green-400">
                  {currentModel?.macroF1 != null ? `${Number(currentModel.macroF1).toFixed(2)}%` : '80.73%'}
                </p>
              </div>
              <div>
                <span className="text-[10px] text-gray-400 uppercase">Val Loss</span>
                <p className="text-sm font-black text-gray-700 dark:text-gray-300">
                  {currentModel?.baselineLoss != null ? Number(currentModel.baselineLoss).toFixed(4) : '0.1286'}
                </p>
              </div>
            </div>

            <p className="text-[11px] text-gray-600 dark:text-gray-300 line-clamp-2">
              Served to edge devices upon connection. Automatically used as base checkpoint for all new training rounds.
            </p>
          </div>

          <div className="pt-3 mt-3 border-t border-green-100 dark:border-gray-700 flex items-center justify-between text-xs">
            <span className="font-mono text-[10px] text-gray-400 truncate max-w-[150px]" title={currentModel?.checksumSha256}>
              SHA: {currentModel?.checksumSha256?.slice(0, 12)}...
            </span>
            {currentModel?.id && (
              <button
                onClick={() => handleDownload(currentModel.id, currentModel.version)}
                className="btn-primary py-1 px-2.5 text-xs flex items-center gap-1 shadow-sm font-bold"
              >
                <Download size={13} /> Checkpoint (.pt)
              </button>
            )}
          </div>
        </div>
      </div>

      {/* ============================================================ */}
      {/* VERSION REGISTRY TABLE                                       */}
      {/* ============================================================ */}
      <div className="card space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-gray-100 dark:border-gray-800 pb-3">
          <div>
            <h2 className="text-base font-bold text-gray-900 dark:text-white flex items-center gap-2">
              <Clock size={18} className="text-arc-600" />
              Model Version Registry & Lineage
            </h2>
            <p className="text-xs text-gray-500">
              Immutable checkpoint repository with multi-metric evaluation records and provenance tracking.
            </p>
          </div>

          {/* Search Bar */}
          <div className="relative min-w-[240px]">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Search version, role, hash..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full text-xs pl-8 pr-3 py-1.5 rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-gray-50 dark:bg-gray-800/80 uppercase text-[10px] text-gray-500 border-b border-gray-200 dark:border-gray-700 tracking-wider">
              <tr>
                <th className="py-2.5 px-3">Version & Role</th>
                <th className="py-2.5 px-3">Name / Architecture</th>
                <th className="py-2.5 px-3 text-right">Accuracy</th>
                <th className="py-2.5 px-3 text-right">Macro F1</th>
                <th className="py-2.5 px-3 text-right">Loss</th>
                <th className="py-2.5 px-3">Source Round</th>
                <th className="py-2.5 px-3">Checksum (SHA-256)</th>
                <th className="py-2.5 px-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-800 font-medium">
              {filteredVersions.map((ver) => {
                const isCurrent = ver.isActive
                const role = ver.role || (isCurrent ? 'ACTIVE_GLOBAL' : (ver.version === 'initial-v1' ? 'INITIAL_BASELINE' : 'PUBLISHED'))

                return (
                  <tr key={ver.id} className={`hover:bg-gray-50 dark:hover:bg-gray-800/40 transition-colors ${isCurrent ? 'bg-green-50/20 dark:bg-green-950/10' : ''}`}>
                    {/* Version & Role */}
                    <td className="py-3 px-3">
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono font-bold text-arc-600 text-sm">
                          {ver.version}
                        </span>
                        {isCurrent && (
                          <span className="badge-green text-[10px] font-bold px-1.5 py-0.2">ACTIVE</span>
                        )}
                        {role === 'INITIAL_BASELINE' && (
                          <span className="badge-blue text-[10px] font-bold px-1.5 py-0.2">BASELINE</span>
                        )}
                        {role === 'PREVIOUS_BEST' && !isCurrent && (
                          <span className="badge-purple text-[10px] font-bold px-1.5 py-0.2">PREV BEST</span>
                        )}
                      </div>
                      <span className="text-[10px] text-gray-400">
                        {new Date(ver.createdAt).toLocaleDateString()}
                      </span>
                    </td>

                    {/* Name / Arch */}
                    <td className="py-3 px-3">
                      <div className="font-semibold text-gray-900 dark:text-white text-xs">{ver.name}</div>
                      <span className="font-mono text-[10px] text-gray-400">{ver.architecture} • 45 feat • 34 cls</span>
                    </td>

                    {/* Accuracy */}
                    <td className="py-3 px-3 text-right font-mono font-bold text-green-600">
                      {ver.baselineAccuracy != null ? `${Number(ver.baselineAccuracy).toFixed(2)}%` : '—'}
                    </td>

                    {/* Macro F1 */}
                    <td className="py-3 px-3 text-right font-mono font-bold text-arc-600">
                      {ver.macroF1 != null ? `${Number(ver.macroF1).toFixed(2)}%` : '—'}
                    </td>

                    {/* Loss */}
                    <td className="py-3 px-3 text-right font-mono text-gray-600 dark:text-gray-400">
                      {ver.baselineLoss != null ? Number(ver.baselineLoss).toFixed(4) : '—'}
                    </td>

                    {/* Source Round */}
                    <td className="py-3 px-3">
                      {ver.sourceRoundId ? (
                        <span className="font-mono text-[11px] text-gray-600 dark:text-gray-300">
                          {ver.resultingRound ? `Round #${ver.resultingRound.roundNumber}` : ver.sourceRoundId.slice(0, 8)}
                        </span>
                      ) : (
                        <span className="text-[10px] text-gray-400 italic">Genesis Model</span>
                      )}
                    </td>

                    {/* Checksum */}
                    <td className="py-3 px-3">
                      <code className="text-[11px] text-gray-500 font-mono bg-gray-100 dark:bg-gray-800 px-1 py-0.5 rounded" title={ver.checksumSha256}>
                        {ver.checksumSha256 ? `${ver.checksumSha256.slice(0, 10)}...` : '—'}
                      </code>
                    </td>

                    {/* Actions */}
                    <td className="py-3 px-3 text-right space-x-1 whitespace-nowrap">
                      {/* Inspect */}
                      <button
                        onClick={() => openInspectModal(ver)}
                        className="p-1 text-gray-600 hover:text-arc-600 dark:text-gray-400 rounded transition-colors"
                        title="Inspect Metadata & Lineage"
                      >
                        <Eye size={15} />
                      </button>

                      {/* Compare */}
                      <button
                        onClick={() => openCompareModal(ver.id, currentModel?.id)}
                        className="p-1 text-gray-600 hover:text-arc-600 dark:text-gray-400 rounded transition-colors"
                        title="Compare with Active Global Model"
                      >
                        <Scale size={15} />
                      </button>

                      {/* Download */}
                      <button
                        onClick={() => handleDownload(ver.id, ver.version)}
                        className="p-1 text-gray-600 hover:text-arc-600 dark:text-gray-400 rounded transition-colors"
                        title="Download Checkpoint"
                      >
                        <Download size={15} />
                      </button>

                      {/* Rollback */}
                      {!isCurrent && (
                        <button
                          onClick={() => openRollbackModal(ver)}
                          className="px-2 py-0.5 text-[11px] font-semibold bg-amber-50 text-amber-700 hover:bg-amber-100 dark:bg-amber-900/30 dark:text-amber-300 rounded transition-colors inline-flex items-center gap-1"
                          title="Activate this version (Rollback)"
                        >
                          <RotateCcw size={11} /> Rollback
                        </button>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* ============================================================ */}
      {/* MODEL COMPARISON MODAL                                       */}
      {/* ============================================================ */}
      {compareModalOpen && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-gray-900 rounded-2xl max-w-3xl w-full p-6 shadow-2xl border border-gray-200 dark:border-gray-800 space-y-5 animate-in fade-in duration-150">
            <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-800 pb-3">
              <div className="flex items-center gap-2">
                <Scale className="text-arc-600" size={22} />
                <h3 className="text-lg font-black text-gray-900 dark:text-white">
                  Side-by-Side Model Comparison
                </h3>
              </div>
              <button onClick={() => setCompareModalOpen(false)} className="text-gray-400 hover:text-gray-600">
                <X size={18} />
              </button>
            </div>

            {/* Selector Row */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-semibold text-gray-600 dark:text-gray-400 block mb-1">
                  Model A (Reference / Baseline):
                </label>
                <select
                  value={compareModelA}
                  onChange={(e) => { setCompareModelA(e.target.value); runComparison(e.target.value, compareModelB); }}
                  className="w-full text-xs p-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 font-bold"
                >
                  {versions.map((v) => (
                    <option key={v.id} value={v.id}>
                      {v.version} - {v.name} {v.isActive ? '(ACTIVE)' : ''}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-xs font-semibold text-gray-600 dark:text-gray-400 block mb-1">
                  Model B (Candidate / Comparison):
                </label>
                <select
                  value={compareModelB}
                  onChange={(e) => { setCompareModelB(e.target.value); runComparison(compareModelA, e.target.value); }}
                  className="w-full text-xs p-2 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 font-bold"
                >
                  {versions.map((v) => (
                    <option key={v.id} value={v.id}>
                      {v.version} - {v.name} {v.isActive ? '(ACTIVE)' : ''}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Equivalence Warning Banner */}
            {compareResult && !compareResult.isEquivalentBenchmark && (
              <div className="p-3 bg-amber-50 dark:bg-amber-950/30 border border-amber-300 dark:border-amber-800 rounded-lg text-amber-700 text-xs flex items-start gap-2">
                <AlertTriangle size={16} className="mt-0.5 flex-shrink-0" />
                <span>{compareResult.equivalenceWarning}</span>
              </div>
            )}

            {/* Comparison Metrics Table */}
            {compareLoading ? (
              <div className="py-12 text-center text-sm text-gray-500">Evaluating and calculating metric deltas...</div>
            ) : compareResult ? (
              <div className="space-y-4">
                <table className="w-full text-xs">
                  <thead className="bg-gray-50 dark:bg-gray-800 uppercase text-[10px] text-gray-500 border-b border-gray-200 dark:border-gray-700">
                    <tr>
                      <th className="py-2 px-3">Metric</th>
                      <th className="py-2 px-3 text-center">{compareResult.modelA.version} (Model A)</th>
                      <th className="py-2 px-3 text-center">{compareResult.modelB.version} (Model B)</th>
                      <th className="py-2 px-3 text-right">Delta (B - A)</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100 dark:divide-gray-800 font-medium">
                    {/* Accuracy */}
                    <tr>
                      <td className="py-2.5 px-3 font-semibold text-gray-700 dark:text-gray-300">Accuracy</td>
                      <td className="py-2.5 px-3 text-center font-mono">{Number(compareResult.modelA.accuracy).toFixed(2)}%</td>
                      <td className="py-2.5 px-3 text-center font-mono">{Number(compareResult.modelB.accuracy).toFixed(2)}%</td>
                      <td className={`py-2.5 px-3 text-right font-mono font-bold ${
                        compareResult.deltas.accuracyDelta >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {compareResult.deltas.accuracyDelta >= 0 ? `+${compareResult.deltas.accuracyDelta.toFixed(2)}%` : `${compareResult.deltas.accuracyDelta.toFixed(2)}%`}
                      </td>
                    </tr>

                    {/* Macro F1 */}
                    <tr>
                      <td className="py-2.5 px-3 font-semibold text-gray-700 dark:text-gray-300">Macro F1 Score</td>
                      <td className="py-2.5 px-3 text-center font-mono">{Number(compareResult.modelA.macroF1).toFixed(2)}%</td>
                      <td className="py-2.5 px-3 text-center font-mono">{Number(compareResult.modelB.macroF1).toFixed(2)}%</td>
                      <td className={`py-2.5 px-3 text-right font-mono font-bold ${
                        compareResult.deltas.macroF1Delta >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {compareResult.deltas.macroF1Delta >= 0 ? `+${compareResult.deltas.macroF1Delta.toFixed(2)}%` : `${compareResult.deltas.macroF1Delta.toFixed(2)}%`}
                      </td>
                    </tr>

                    {/* Validation Loss */}
                    <tr>
                      <td className="py-2.5 px-3 font-semibold text-gray-700 dark:text-gray-300">Validation Loss</td>
                      <td className="py-2.5 px-3 text-center font-mono">{Number(compareResult.modelA.loss).toFixed(4)}</td>
                      <td className="py-2.5 px-3 text-center font-mono">{Number(compareResult.modelB.loss).toFixed(4)}</td>
                      <td className={`py-2.5 px-3 text-right font-mono font-bold ${
                        compareResult.deltas.lossDelta <= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {compareResult.deltas.lossDelta <= 0 ? `${compareResult.deltas.lossDelta.toFixed(4)}` : `+${compareResult.deltas.lossDelta.toFixed(4)}`}
                      </td>
                    </tr>

                    {/* Benchmark Dataset */}
                    <tr>
                      <td className="py-2.5 px-3 font-semibold text-gray-700 dark:text-gray-300">Evaluation Benchmark</td>
                      <td className="py-2.5 px-3 text-center font-mono text-[11px] text-gray-500">{compareResult.modelA.benchmark}</td>
                      <td className="py-2.5 px-3 text-center font-mono text-[11px] text-gray-500">{compareResult.modelB.benchmark}</td>
                      <td className="py-2.5 px-3 text-right">
                        {compareResult.isEquivalentBenchmark ? (
                          <span className="badge-green text-[10px] font-bold">EQUIVALENT</span>
                        ) : (
                          <span className="badge-yellow text-[10px] font-bold">VARIED</span>
                        )}
                      </td>
                    </tr>

                    {/* SHA Checksums */}
                    <tr>
                      <td className="py-2.5 px-3 font-semibold text-gray-700 dark:text-gray-300">SHA-256 Digest</td>
                      <td className="py-2.5 px-3 text-center font-mono text-[10px] text-gray-400">{compareResult.modelA.checksum?.slice(0, 12)}...</td>
                      <td className="py-2.5 px-3 text-center font-mono text-[10px] text-gray-400">{compareResult.modelB.checksum?.slice(0, 12)}...</td>
                      <td className="py-2.5 px-3 text-right text-[10px] text-gray-400">
                        {compareResult.modelA.checksum === compareResult.modelB.checksum ? 'IDENTICAL' : 'DISTINCT'}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            ) : null}

            <div className="flex justify-end pt-3 border-t border-gray-100 dark:border-gray-800">
              <button
                onClick={() => setCompareModalOpen(false)}
                className="btn-secondary text-xs px-4 py-2"
              >
                Close Comparison
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ============================================================ */}
      {/* INSPECT METADATA MODAL                                       */}
      {/* ============================================================ */}
      {inspectModalOpen && inspectModel && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-gray-900 rounded-2xl max-w-2xl w-full p-6 shadow-2xl border border-gray-200 dark:border-gray-800 space-y-4">
            <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-800 pb-3">
              <div className="flex items-center gap-2">
                <Brain className="text-arc-600" size={22} />
                <h3 className="text-lg font-black text-gray-900 dark:text-white">
                  Inspect Model: {inspectModel.version}
                </h3>
              </div>
              <button onClick={() => setInspectModalOpen(false)} className="text-gray-400 hover:text-gray-600">
                <X size={18} />
              </button>
            </div>

            <div className="space-y-3 text-xs">
              <div className="grid grid-cols-2 gap-3 p-3 bg-gray-50 dark:bg-gray-800/50 rounded-xl">
                <div>
                  <span className="text-gray-400">Model Name:</span>
                  <p className="font-bold text-gray-900 dark:text-white mt-0.5">{inspectModel.name}</p>
                </div>
                <div>
                  <span className="text-gray-400">Architecture:</span>
                  <p className="font-mono font-bold text-arc-600 mt-0.5">{inspectModel.architecture}</p>
                </div>
                <div>
                  <span className="text-gray-400">Total Parameters:</span>
                  <p className="font-mono font-semibold text-gray-800 dark:text-gray-200 mt-0.5">{inspectModel.totalParameters?.toLocaleString() || '4,981,628'}</p>
                </div>
                <div>
                  <span className="text-gray-400">Features / Classes:</span>
                  <p className="font-mono font-semibold text-gray-800 dark:text-gray-200 mt-0.5">{inspectModel.inputFeatures} features • {inspectModel.numClasses} classes</p>
                </div>
              </div>

              <div>
                <span className="text-gray-400 font-semibold block mb-1">Full SHA-256 Checksum:</span>
                <code className="block p-2 bg-gray-100 dark:bg-gray-800 rounded font-mono text-[11px] text-gray-800 dark:text-gray-200 break-all select-all">
                  {inspectModel.checksumSha256 || 'None'}
                </code>
              </div>

              <div>
                <span className="text-gray-400 font-semibold block mb-1">Disk Storage Location:</span>
                <code className="block p-2 bg-gray-100 dark:bg-gray-800 rounded font-mono text-[11px] text-gray-800 dark:text-gray-200">
                  {inspectModel.storagePath}
                </code>
              </div>

              {inspectModel.evaluations && inspectModel.evaluations.length > 0 && (
                <div>
                  <span className="text-gray-400 font-semibold block mb-1">Evaluation History ({inspectModel.evaluations.length} records):</span>
                  <div className="space-y-1.5 max-h-36 overflow-y-auto">
                    {inspectModel.evaluations.map((ev, i) => (
                      <div key={i} className="p-2 border border-gray-200 dark:border-gray-700 rounded flex items-center justify-between text-[11px]">
                        <span className="text-gray-600 dark:text-gray-300 font-mono">{ev.datasetName} ({ev.sampleCount} samples)</span>
                        <div className="flex items-center gap-3">
                          <span className="font-bold text-green-600">Acc: {ev.accuracy}%</span>
                          <span className="font-bold text-arc-600">F1: {ev.macroF1}%</span>
                          <span className="text-gray-400">Loss: {Number(ev.loss).toFixed(4)}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="flex justify-end pt-3 border-t border-gray-100 dark:border-gray-800">
              <button
                onClick={() => setInspectModalOpen(false)}
                className="btn-secondary text-xs px-4 py-2"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ============================================================ */}
      {/* ROLLBACK CONFIRMATION MODAL                                  */}
      {/* ============================================================ */}
      {rollbackModalOpen && rollbackTarget && (
        <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-gray-900 rounded-2xl max-w-lg w-full p-6 shadow-2xl border border-amber-300 dark:border-amber-800 space-y-4">
            <div className="flex items-center gap-3 border-b border-gray-100 dark:border-gray-800 pb-3">
              <div className="w-10 h-10 rounded-full bg-amber-100 dark:bg-amber-900/50 flex items-center justify-center text-amber-600 flex-shrink-0">
                <RotateCcw size={20} />
              </div>
              <div>
                <h3 className="text-lg font-black text-gray-900 dark:text-white">
                  Confirm Model Rollback
                </h3>
                <p className="text-xs text-gray-500">Zero-downtime pointer reversion to prior global model</p>
              </div>
            </div>

            <div className="p-3 bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-800 rounded-xl space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-gray-600 dark:text-gray-400">Current Production Model:</span>
                <span className="font-mono font-bold text-red-600">{currentModel?.version}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-600 dark:text-gray-400">Target Rollback Model:</span>
                <span className="font-mono font-bold text-green-600">{rollbackTarget.version}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-gray-600 dark:text-gray-400">Target Accuracy:</span>
                <span className="font-mono font-bold text-green-600">{rollbackTarget.baselineAccuracy}%</span>
              </div>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-bold text-gray-700 dark:text-gray-300 flex items-center gap-1">
                Administrator Written Justification <span className="text-red-500">*</span>
              </label>
              <textarea
                rows={3}
                value={rollbackReason}
                onChange={(e) => setRollbackReason(e.target.value)}
                placeholder="Specify reason for rollback (e.g. edge drift observed in latest version, reverting to baseline initial-v1)..."
                className="w-full text-xs p-2.5 rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-gray-900 dark:text-white"
              />
            </div>

            <div className="flex items-center justify-end gap-2 pt-3 border-t border-gray-100 dark:border-gray-800">
              <button
                type="button"
                onClick={() => setRollbackModalOpen(false)}
                className="btn-secondary text-xs px-4 py-2"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={actionLoading || !rollbackReason.trim()}
                onClick={handleConfirmRollback}
                className="btn-primary text-xs px-4 py-2 bg-amber-600 hover:bg-amber-700 text-white font-bold flex items-center gap-1.5"
              >
                <RotateCcw size={14} />
                {actionLoading ? 'Executing Rollback...' : `Rollback to ${rollbackTarget.version}`}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
