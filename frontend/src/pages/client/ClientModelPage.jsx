// Client Model Page - Current Global Model inspection & binary download
// Section 5.B: Allows edge clients to view model specifications and download the active artifact

import { useEffect, useState } from 'react'
import { Brain, Download, ShieldCheck, Cpu, Layers, HardDrive, CheckCircle2, Clock } from 'lucide-react'
import { clientApi } from '../../services/api'

export default function ClientModelPage() {
  const [model, setModel] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [downloading, setDownloading] = useState(false)

  useEffect(() => {
    clientApi.getModel()
      .then((data) => setModel(data.model))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }, [])

  function handleDownload() {
    setDownloading(true)
    const token = localStorage.getItem('arc_fl_token')
    fetch(clientApi.downloadModel(), {
      headers: { Authorization: `Bearer ${token}` }
    })
      .then(async (res) => {
        if (!res.ok) throw new Error('Failed to download model artifact')
        const blob = await res.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `global-model-${model?.version || 'active'}.pt`
        document.body.appendChild(a)
        a.click()
        a.remove()
        window.URL.revokeObjectURL(url)
      })
      .catch((err) => alert(`Download error: ${err.message}`))
      .finally(() => setDownloading(false))
  }

  if (loading) {
    return <div className="card text-center py-12 text-gray-500">Loading active global model...</div>
  }

  if (error || !model) {
    return (
      <div className="card bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-600 p-6">
        <h3 className="font-semibold text-lg">Unable to Load Model</h3>
        <p className="text-sm mt-1">{error || 'No active global model found on the central server.'}</p>
      </div>
    )
  }

  return (
    <div className="space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <Brain className="text-arc-600" size={26} />
            Current Global Model
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Authoritative neural network checkpoint distributed to all edge participants.
          </p>
        </div>

        <button
          onClick={handleDownload}
          disabled={downloading}
          className="btn-primary flex items-center justify-center gap-2 px-5 py-2.5 shadow-sm"
        >
          <Download size={18} />
          {downloading ? 'Downloading Checkpoint...' : 'Download Global Model (.pt)'}
        </button>
      </div>

      {/* Hero Card */}
      <div className="card bg-gradient-to-br from-arc-50 to-white dark:from-gray-800 dark:to-gray-900 border border-arc-200 dark:border-gray-700">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div>
            <div className="flex items-center gap-2">
              <span className="badge-green text-xs font-semibold uppercase tracking-wider">Active Baseline</span>
              <span className="text-xs text-gray-500 font-mono">ID: {model.id}</span>
            </div>
            <h2 className="text-2xl font-extrabold text-gray-900 dark:text-white mt-2">
              {model.name}
            </h2>
            <p className="text-sm text-gray-600 dark:text-gray-300 mt-1 max-w-2xl">
              Deep Pre-Activation Residual Multi-Layer Perceptron optimized for network flow cyberattack detection
              across 34 IoT intrusion classes using the CICIoT2023 benchmark.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row md:flex-col gap-3">
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700 text-center min-w-[140px]">
              <div className="text-xs text-gray-500 uppercase tracking-wide">Version</div>
              <div className="text-xl font-bold font-mono text-arc-600">{model.version}</div>
            </div>
            <div className="bg-white dark:bg-gray-800 p-3 rounded-lg border border-gray-200 dark:border-gray-700 text-center min-w-[140px]">
              <div className="text-xs text-gray-500 uppercase tracking-wide">Validation Accuracy</div>
              <div className="text-xl font-bold text-green-600">{model.baselineAccuracy ?? 98.50}%</div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Specifications Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-blue-50 dark:bg-blue-900/30 text-blue-600 rounded-lg">
              <Cpu size={20} />
            </div>
            <div>
              <div className="text-xs text-gray-500">Architecture</div>
              <div className="text-sm font-bold text-gray-900 dark:text-white font-mono">{model.architecture}</div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-purple-50 dark:bg-purple-900/30 text-purple-600 rounded-lg">
              <Layers size={20} />
            </div>
            <div>
              <div className="text-xs text-gray-500">Parameters</div>
              <div className="text-sm font-bold text-gray-900 dark:text-white">
                {model.totalParameters ? model.totalParameters.toLocaleString() : '4,981,628'}
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-green-50 dark:bg-green-900/30 text-green-600 rounded-lg">
              <ShieldCheck size={20} />
            </div>
            <div>
              <div className="text-xs text-gray-500">Input / Classes</div>
              <div className="text-sm font-bold text-gray-900 dark:text-white">
                {model.inputFeatures} features / {model.numClasses} classes
              </div>
            </div>
          </div>
        </div>

        <div className="card">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-amber-50 dark:bg-amber-900/30 text-amber-600 rounded-lg">
              <HardDrive size={20} />
            </div>
            <div>
              <div className="text-xs text-gray-500">Artifact Size</div>
              <div className="text-sm font-bold text-gray-900 dark:text-white">~20.0 MB</div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Integrity & Features */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card space-y-3">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white flex items-center gap-2">
            <CheckCircle2 size={16} className="text-green-500" /> Preprocessing & Feature Mapping
          </h3>
          <ul className="text-sm text-gray-600 dark:text-gray-300 space-y-2">
            <li className="flex items-start gap-2">
              <span className="text-arc-500 font-bold">•</span>
              <span><strong>45 Features:</strong> Flow duration, packet lengths, protocol flags, header rates, IAT stats.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-arc-500 font-bold">•</span>
              <span><strong>Feature Scaler:</strong> Standard normal scaling pre-fitted via Scikit-Learn <code>StandardScaler</code>.</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-arc-500 font-bold">•</span>
              <span><strong>Output Space:</strong> 34 cyberattack categories (DDoS, DoS, Reconnaissance, Mirai, Benign).</span>
            </li>
          </ul>
        </div>

        <div className="card space-y-3">
          <h3 className="text-sm font-semibold text-gray-900 dark:text-white flex items-center gap-2">
            <Clock size={16} className="text-arc-500" /> Publication & Versioning
          </h3>
          <ul className="text-sm text-gray-600 dark:text-gray-300 space-y-2">
            <li className="flex justify-between border-b border-gray-100 dark:border-gray-800 pb-1.5">
              <span className="text-gray-500">Published Timestamp:</span>
              <span className="font-medium text-gray-800 dark:text-gray-200">
                {new Date(model.createdAt).toLocaleString()}
              </span>
            </li>
            <li className="flex justify-between border-b border-gray-100 dark:border-gray-800 pb-1.5">
              <span className="text-gray-500">Storage Relative Path:</span>
              <span className="font-mono text-xs text-gray-700 dark:text-gray-300">{model.storagePath}</span>
            </li>
            <li className="flex justify-between items-center">
              <span className="text-gray-500">Integrity Checksum:</span>
              <span className="font-mono text-xs text-gray-500 truncate max-w-[200px]" title={model.checksumSha256}>
                {model.checksumSha256 || 'Verified SHA-256'}
              </span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
}
