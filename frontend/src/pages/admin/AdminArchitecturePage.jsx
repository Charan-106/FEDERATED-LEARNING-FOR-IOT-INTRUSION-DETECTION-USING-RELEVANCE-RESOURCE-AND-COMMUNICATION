// Admin Architecture Page - Interactive Federated System Topology
// Section 12: Interactive system architecture diagram with vertical clients and live service reachability

import { useState, useEffect } from 'react'
import {
  Network,
  Server,
  Database,
  Cpu,
  Brain,
  HardDrive,
  ArrowDown,
  ArrowRight,
  ArrowLeft,
  CheckCircle2,
  Info,
  Layers,
  Shield,
  Zap,
  RefreshCw,
  AlertCircle
} from 'lucide-react'
import { adminApi } from '../../services/api'

export default function AdminArchitecturePage() {
  const [activeFlow, setActiveFlow] = useState('ALL') // ALL, DOWNLOAD, TRAIN, SUBMIT, ARC, AGGREGATE, PUBLISH
  const [systemHealth, setSystemHealth] = useState(null)
  const [healthLoading, setHealthLoading] = useState(true)
  const [lastChecked, setLastChecked] = useState(null)
  const [healthError, setHealthError] = useState(null)

  async function fetchHealth() {
    setHealthLoading(true)
    setHealthError(null)
    try {
      const res = await adminApi.getSystemHealth()
      setSystemHealth(res)
      setLastChecked(new Date())
    } catch (err) {
      setHealthError(err.message || 'Failed to connect to backend health proxy')
    } finally {
      setHealthLoading(false)
    }
  }

  useEffect(() => {
    fetchHealth()
    const timer = setInterval(fetchHealth, 15000)
    return () => clearInterval(timer)
  }, [])

  const flows = [
    { id: 'ALL', label: 'Complete System Architecture' },
    { id: 'DOWNLOAD', label: '1. Model Download Flow' },
    { id: 'TRAIN', label: '2. Local On-Device Training' },
    { id: 'SUBMIT', label: '3. Update Submission Flow' },
    { id: 'ARC', label: '4. ARC Evaluation Flow' },
    { id: 'AGGREGATE', label: '5. Federated Aggregation' },
    { id: 'PUBLISH', label: '6. Global Publication Flow' },
  ]

  return (
    <div className="space-y-6 max-w-6xl">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-black text-gray-900 dark:text-white flex items-center gap-2">
            <Network className="text-arc-600" size={28} />
            Interactive System Architecture
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Architectural topology showing 3-tier decoupling (React, Node.js Express, Python ML Service),
            relational database, and vertical edge IoT client nodes.
          </p>
        </div>
        <button
          onClick={fetchHealth}
          disabled={healthLoading}
          className="btn-secondary self-start sm:self-auto text-xs px-3 py-1.5 flex items-center gap-1.5"
        >
          <RefreshCw size={13} className={healthLoading ? 'animate-spin' : ''} />
          {healthLoading ? 'Checking...' : 'Refresh Status'}
        </button>
      </div>

      {/* Live System Health Banner (Phase 6 Live Verification) */}
      <div className="card p-4 bg-gray-50 dark:bg-gray-800/80 border border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between mb-3 border-b border-gray-200 dark:border-gray-700 pb-2">
          <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-wider text-gray-700 dark:text-gray-300">
            <Server size={15} className="text-arc-600" />
            Live Runtime Service Reachability (Backend-Proxied)
          </div>
          <div className="text-[11px] text-gray-400">
            Last Checked: {lastChecked ? lastChecked.toLocaleTimeString() : 'Pending...'}
          </div>
        </div>

        {healthError ? (
          <div className="p-2.5 bg-red-50 dark:bg-red-900/20 text-red-600 text-xs rounded border border-red-200 flex items-center gap-2">
            <AlertCircle size={15} />
            <span>Health Check Error: {healthError}</span>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {/* Node Backend */}
            <div className="p-3 bg-white dark:bg-gray-700/60 rounded-lg border border-gray-200 dark:border-gray-600">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-gray-600 dark:text-gray-300">Node Express API</span>
                <span className="badge-green text-[10px]">Port {systemHealth?.nodeBackend?.port || 5000}</span>
              </div>
              <div className="mt-2 flex items-center gap-1.5 text-xs font-bold text-green-600 dark:text-green-400">
                <CheckCircle2 size={14} /> {systemHealth?.nodeBackend?.status === 'online' ? 'ONLINE (ACTIVE)' : 'CHECKING...'}
              </div>
              <div className="text-[10px] text-gray-400 mt-1">
                Uptime: {systemHealth?.nodeBackend?.uptimeSeconds ? `${Math.floor(systemHealth.nodeBackend.uptimeSeconds / 60)} min` : 'N/A'} • RSS: {systemHealth?.nodeBackend?.memoryRssMb || 0} MB
              </div>
            </div>

            {/* PostgreSQL Database */}
            <div className="p-3 bg-white dark:bg-gray-700/60 rounded-lg border border-gray-200 dark:border-gray-600">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-gray-600 dark:text-gray-300">PostgreSQL DB</span>
                <span className="badge-blue text-[10px]">Port 5433</span>
              </div>
              <div className="mt-2 flex items-center gap-1.5 text-xs font-bold text-emerald-600 dark:text-emerald-400">
                <CheckCircle2 size={14} /> {systemHealth?.database?.status === 'connected' ? 'CONNECTED' : 'CHECKING...'}
              </div>
              <div className="text-[10px] text-gray-400 mt-1">
                Latency: {systemHealth?.database?.latencyMs !== null ? `${systemHealth?.database?.latencyMs} ms` : 'N/A'} • Prisma ORM
              </div>
            </div>

            {/* Python ML Service */}
            <div className="p-3 bg-white dark:bg-gray-700/60 rounded-lg border border-gray-200 dark:border-gray-600">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-gray-600 dark:text-gray-300">Python ML Service</span>
                <span className="badge-purple text-[10px]">Port {systemHealth?.pythonMlService?.port || 8000}</span>
              </div>
              <div className={`mt-2 flex items-center gap-1.5 text-xs font-bold ${systemHealth?.pythonMlService?.status === 'online' ? 'text-purple-600 dark:text-purple-400' : 'text-red-500'}`}>
                {systemHealth?.pythonMlService?.status === 'online' ? (
                  <>
                    <CheckCircle2 size={14} /> ONLINE (PyTorch 2.14)
                  </>
                ) : (
                  <>
                    <AlertCircle size={14} /> {systemHealth?.pythonMlService?.status ? 'UNREACHABLE' : 'CHECKING...'}
                  </>
                )}
              </div>
              <div className="text-[10px] text-gray-400 mt-1 truncate">
                {systemHealth?.pythonMlService?.details?.modelParams ? `ResMLP: 4,981,628 params • Ping: ${systemHealth.pythonMlService.latencyMs} ms` : (systemHealth?.pythonMlService?.error || 'Checking...')}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Flow Walkthrough Filter Bar */}
      <div className="card bg-arc-50/40 dark:bg-gray-800/60 border border-arc-200 dark:border-gray-700 p-3">
        <div className="flex items-center gap-2 overflow-x-auto pb-1 sm:pb-0">
          <span className="text-xs font-bold text-gray-700 dark:text-gray-300 uppercase tracking-wide flex items-center gap-1">
            <Zap size={14} className="text-arc-600" /> Interactive Flows:
          </span>
          {flows.map((f) => (
            <button
              key={f.id}
              onClick={() => setActiveFlow(f.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold whitespace-nowrap transition-colors ${
                activeFlow === f.id
                  ? 'bg-arc-600 text-white shadow-sm'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 border border-gray-200 dark:border-gray-700'
              }`}
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Diagram Canvas */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Edge Client Fleet (STRICT VERTICAL ARRANGEMENT) */}
        <div className="lg:col-span-5 space-y-4">
          <div className="p-2.5 bg-gray-100 dark:bg-gray-800 rounded-lg text-center font-bold text-xs uppercase tracking-wider text-gray-700 dark:text-gray-300 flex items-center justify-center gap-2">
            <Cpu size={15} className="text-arc-600" />
            Distributed Edge Client Fleet (Vertical Arrangement)
          </div>

          {/* Client 1 */}
          <div className={`card border-2 transition-all ${
            ['ALL', 'DOWNLOAD', 'TRAIN', 'SUBMIT'].includes(activeFlow)
              ? 'border-arc-500 shadow-md bg-arc-50/20 dark:bg-gray-800'
              : 'border-gray-200 dark:border-gray-700 opacity-60'
          }`}>
            <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-800 pb-2">
              <span className="font-bold text-sm text-gray-900 dark:text-white flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-green-500"></span> Client 1 (client-001)
              </span>
              <span className="text-[11px] font-mono text-arc-600 font-semibold">127.0.0.1</span>
            </div>
            <ul className="text-xs text-gray-600 dark:text-gray-300 mt-2 space-y-1 font-mono">
              <li>• Local Dataset: <code>client1_traffic.csv</code> (600 rows)</li>
              <li>• Architecture: ResMLP (preact_resmlp_v3)</li>
              <li>• Training: On-device SGD / Adam (PyTorch)</li>
              <li>• Submission: Serialized <code>.pt</code> parameter state dict</li>
            </ul>
          </div>

          {/* Client 2 */}
          <div className={`card border-2 transition-all ${
            ['ALL', 'DOWNLOAD', 'TRAIN', 'SUBMIT'].includes(activeFlow)
              ? 'border-arc-500 shadow-md bg-arc-50/20 dark:bg-gray-800'
              : 'border-gray-200 dark:border-gray-700 opacity-60'
          }`}>
            <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-800 pb-2">
              <span className="font-bold text-sm text-gray-900 dark:text-white flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-green-500"></span> Client 2 (client-002)
              </span>
              <span className="text-[11px] font-mono text-arc-600 font-semibold">127.0.0.1</span>
            </div>
            <ul className="text-xs text-gray-600 dark:text-gray-300 mt-2 space-y-1 font-mono">
              <li>• Local Dataset: <code>client2_traffic.csv</code> (600 rows)</li>
              <li>• Architecture: ResMLP (preact_resmlp_v3)</li>
              <li>• Training: On-device SGD / Adam (PyTorch)</li>
              <li>• Submission: Serialized <code>.pt</code> parameter state dict</li>
            </ul>
          </div>

          {/* Client 3 */}
          <div className={`card border-2 transition-all ${
            ['ALL', 'DOWNLOAD', 'TRAIN', 'SUBMIT'].includes(activeFlow)
              ? 'border-arc-500 shadow-md bg-arc-50/20 dark:bg-gray-800'
              : 'border-gray-200 dark:border-gray-700 opacity-60'
          }`}>
            <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-800 pb-2">
              <span className="font-bold text-sm text-gray-900 dark:text-white flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-green-500"></span> Client 3 (client-003)
              </span>
              <span className="text-[11px] font-mono text-arc-600 font-semibold">127.0.0.1</span>
            </div>
            <ul className="text-xs text-gray-600 dark:text-gray-300 mt-2 space-y-1 font-mono">
              <li>• Local Dataset: <code>client3_traffic.csv</code> (600 rows)</li>
              <li>• Architecture: ResMLP (preact_resmlp_v3)</li>
              <li>• Training: On-device SGD / Adam (PyTorch)</li>
              <li>• Submission: Serialized <code>.pt</code> parameter state dict</li>
            </ul>
          </div>

          <div className="p-3 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800 text-xs text-amber-800 dark:text-amber-300 flex items-start gap-2">
            <Shield size={16} className="mt-0.5 flex-shrink-0 text-amber-600" />
            <span>
              <strong>Zero Raw Data Leakage:</strong> Raw CSV packet rows never leave the client devices.
              Only float32 tensor weights and non-sensitive summary telemetry are transmitted.
            </span>
          </div>
        </div>

        {/* Right Column: Central Orchestrator & Backend Infrastructure */}
        <div className="lg:col-span-7 space-y-4">
          <div className="p-2.5 bg-gray-100 dark:bg-gray-800 rounded-lg text-center font-bold text-xs uppercase tracking-wider text-gray-700 dark:text-gray-300 flex items-center justify-center gap-2">
            <Server size={15} className="text-arc-600" />
            Central Server Infrastructure Stack
          </div>

          {/* Tier 1: React Frontend */}
          <div className="card border-2 border-blue-400 bg-blue-50/20 dark:bg-gray-800 p-4">
            <div className="flex items-center justify-between">
              <span className="font-bold text-sm text-blue-700 dark:text-blue-300 flex items-center gap-2">
                <Network size={16} /> React 18 + Vite Frontend Dashboard (Port 5173)
              </span>
              <span className="badge-blue text-[11px]">Role-Isolated Views</span>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-300 mt-1">
              Separate Client & Admin interfaces. Communicates with Node.js backend using JWT Bearer authentication.
            </p>
          </div>

          <div className="flex justify-center my-1 text-gray-400">
            <ArrowDown size={18} />
          </div>

          {/* Tier 2: Node.js Express Gateway */}
          <div className="card border-2 border-green-500 bg-green-50/20 dark:bg-gray-800 p-4">
            <div className="flex items-center justify-between">
              <span className="font-bold text-sm text-green-700 dark:text-green-300 flex items-center gap-2">
                <Server size={16} /> Node.js Express Backend Coordinator (Port 5000)
              </span>
              <span className="badge-green text-[11px]">JavaScript API Gateway</span>
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-300 mt-1">
              Stateless JWT authentication, role guards, Prisma ORM, file upload validator, and authenticated HTTP proxy to ML service.
            </p>
          </div>

          {/* Branching from Backend */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-2">
            {/* Database */}
            <div className="card border-2 border-purple-400 bg-purple-50/20 dark:bg-gray-800 p-3.5 space-y-1.5">
              <span className="font-bold text-xs text-purple-700 dark:text-purple-300 flex items-center gap-1.5">
                <Database size={15} /> PostgreSQL 18 + Prisma (Port 5433)
              </span>
              <p className="text-[11px] text-gray-600 dark:text-gray-300">
                Authoritative registry for Users, Profiles, Rounds, Submissions, ARC Scores, and Audit Logs.
              </p>
              <div className="text-[10px] font-mono text-purple-600 font-semibold">
                No model binaries stored in DB
              </div>
            </div>

            {/* Python ML Service */}
            <div className="card border-2 border-arc-500 bg-arc-50/20 dark:bg-gray-800 p-3.5 space-y-1.5">
              <span className="font-bold text-xs text-arc-700 dark:text-arc-300 flex items-center gap-1.5">
                <Cpu size={15} /> Python FastAPI ML Service (Port 8000)
              </span>
              <p className="text-[11px] text-gray-600 dark:text-gray-300">
                PyTorch 2.14, ResMLP architecture, ARC Engine v1.0, FedAvg/Equal/ARC-weighted aggregation engine.
              </p>
              <div className="text-[10px] font-mono text-arc-600 font-semibold">
                Protected via X-API-Key header
              </div>
            </div>
          </div>

          <div className="flex justify-center my-1 text-gray-400">
            <ArrowDown size={18} />
          </div>

          {/* Tier 3: Artifact Storage Engine */}
          <div className="card border-2 border-amber-400 bg-amber-50/20 dark:bg-gray-800 p-4">
            <div className="flex items-center justify-between">
              <span className="font-bold text-sm text-amber-700 dark:text-amber-300 flex items-center gap-2">
                <HardDrive size={16} /> Dedicated Artifact Storage (Filesystem Abstraction)
              </span>
              <span className="badge-yellow text-[11px]">S3-Ready Storage</span>
            </div>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-2 font-mono text-[11px] text-gray-600 dark:text-gray-300">
              <div className="bg-white dark:bg-gray-700 p-2 rounded text-center">
                <code>global-models/</code>
              </div>
              <div className="bg-white dark:bg-gray-700 p-2 rounded text-center">
                <code>client-updates/</code>
              </div>
              <div className="bg-white dark:bg-gray-700 p-2 rounded text-center">
                <code>candidates/</code>
              </div>
              <div className="bg-white dark:bg-gray-700 p-2 rounded text-center">
                <code>datasets/</code>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Step by Step Flow Walkthrough Card */}
      <div className="card bg-gray-50 dark:bg-gray-800/60 border border-gray-200 dark:border-gray-700 space-y-3">
        <h3 className="font-bold text-sm text-gray-900 dark:text-white flex items-center gap-2">
          <Info size={16} className="text-arc-600" />
          Selected Walkthrough: {flows.find(f => f.id === activeFlow)?.label}
        </h3>
        <p className="text-xs text-gray-600 dark:text-gray-300 leading-relaxed">
          {activeFlow === 'DOWNLOAD' && (
            'Edge nodes request the latest verified model artifact via GET /api/client/model/download. The Node.js coordinator streams best.pt alongside preprocessing scaler.pkl and label_encoder.pkl directly from storage/global-models/.'
          )}
          {activeFlow === 'TRAIN' && (
            'Clients train the ResMLP neural network locally using PyTorch and Adam optimizer on their private network traffic partition (e.g., client1_traffic.csv) without communicating with the coordinator during mini-batch forward/backward passes.'
          )}
          {activeFlow === 'SUBMIT' && (
            'Upon completing local SGD epochs, the client transmits only the serialized PyTorch state dictionary and telemetry (accuracy, loss, samples, RAM headroom, CPU idle) via POST /api/client/submit.'
          )}
          {activeFlow === 'ARC' && (
            'The Python ML service calculates Relevance (R_i), Resource (S_i), and Communication (C_i) scores normalized to [0, 100]. Overall ARC score is computed as: ARC_i = 0.40·R_i + 0.30·S_i + 0.30·C_i and stored per round.'
          )}
          {activeFlow === 'AGGREGATE' && (
            'The administrator selects participating clients and chooses an aggregation formula (FedAvg, Equal, or ARC-Weighted). The Python engine validates tensor shapes and executes weighted parameter averaging with BatchNorm handling.'
          )}
          {activeFlow === 'PUBLISH' && (
            'The candidate is benchmarked against validation.csv. If accuracy meets criteria, the admin publishes the model as global-v{N}, updating the authoritative pointer while preserving baseline checkpoints for instant rollback.'
          )}
          {activeFlow === 'ALL' && (
            'The complete lifecycle ensures rigorous privacy, verifiable multi-criteria ARC weighting, atomic versioning, and zero direct exposure of Python machine-learning internals to client browsers.'
          )}
        </p>
      </div>
    </div>
  )
}
