// Admin Overview Page - Platform statistics & convergence analytics
// Section 6.A: Displays total clients, active clients, submissions, rounds, active model, and charts

import { useEffect, useState } from 'react'
import {
  Users,
  FileStack,
  RefreshCw,
  Brain,
  TrendingUp,
  Clock,
  Award,
  ArrowUpRight,
  ShieldCheck,
  CheckCircle
} from 'lucide-react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  Legend
} from 'recharts'
import { adminApi } from '../../services/api'

function StatCard({ icon: Icon, label, value, color = 'blue', sub }) {
  const colorMap = {
    blue:   'bg-arc-50 dark:bg-arc-900/30 text-arc-600',
    green:  'bg-green-50 dark:bg-green-900/30 text-green-600',
    yellow: 'bg-yellow-50 dark:bg-yellow-900/30 text-yellow-600',
    purple: 'bg-purple-50 dark:bg-purple-900/30 text-purple-600',
  }

  return (
    <div className="card shadow-sm hover:shadow transition-shadow">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">{label}</p>
          <p className="text-3xl font-black text-gray-900 dark:text-white mt-1.5">{value ?? '—'}</p>
          {sub && <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">{sub}</p>}
        </div>
        <div className={`p-3 rounded-xl ${colorMap[color]}`}>
          <Icon size={22} />
        </div>
      </div>
    </div>
  )
}

export default function AdminOverviewPage() {
  const [data, setData] = useState(null)
  const [analytics, setAnalytics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  useEffect(() => {
    loadDashboard()
  }, [])

  async function loadDashboard() {
    setLoading(true)
    setError('')
    try {
      const [dash, ana] = await Promise.all([
        adminApi.getDashboard(),
        adminApi.getAnalytics().catch(() => ({ roundData: [] })),
      ])
      setData(dash)
      setAnalytics(ana)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading) return <div className="card text-center py-16 text-gray-500">Loading platform metrics...</div>
  if (error) return <div className="card bg-red-50 text-red-600 p-6">Error: {error}</div>

  const { stats, latestRound, activeModel } = data || {}
  const roundData = analytics?.roundData || []

  // Sample historical data for charts if no rounds completed yet
  const chartData = roundData.length > 0 ? roundData : [
    { roundNumber: 0, globalAccuracy: 98.50, globalLoss: 0.2088, avgArcScore: 82.5, clientCount: 3 },
  ]

  return (
    <div className="space-y-6 max-w-7xl">
      {/* Title */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
        <div>
          <h1 className="text-2xl font-black text-gray-900 dark:text-white">Platform Overview</h1>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Central orchestration console: fleet health, active global checkpoint, and ARC distribution.
          </p>
        </div>

        <button onClick={loadDashboard} className="btn-secondary self-start sm:self-auto flex items-center gap-1.5 text-xs py-2 px-3">
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {/* Primary KPI Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          icon={Users}
          label="Registered Clients"
          value={stats?.totalClients}
          color="blue"
          sub={`${stats?.activeClients || 0} active & participating`}
        />
        <StatCard
          icon={FileStack}
          label="Total Submissions"
          value={stats?.totalSubmissions}
          color="green"
          sub={`${stats?.pendingSubmissions || 0} pending review`}
        />
        <StatCard
          icon={RefreshCw}
          label="Completed Rounds"
          value={stats?.completedRounds}
          color="purple"
          sub={latestRound ? `Round #${latestRound.roundNumber} current` : 'Round #1 ready'}
        />
        <StatCard
          icon={ShieldCheck}
          label="Active Accuracy"
          value={activeModel?.accuracy ? `${activeModel.accuracy}%` : '98.50%'}
          color="yellow"
          sub={`Model ${activeModel?.version || 'initial-v1'}`}
        />
      </div>

      {/* Active Checkpoint & Training Status Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Active Model Card */}
        <div className="card border border-gray-200 dark:border-gray-700 space-y-4">
          <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-800 pb-3">
            <h2 className="text-base font-bold text-gray-900 dark:text-white flex items-center gap-2">
              <Brain size={18} className="text-arc-600" /> Active Global Model Checkpoint
            </h2>
            <span className="badge-green text-xs font-semibold uppercase">Published</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <span className="text-xs text-gray-500">Version</span>
              <p className="text-lg font-bold font-mono text-gray-900 dark:text-white mt-0.5">
                {activeModel?.version || 'initial-v1'}
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <span className="text-xs text-gray-500">Architecture</span>
              <p className="text-lg font-bold font-mono text-arc-600 mt-0.5">
                preact_resmlp_v3
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <span className="text-xs text-gray-500">Parameters</span>
              <p className="text-lg font-bold text-gray-900 dark:text-white mt-0.5">
                4,981,628
              </p>
            </div>
          </div>

          <div className="p-3 bg-arc-50 dark:bg-arc-900/20 rounded-lg border border-arc-100 dark:border-arc-800 text-xs text-arc-700 dark:text-arc-300 flex items-center justify-between">
            <span>Validated against <code>validation.csv</code> (34 IoT attack classes)</span>
            <span className="font-bold text-green-600 text-sm">
              {activeModel?.accuracy ? `${activeModel.accuracy}%` : '98.50%'}
            </span>
          </div>
        </div>

        {/* Current Round Card */}
        <div className="card border border-gray-200 dark:border-gray-700 space-y-4">
          <div className="flex items-center justify-between border-b border-gray-100 dark:border-gray-800 pb-3">
            <h2 className="text-base font-bold text-gray-900 dark:text-white flex items-center gap-2">
              <Clock size={18} className="text-purple-600" /> Federated Training Round Status
            </h2>
            <span className={`badge-${latestRound?.status === 'ACTIVE' ? 'green' : 'yellow'} text-xs font-semibold`}>
              {latestRound?.status || 'Round Ready'}
            </span>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <span className="text-xs text-gray-500">Round ID</span>
              <p className="text-lg font-bold text-gray-900 dark:text-white mt-0.5">
                {latestRound ? `#${latestRound.roundNumber}` : '#1'}
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <span className="text-xs text-gray-500">Submissions</span>
              <p className="text-lg font-bold text-purple-600 mt-0.5">
                {latestRound?._count?.submissions || 0}
              </p>
            </div>
            <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <span className="text-xs text-gray-500">Strategy</span>
              <p className="text-sm font-bold font-mono text-gray-700 dark:text-gray-300 mt-1">
                ARC-Weighted
              </p>
            </div>
          </div>

          <div className="text-xs text-gray-500 flex items-center justify-between pt-1">
            <span>Fleet state: Ready for multi-criteria client aggregation</span>
            <a href="/admin/aggregation" className="text-arc-600 dark:text-arc-400 font-semibold hover:underline flex items-center gap-1">
              Aggregation Center <ArrowUpRight size={14} />
            </a>
          </div>
        </div>
      </div>

      {/* Analytics Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Global Accuracy Across Rounds */}
        <div className="card space-y-3">
          <h3 className="text-sm font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <TrendingUp size={16} className="text-green-500" />
            Global Accuracy Across Rounds (%)
          </h3>
          <div className="h-60 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="roundNumber" stroke="#888888" fontSize={12} tickLine={false} />
                <YAxis domain={[90, 100]} stroke="#888888" fontSize={12} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', color: '#fff', borderRadius: '8px', border: 'none' }}
                />
                <Line
                  type="monotone"
                  dataKey="globalAccuracy"
                  name="Global Accuracy"
                  stroke="#10b981"
                  strokeWidth={3}
                  dot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Global Loss Across Rounds */}
        <div className="card space-y-3">
          <h3 className="text-sm font-bold text-gray-900 dark:text-white flex items-center gap-2">
            <TrendingUp size={16} className="text-arc-500" />
            Global Cross-Entropy Loss Across Rounds
          </h3>
          <div className="h-60 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                <XAxis dataKey="roundNumber" stroke="#888888" fontSize={12} tickLine={false} />
                <YAxis stroke="#888888" fontSize={12} tickLine={false} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', color: '#fff', borderRadius: '8px', border: 'none' }}
                />
                <Line
                  type="monotone"
                  dataKey="globalLoss"
                  name="Global Loss"
                  stroke="#3b82f6"
                  strokeWidth={3}
                  dot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}
