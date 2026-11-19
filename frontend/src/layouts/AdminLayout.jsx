// AdminLayout - Sidebar navigation layout for the admin dashboard

import { useState } from 'react'
import { NavLink, useNavigate, Outlet } from 'react-router-dom'
import {
  LayoutDashboard,
  Users,
  Brain,
  FileStack,
  BarChart3,
  GitMerge,
  RefreshCw,
  TrendingUp,
  Network,
  LogOut,
  Menu,
  X,
  Moon,
  Sun,
  Activity,
} from 'lucide-react'
import { useTheme } from '../hooks/useTheme'

const NAV_ITEMS = [
  { to: '/admin/overview',         label: 'Overview',         icon: LayoutDashboard },
  { to: '/admin/training-monitor', label: 'Training Monitor', icon: Activity },
  { to: '/admin/clients',          label: 'Clients',          icon: Users },
  { to: '/admin/model',            label: 'Global Model',     icon: Brain },
  { to: '/admin/submissions',      label: 'Submissions',      icon: FileStack },
  { to: '/admin/arc',              label: 'ARC Evaluation',   icon: BarChart3 },
  { to: '/admin/aggregation',      label: 'Aggregation',      icon: GitMerge },
  { to: '/admin/rounds',           label: 'Rounds',           icon: RefreshCw },
  { to: '/admin/analytics',        label: 'Analytics',        icon: TrendingUp },
  { to: '/admin/architecture',     label: 'Architecture',     icon: Network },
]

export default function AdminLayout({ user, onLogout }) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const { isDark, toggleTheme } = useTheme()
  const navigate = useNavigate()

  function handleLogout() {
    onLogout()
    navigate('/login')
  }

  const SidebarContent = () => (
    <div className="flex flex-col h-full">
      {/* Brand */}
      <div className="flex items-center gap-3 px-6 py-5 border-b border-gray-200 dark:border-gray-700">
        <div className="w-8 h-8 bg-arc-600 rounded-lg flex items-center justify-center">
          <span className="text-white text-sm font-bold">FL</span>
        </div>
        <div>
          <div className="font-semibold text-gray-900 dark:text-white text-sm">ARC-FL Admin</div>
          <div className="text-xs text-gray-500 dark:text-gray-400">Platform Management</div>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {NAV_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            onClick={() => setSidebarOpen(false)}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                isActive
                  ? 'bg-arc-50 dark:bg-arc-900/40 text-arc-700 dark:text-arc-300'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 hover:text-gray-900 dark:hover:text-gray-100'
              }`
            }
          >
            <Icon size={16} />
            {label}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-4 py-4 border-t border-gray-200 dark:border-gray-700 space-y-2">
        <div className="text-xs text-gray-500 dark:text-gray-400 px-2">
          Signed in as <span className="font-medium text-gray-700 dark:text-gray-300">{user?.name}</span>
        </div>
        <button
          onClick={handleLogout}
          className="flex items-center gap-2 w-full px-3 py-2 text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-lg transition-colors"
        >
          <LogOut size={16} />
          Sign Out
        </button>
      </div>
    </div>
  )

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900 overflow-hidden">
      {/* Desktop Sidebar */}
      <aside className="hidden lg:flex lg:flex-col w-56 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex-shrink-0">
        <SidebarContent />
      </aside>

      {/* Mobile Sidebar Overlay */}
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="fixed inset-0 bg-black/50" onClick={() => setSidebarOpen(false)} />
          <aside className="fixed left-0 top-0 bottom-0 w-56 bg-white dark:bg-gray-800 z-50">
            <SidebarContent />
          </aside>
        </div>
      )}

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Bar */}
        <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3 flex items-center justify-between">
          <button
            className="lg:hidden text-gray-500 hover:text-gray-900 dark:hover:text-white"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={20} />
          </button>
          <div className="flex items-center gap-3 ml-auto">
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            >
              {isDark ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{user?.name}</span>
            <span className="badge-blue">ADMIN</span>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-y-auto p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}
