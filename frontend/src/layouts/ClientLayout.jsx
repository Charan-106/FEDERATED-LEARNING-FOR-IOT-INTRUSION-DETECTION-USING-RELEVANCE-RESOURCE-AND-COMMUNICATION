// ClientLayout - Top navbar layout for the client dashboard

import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { LogOut, Moon, Sun, Cpu } from 'lucide-react'
import { useTheme } from '../hooks/useTheme'

const NAV_ITEMS = [
  { to: '/client/overview',         label: 'Overview' },
  { to: '/client/model',            label: 'Global Model' },
  { to: '/client/training',         label: 'Training Studio' },
  { to: '/client/training-monitor', label: 'Training Monitor' },
  { to: '/client/submissions',      label: 'Submissions' },
  { to: '/client/arc',              label: 'ARC Results' },
]

export default function ClientLayout({ user, onLogout }) {
  const { isDark, toggleTheme } = useTheme()
  const navigate = useNavigate()

  function handleLogout() {
    onLogout()
    navigate('/login')
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex flex-col">
      {/* Top Navbar */}
      <nav className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-14">
            {/* Brand */}
            <div className="flex items-center gap-3">
              <div className="w-7 h-7 bg-arc-600 rounded-lg flex items-center justify-center">
                <Cpu size={14} className="text-white" />
              </div>
              <span className="font-semibold text-gray-900 dark:text-white text-sm">ARC-FL</span>
            </div>

            {/* Nav Links */}
            <div className="hidden sm:flex items-center gap-1">
              {NAV_ITEMS.map(({ to, label }) => (
                <NavLink
                  key={to}
                  to={to}
                  className={({ isActive }) =>
                    `px-3 py-1.5 rounded-md text-sm font-medium transition-colors ${
                      isActive
                        ? 'bg-arc-100 dark:bg-arc-900/40 text-arc-700 dark:text-arc-300'
                        : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100'
                    }`
                  }
                >
                  {label}
                </NavLink>
              ))}
            </div>

            {/* Right section */}
            <div className="flex items-center gap-2">
              <button
                onClick={toggleTheme}
                className="p-1.5 rounded-md text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                {isDark ? <Sun size={15} /> : <Moon size={15} />}
              </button>
              <span className="text-sm text-gray-700 dark:text-gray-300 hidden sm:block">{user?.name}</span>
              <span className="badge-green hidden sm:inline-flex">CLIENT</span>
              <button
                onClick={handleLogout}
                className="p-1.5 rounded-md text-gray-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/20 transition-colors"
                title="Sign out"
              >
                <LogOut size={15} />
              </button>
            </div>
          </div>
        </div>
      </nav>

      {/* Page Content */}
      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-6">
        <Outlet />
      </main>
    </div>
  )
}
