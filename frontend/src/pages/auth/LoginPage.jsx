// LoginPage - Email/password login with role-based redirect

import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { Eye, EyeOff, LogIn } from 'lucide-react'
import AuthLayout from '../../layouts/AuthLayout'

export default function LoginPage({ onLogin }) {
  const navigate = useNavigate()
  const [form, setForm] = useState({ email: '', password: '' })
  const [showPass, setShowPass] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(e) {
    e.preventDefault()
    setLoading(true)
    setError('')
    try {
      const user = await onLogin(form.email, form.password)
      // Role-based redirect
      if (user.role === 'ADMIN') {
        navigate('/admin/overview')
      } else {
        navigate('/client/overview')
      }
    } catch (err) {
      setError(err.message || 'Invalid email or password')
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthLayout title="Sign In" subtitle="Access your federated learning workspace">
      <form onSubmit={handleSubmit} className="space-y-5">
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400 text-sm rounded-lg px-4 py-3">
            {error}
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Email Address
          </label>
          <input
            type="email"
            required
            className="input"
            placeholder="admin@arc-fl.local"
            value={form.email}
            onChange={(e) => setForm({ ...form, email: e.target.value })}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Password
          </label>
          <div className="relative">
            <input
              type={showPass ? 'text' : 'password'}
              required
              className="input pr-10"
              placeholder="••••••••"
              value={form.password}
              onChange={(e) => setForm({ ...form, password: e.target.value })}
            />
            <button
              type="button"
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
              onClick={() => setShowPass(!showPass)}
            >
              {showPass ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>
        </div>

        <button type="submit" disabled={loading} className="btn-primary w-full flex items-center justify-center gap-2">
          <LogIn size={16} />
          {loading ? 'Signing in...' : 'Sign In'}
        </button>

        <p className="text-center text-sm text-gray-500 dark:text-gray-400">
          Don't have an account?{' '}
          <Link to="/register" className="text-arc-600 dark:text-arc-400 hover:underline font-medium">
            Register
          </Link>
        </p>

        {/* Quick fill hints for demo */}
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <p className="text-xs text-gray-400 text-center mb-2">Demo accounts:</p>
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => setForm({ email: 'admin@arc-fl.local', password: 'Admin@123' })}
              className="text-xs bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-600 dark:text-gray-400 rounded-md px-2 py-1 transition-colors"
            >
              Admin
            </button>
            <button
              type="button"
              onClick={() => setForm({ email: 'client1@arc-fl.local', password: 'Client@123' })}
              className="text-xs bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-600 dark:text-gray-400 rounded-md px-2 py-1 transition-colors"
            >
              Client 1
            </button>
          </div>
        </div>
      </form>
    </AuthLayout>
  )
}
