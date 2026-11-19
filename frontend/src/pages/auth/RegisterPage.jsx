// RegisterPage - New account registration

import { useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import { UserPlus } from 'lucide-react'
import AuthLayout from '../../layouts/AuthLayout'

export default function RegisterPage({ onRegister }) {
  const navigate = useNavigate()
  const [form, setForm] = useState({ name: '', email: '', password: '', role: 'CLIENT' })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(e) {
    e.preventDefault()
    if (form.password.length < 8) {
      setError('Password must be at least 8 characters')
      return
    }
    setLoading(true)
    setError('')
    try {
      const user = await onRegister(form)
      navigate(user.role === 'ADMIN' ? '/admin/overview' : '/client/overview')
    } catch (err) {
      setError(err.message || 'Registration failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <AuthLayout title="Create Account" subtitle="Join the federated learning network">
      <form onSubmit={handleSubmit} className="space-y-4">
        {error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 text-red-700 dark:text-red-400 text-sm rounded-lg px-4 py-3">
            {error}
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Full Name</label>
          <input
            type="text" required className="input" placeholder="Edge Node Alpha"
            value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Email Address</label>
          <input
            type="email" required className="input" placeholder="node@example.com"
            value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Password</label>
          <input
            type="password" required className="input" placeholder="Min. 8 characters"
            value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Role</label>
          <select
            className="input"
            value={form.role}
            onChange={(e) => setForm({ ...form, role: e.target.value })}
          >
            <option value="CLIENT">Client (Edge Node)</option>
            <option value="ADMIN">Administrator</option>
          </select>
        </div>

        <button type="submit" disabled={loading} className="btn-primary w-full flex items-center justify-center gap-2">
          <UserPlus size={16} />
          {loading ? 'Creating account...' : 'Create Account'}
        </button>

        <p className="text-center text-sm text-gray-500 dark:text-gray-400">
          Already have an account?{' '}
          <Link to="/login" className="text-arc-600 dark:text-arc-400 hover:underline font-medium">Sign in</Link>
        </p>
      </form>
    </AuthLayout>
  )
}
