// useAuth hook - Manages authentication state for the ARC-FL Platform
// Persists JWT and user info in localStorage

import { useState, useCallback, useMemo } from 'react'
import { authApi } from '../services/api'

const TOKEN_KEY = 'arc_fl_token'
const USER_KEY = 'arc_fl_user'

/**
 * Decode JWT payload without verification (browser-side only).
 * We rely on the server to verify; this is just for reading claims.
 */
function decodeJwt(token) {
  try {
    const payload = token.split('.')[1]
    return JSON.parse(atob(payload))
  } catch {
    return null
  }
}

export function useAuth() {
  const [token, setToken] = useState(() => localStorage.getItem(TOKEN_KEY))
  const [user, setUser] = useState(() => {
    const stored = localStorage.getItem(USER_KEY)
    return stored ? JSON.parse(stored) : null
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const persistSession = useCallback((newToken, newUser) => {
    localStorage.setItem(TOKEN_KEY, newToken)
    localStorage.setItem(USER_KEY, JSON.stringify(newUser))
    setToken(newToken)
    setUser(newUser)
  }, [])

  const login = useCallback(async (email, password) => {
    setLoading(true)
    setError(null)
    try {
      const { token: newToken, user: newUser } = await authApi.login(email, password)
      persistSession(newToken, newUser)
      return newUser
    } catch (err) {
      setError(err.message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [persistSession])

  const register = useCallback(async (data) => {
    setLoading(true)
    setError(null)
    try {
      const { token: newToken, user: newUser } = await authApi.register(data)
      persistSession(newToken, newUser)
      return newUser
    } catch (err) {
      setError(err.message)
      throw err
    } finally {
      setLoading(false)
    }
  }, [persistSession])

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
    setToken(null)
    setUser(null)
  }, [])

  // Derived role helpers
  const isAdmin = useMemo(() => user?.role === 'ADMIN', [user])
  const isClient = useMemo(() => user?.role === 'CLIENT', [user])
  const isAuthenticated = useMemo(() => !!token && !!user, [token, user])

  return {
    user,
    token,
    loading,
    error,
    login,
    logout,
    register,
    isAdmin,
    isClient,
    isAuthenticated,
  }
}
