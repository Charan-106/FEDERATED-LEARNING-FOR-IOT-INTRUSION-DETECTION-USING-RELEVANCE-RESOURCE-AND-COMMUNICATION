// ARC-FL Platform - Main App Router
// Sets up React Router with role-based protected routes

import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { useAuth } from './hooks/useAuth'

// Layouts
import AuthLayout from './layouts/AuthLayout'
import AdminLayout from './layouts/AdminLayout'
import ClientLayout from './layouts/ClientLayout'

// Auth pages
import LoginPage from './pages/auth/LoginPage'
import RegisterPage from './pages/auth/RegisterPage'

// Admin pages
import AdminOverviewPage from './pages/admin/AdminOverviewPage'
import AdminTrainingMonitorPage from './pages/admin/AdminTrainingMonitorPage'
import AdminClientsPage from './pages/admin/AdminClientsPage'
import AdminModelPage from './pages/admin/AdminModelPage'
import AdminSubmissionsPage from './pages/admin/AdminSubmissionsPage'
import AdminArcPage from './pages/admin/AdminArcPage'
import AdminAggregationPage from './pages/admin/AdminAggregationPage'
import AdminRoundsPage from './pages/admin/AdminRoundsPage'
import AdminAnalyticsPage from './pages/admin/AdminAnalyticsPage'
import AdminArchitecturePage from './pages/admin/AdminArchitecturePage'

// Client pages
import ClientOverviewPage from './pages/client/ClientOverviewPage'
import ClientModelPage from './pages/client/ClientModelPage'
import ClientTrainingPage from './pages/client/ClientTrainingPage'
import ClientTrainingMonitorPage from './pages/client/ClientTrainingMonitorPage'
import ClientSubmissionsPage from './pages/client/ClientSubmissionsPage'
import ClientArcPage from './pages/client/ClientArcPage'

/**
 * ProtectedRoute - Redirects to /login if not authenticated or wrong role.
 */
function ProtectedRoute({ children, requiredRole }) {
  const { isAuthenticated, user } = useAuth()
  if (!isAuthenticated) return <Navigate to="/login" replace />
  if (requiredRole && user?.role !== requiredRole) {
    return <Navigate to={user?.role === 'ADMIN' ? '/admin/overview' : '/client/overview'} replace />
  }
  return children
}

export default function App() {
  const { user, login, logout, register, isAuthenticated } = useAuth()

  return (
    <BrowserRouter>
      <Routes>
        {/* Root redirect based on role */}
        <Route
          path="/"
          element={
            isAuthenticated
              ? <Navigate to={user?.role === 'ADMIN' ? '/admin/overview' : '/client/overview'} replace />
              : <Navigate to="/login" replace />
          }
        />

        {/* Auth routes */}
        <Route path="/login" element={<LoginPage onLogin={login} />} />
        <Route path="/register" element={<RegisterPage onRegister={register} />} />

        {/* Admin routes */}
        <Route
          path="/admin"
          element={
            <ProtectedRoute requiredRole="ADMIN">
              <AdminLayout user={user} onLogout={logout} />
            </ProtectedRoute>
          }
        >
          <Route index element={<Navigate to="overview" replace />} />
          <Route path="overview"          element={<AdminOverviewPage />} />
          <Route path="training-monitor"  element={<AdminTrainingMonitorPage />} />
          <Route path="clients"           element={<AdminClientsPage />} />
          <Route path="model"             element={<AdminModelPage />} />
          <Route path="submissions"       element={<AdminSubmissionsPage />} />
          <Route path="arc"               element={<AdminArcPage />} />
          <Route path="aggregation"       element={<AdminAggregationPage />} />
          <Route path="rounds"            element={<AdminRoundsPage />} />
          <Route path="analytics"         element={<AdminAnalyticsPage />} />
          <Route path="architecture"      element={<AdminArchitecturePage />} />
        </Route>

        {/* Client routes */}
        <Route
          path="/client"
          element={
            <ProtectedRoute requiredRole="CLIENT">
              <ClientLayout user={user} onLogout={logout} />
            </ProtectedRoute>
          }
        >
          <Route index element={<Navigate to="overview" replace />} />
          <Route path="overview"          element={<ClientOverviewPage />} />
          <Route path="model"             element={<ClientModelPage />} />
          <Route path="training"          element={<ClientTrainingPage />} />
          <Route path="training-monitor"  element={<ClientTrainingMonitorPage />} />
          <Route path="submissions"       element={<ClientSubmissionsPage />} />
          <Route path="arc"               element={<ClientArcPage />} />
        </Route>

        {/* Catch-all */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
