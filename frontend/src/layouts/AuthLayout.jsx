// AuthLayout - Centered card layout for Login and Register pages

export default function AuthLayout({ children, title, subtitle }) {
  return (
    <div className="min-h-screen bg-gradient-to-br from-arc-900 via-arc-800 to-arc-700 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo / Brand */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-white rounded-2xl shadow-lg mb-4">
            <span className="text-2xl font-bold text-arc-600">FL</span>
          </div>
          <h1 className="text-3xl font-bold text-white">ARC-FL Platform</h1>
          <p className="text-arc-200 mt-1 text-sm">Adaptive Resource-aware Contribution</p>
        </div>

        {/* Card */}
        <div className="bg-white dark:bg-gray-800 rounded-2xl shadow-2xl p-8">
          {title && (
            <div className="mb-6">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">{title}</h2>
              {subtitle && (
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">{subtitle}</p>
              )}
            </div>
          )}
          {children}
        </div>
      </div>
    </div>
  )
}
