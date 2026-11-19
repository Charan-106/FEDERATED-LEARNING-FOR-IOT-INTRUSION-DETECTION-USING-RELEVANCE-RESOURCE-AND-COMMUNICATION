// useTheme hook - Light/dark mode toggle with localStorage persistence

import { useState, useEffect, useCallback } from 'react'

const THEME_KEY = 'arc_fl_theme'

export function useTheme() {
  const [theme, setTheme] = useState(() => {
    const stored = localStorage.getItem(THEME_KEY)
    if (stored) return stored
    // Default to system preference
    return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
  })

  // Apply theme class to <html> element
  useEffect(() => {
    const root = document.documentElement
    if (theme === 'dark') {
      root.classList.add('dark')
    } else {
      root.classList.remove('dark')
    }
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  const toggleTheme = useCallback(() => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'))
  }, [])

  return { theme, toggleTheme, isDark: theme === 'dark' }
}
