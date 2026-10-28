import { Routes, Route, NavLink } from 'react-router-dom'
import InputPage     from './pages/InputPage'
import ResultsPage   from './pages/ResultsPage'
import BenchmarkPage from './pages/BenchmarkPage'
import HistoryPage   from './pages/HistoryPage'
import AboutPage     from './pages/AboutPage'

const navClass = ({ isActive }) =>
  `px-3 py-2 rounded-lg text-sm font-medium transition ${
    isActive ? 'bg-rangeiq-600 text-white' : 'text-slate-300 hover:bg-slate-800'
  }`

export default function App() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-4 flex flex-wrap items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold text-rangeiq-500">RangeIQ</h1>
            <p className="text-xs text-slate-400">Honest EV range — not manufacturer lab claims</p>
          </div>
          <nav className="flex flex-wrap gap-1">
            <NavLink to="/"          end className={navClass}>Find EVs</NavLink>
            <NavLink to="/results"       className={navClass}>Results</NavLink>
            <NavLink to="/benchmark"     className={navClass}>Benchmark</NavLink>
            <NavLink to="/history"       className={navClass}>History</NavLink>
            <NavLink to="/about"         className={navClass}>Methodology</NavLink>
          </nav>
        </div>
      </header>

      <main className="max-w-6xl mx-auto px-4 py-8">
        <Routes>
          <Route path="/"          element={<InputPage />} />
          <Route path="/results"   element={<ResultsPage />} />
          <Route path="/benchmark" element={<BenchmarkPage />} />
          <Route path="/history"   element={<HistoryPage />} />
          <Route path="/about"     element={<AboutPage />} />
        </Routes>
      </main>
    </div>
  )
}
