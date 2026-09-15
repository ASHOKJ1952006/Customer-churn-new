import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import './styles/theme.css'
import Landing from './pages/Landing'
import SingleCustomer from './pages/SingleCustomer'
import BatchUpload from './pages/BatchUpload'
import WhatIfSimulator from './pages/WhatIfSimulator'
import Dashboard from './pages/Dashboard'
import ModelComparison from './pages/ModelComparison'

function AppContent() {
  const [dataset, setDataset] = useState('ds1')
  const location = useLocation()
  const isLanding = location.pathname === '/'

  return (
    <div className="app">
      {!isLanding && (
        <header className="app-header">
          <div className="container">
            <div className="flex-between">
              <Link to="/" className="app-title-link">
                <h1>👑 Churn Predictor</h1>
              </Link>
            </div>
          </div>
        </header>
      )}

      {!isLanding && (
        <nav className="app-nav">
          <div className="container">
            <div className="flex">
              <Link to="/single" className={`nav-btn ${location.pathname === '/single' ? 'active' : ''}`}>
                👤 Single Customer
              </Link>
              <Link to="/batch" className={`nav-btn ${location.pathname === '/batch' ? 'active' : ''}`}>
                📋 Batch Upload
              </Link>
              <Link to="/whatif" className={`nav-btn ${location.pathname === '/whatif' ? 'active' : ''}`}>
                🔄 What-If Simulator
              </Link>
              <Link to="/dashboard" className={`nav-btn ${location.pathname === '/dashboard' ? 'active' : ''}`}>
                📊 Dashboard
              </Link>
              <Link to="/model-comparison" className={`nav-btn ${location.pathname === '/model-comparison' ? 'active' : ''}`}>
                🏆 Model Comparison
              </Link>
            </div>
          </div>
        </nav>
      )}

      <main className="app-main">
        <div className="container">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/single" element={<SingleCustomer dataset={dataset} />} />
            <Route path="/batch" element={<BatchUpload dataset={dataset} />} />
            <Route path="/whatif" element={<WhatIfSimulator dataset={dataset} />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/model-comparison" element={<ModelComparison />} />
          </Routes>
        </div>
      </main>

      {!isLanding && (
        <footer className="app-footer">
          <div className="container">
            <p>© 2024 Churn Predictor | Powered by Advanced ML</p>
          </div>
        </footer>
      )}
    </div>
  )
}

function App() {
  return (
    <Router>
      <AppContent />
    </Router>
  )
}

export default App
