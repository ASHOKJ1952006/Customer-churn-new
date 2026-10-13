import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer } from 'recharts'
import Tooltip from '../components/Tooltip'
import '../styles/theme.css'

const API_URL = 'http://localhost:8000'

function WhatIfSimulator({ dataset }) {
  const [formData, setFormData] = useState({
    dataset: dataset,
    tenure: 12,
    monthly_charges: 70.0,
    total_charges: 1000.0,
    contract: 'Month-to-month',
    payment_method: 'Electronic check',
    paperless_billing: 'Yes',
    phone_service: 'Yes',
    multiple_lines: 'No',
    internet_service: 'DSL',
    online_security: 'No',
    online_backup: 'No',
    device_protection: 'No',
    tech_support: 'No',
    streaming_tv: 'No',
    streaming_movies: 'No',
    gender: 'Male',
    partner: 'No',
    dependents: 'No',
    senior_citizen: 'No',
  })

  const [selectedFactor, setSelectedFactor] = useState('contract')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleSimulate = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`${API_URL}/api/whatif/simulate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          dataset: dataset,
          base_profile: formData,
          factor: selectedFactor
        }),
      })

      if (!response.ok) {
        throw new Error('Simulation failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const factorOptions = [
    { value: 'contract', label: 'Contract Type' },
    { value: 'payment_method', label: 'Payment Method' },
    { value: 'internet_service', label: 'Internet Service' },
    { value: 'phone_service', label: 'Phone Service' },
    { value: 'paperless_billing', label: 'Paperless Billing' },
  ]

  const chartData = result?.scenarios?.map(s => ({
    name: s.scenario_label.length > 15 ? s.scenario_label.substring(0, 15) + '...' : s.scenario_label,
    probability: (s.probability * 100).toFixed(1),
    value: s.probability * 100
  })) || []

  return (
    <div className="whatif-simulator-page">
      <div className="card card-royal mb-4">
        <h2>🔄 What-If Simulator</h2>
        <p>Explore how changing different factors affects churn risk.</p>
      </div>

      <div className="grid grid-3">
        {/* Account Details */}
        <div className="card">
          <h3>📋 Account Details</h3>
          <div className="form-group">
            <label>Tenure (months)</label>
            <input
              type="number"
              name="tenure"
              value={formData.tenure}
              onChange={handleChange}
              min="0"
              max="72"
            />
          </div>
          <div className="form-group">
            <label>Monthly Charges ($)</label>
            <input
              type="number"
              name="monthly_charges"
              value={formData.monthly_charges}
              onChange={handleChange}
              min="18"
              max="120"
              step="0.1"
            />
          </div>
          <div className="form-group">
            <label>Total Charges ($)</label>
            <input
              type="number"
              name="total_charges"
              value={formData.total_charges}
              onChange={handleChange}
              min="0"
              max="10000"
              step="0.1"
            />
          </div>
          <div className="form-group">
            <label>Contract Type</label>
            <select name="contract" value={formData.contract} onChange={handleChange}>
              <option value="Month-to-month">Month-to-month</option>
              <option value="One year">One year</option>
              <option value="Two year">Two year</option>
            </select>
          </div>
          <div className="form-group">
            <label>Payment Method</label>
            <select name="payment_method" value={formData.payment_method} onChange={handleChange}>
              <option value="Electronic check">Electronic check</option>
              <option value="Mailed check">Mailed check</option>
              <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
              <option value="Credit card (automatic)">Credit card (automatic)</option>
            </select>
          </div>
          <div className="form-group">
            <label>Paperless Billing</label>
            <select name="paperless_billing" value={formData.paperless_billing} onChange={handleChange}>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>
          </div>
        </div>

        {/* Services */}
        <div className="card">
          <h3>🔧 Services</h3>
          <div className="form-group">
            <label>Phone Service</label>
            <select name="phone_service" value={formData.phone_service} onChange={handleChange}>
              <option value="Yes">Yes</option>
              <option value="No">No</option>
            </select>
          </div>
          <div className="form-group">
            <label>Multiple Lines</label>
            <select name="multiple_lines" value={formData.multiple_lines} onChange={handleChange}>
              <option value="No">No</option>
              <option value="Yes">Yes</option>
              <option value="No phone service">No phone service</option>
            </select>
          </div>
          <div className="form-group">
            <label>Internet Service</label>
            <select name="internet_service" value={formData.internet_service} onChange={handleChange}>
              <option value="DSL">DSL</option>
              <option value="Fiber optic">Fiber optic</option>
              <option value="No">No</option>
            </select>
          </div>
          <div className="form-group">
            <label>Online Security</label>
            <select name="online_security" value={formData.online_security} onChange={handleChange}>
              <option value="No">No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
          <div className="form-group">
            <label>Online Backup</label>
            <select name="online_backup" value={formData.online_backup} onChange={handleChange}>
              <option value="No">No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
          <div className="form-group">
            <label>Device Protection</label>
            <select name="device_protection" value={formData.device_protection} onChange={handleChange}>
              <option value="No">No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
        </div>

        {/* Demographics & Factor Selection */}
        <div className="card">
          <h3>👥 Demographics</h3>
          <div className="form-group">
            <label>Tech Support</label>
            <select name="tech_support" value={formData.tech_support} onChange={handleChange}>
              <option value="No">No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
          <div className="form-group">
            <label>Streaming TV</label>
            <select name="streaming_tv" value={formData.streaming_tv} onChange={handleChange}>
              <option value="No">No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
          <div className="form-group">
            <label>Streaming Movies</label>
            <select name="streaming_movies" value={formData.streaming_movies} onChange={handleChange}>
              <option value="No">No</option>
              <option value="Yes">Yes</option>
              <option value="No internet service">No internet service</option>
            </select>
          </div>
          <div className="form-group">
            <label>Gender</label>
            <select name="gender" value={formData.gender} onChange={handleChange}>
              <option value="Male">Male</option>
              <option value="Female">Female</option>
            </select>
          </div>
          <div className="form-group">
            <label>Has Partner</label>
            <select name="partner" value={formData.partner} onChange={handleChange}>
              <option value="No">No</option>
              <option value="Yes">Yes</option>
            </select>
          </div>
          <div className="form-group">
            <label>Has Dependents</label>
            <select name="dependents" value={formData.dependents} onChange={handleChange}>
              <option value="No">No</option>
              <option value="Yes">Yes</option>
            </select>
          </div>
          <div className="form-group">
            <label>Senior Citizen</label>
            <select name="senior_citizen" value={formData.senior_citizen} onChange={handleChange}>
              <option value="No">No</option>
              <option value="Yes">Yes</option>
            </select>
          </div>
        </div>
      </div>

      <div className="card mt-4">
        <h3>🎛️ Factor to Simulate</h3>
        <div className="form-group">
          <label>Select Factor</label>
          <select 
            value={selectedFactor} 
            onChange={(e) => setSelectedFactor(e.target.value)}
          >
            {factorOptions.map(option => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>
        </div>
        <button 
          className="btn btn-royal" 
          onClick={handleSimulate}
          disabled={loading}
          style={{ width: '100%', padding: '1rem', fontSize: '1.125rem' }}
        >
          {loading ? 'Simulating...' : '🔮 Run Simulation'}
        </button>
      </div>

      {error && (
        <div className="card mt-4" style={{ background: 'var(--risk-high-bg)', border: '2px solid var(--risk-high)' }}>
          <p style={{ color: 'var(--risk-high)', fontWeight: '600' }}>❌ Error: {error}</p>
        </div>
      )}

      {result && (
        <div className="card card-royal mt-4">
          <h3>📊 Simulation Results</h3>
          
          <div className="result-card mt-4">
            <span className="result-label">
              <Tooltip content="The model's estimate of how likely this customer is to cancel their service. This number has been calibrated, meaning a 70% prediction should actually match reality about 70% of the time — not just a rough ranking score.">
                Baseline Probability
              </Tooltip>
            </span>
            <p className="result-value">{(result.baseline_probability * 100).toFixed(1)}%</p>
          </div>

          <div className="mt-4">
            <h4>📈 Probability Across Scenarios</h4>
            <div style={{ height: '300px', marginTop: '1rem' }}>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                  <YAxis />
                  <RechartsTooltip formatter={(value) => `${value}%`} />
                  <Bar dataKey="value" fill="var(--color-accent)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="mt-4">
            <h4>🏆 Best Option</h4>
            <div className="card" style={{ background: 'var(--color-gray-50)', border: '2px solid var(--color-gold)' }}>
              <p style={{ fontSize: '1.125rem', fontWeight: '600' }}>
                {result.best_option} - {(result.best_probability * 100).toFixed(1)}% churn risk
              </p>
            </div>
          </div>

          <div className="mt-4">
            <h4>💡 Recommended Retention Strategy</h4>
            <div className="card mb-3" style={{ background: 'var(--color-gray-50)', border: '2px solid var(--color-accent)' }}>
              <p style={{ fontSize: '1.05rem', lineHeight: '1.6', margin: 0 }}>{result.recommendation}</p>
            </div>
            {result.recommendations && result.recommendations.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', marginTop: '0.75rem' }}>
                {result.recommendations.map((rec, i) => (
                  <div key={i} className="card" style={{
                    padding: '0.85rem 1.15rem',
                    borderLeft: `4px solid ${rec.priority === 'HIGH' ? '#ef4444' : rec.priority === 'MEDIUM' ? '#f59e0b' : '#3b82f6'}`,
                    background: '#f8fafc'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
                      <strong style={{ fontSize: '0.9rem', color: '#1e293b' }}>{rec.factor}</strong>
                      <span style={{
                        fontSize: '0.7rem',
                        fontWeight: '700',
                        padding: '0.15rem 0.5rem',
                        borderRadius: '9999px',
                        background: rec.priority === 'HIGH' ? '#fee2e2' : rec.priority === 'MEDIUM' ? '#fef3c7' : '#dbeafe',
                        color: rec.priority === 'HIGH' ? '#b91c1c' : rec.priority === 'MEDIUM' ? '#b45309' : '#1d4ed8'
                      }}>
                        {rec.priority}
                      </span>
                    </div>
                    <p style={{ margin: '0.2rem 0', fontSize: '0.95rem', color: '#334155' }}>👉 {rec.action}</p>
                    <p style={{ margin: 0, fontSize: '0.8rem', color: '#64748b' }}><em>{rec.reason}</em></p>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default WhatIfSimulator
