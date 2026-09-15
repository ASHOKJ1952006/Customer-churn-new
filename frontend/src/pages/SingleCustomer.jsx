import { useState } from 'react'
import { BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import Tooltip from '../components/Tooltip'
import '../styles/theme.css'

const API_URL = 'http://localhost:8000'

function SingleCustomer({ dataset }) {
  const [formData, setFormData] = useState({
    dataset: dataset,
    tenure: 12,
    monthly_charges: 70.0,
    total_charges: 1000.0,
    contract: 'Month-to-month',
    payment_method: 'Electronic check',
    internet_service: 'DSL',
  })

  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false)
  const [activeSimulation, setActiveSimulation] = useState(null)
  const [simulatingActionId, setSimulatingActionId] = useState(null)

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    setActiveSimulation(null)

    try {
      const response = await fetch(`${API_URL}/api/predict/single`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      })

      if (!response.ok) {
        throw new Error('Prediction failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleSimulateAction = async (rec) => {
    if (!rec.simulation) return
    setSimulatingActionId(rec.action_id)
    // Small timeout for visual confirmation of simulation trigger
    setTimeout(() => {
      setActiveSimulation({
        title: rec.title,
        ...rec.simulation
      })
      setSimulatingActionId(null)
    }, 200)
  }

  const getRiskBadgeClass = (riskLevel) => {
    switch (riskLevel) {
      case 'Critical': return 'badge-high'
      case 'High': return 'badge-high'
      case 'Medium': return 'badge-medium'
      case 'Low': return 'badge-low'
      default: return ''
    }
  }

  const getPriorityBadgeStyle = (priority) => {
    switch (priority) {
      case 'CRITICAL':
        return { background: '#fee2e2', color: '#991b1b', border: '1px solid #f87171' }
      case 'HIGH':
        return { background: '#fee2e2', color: '#b91c1c', border: '1px solid #fca5a5' }
      case 'MEDIUM':
        return { background: '#fef3c7', color: '#b45309', border: '1px solid #fcd34d' }
      case 'LOW':
      default:
        return { background: '#dbeafe', color: '#1d4ed8', border: '1px solid #93c5fd' }
    }
  }

  const getCostBadgeStyle = (cost) => {
    switch (cost) {
      case 'HIGH':
        return { background: '#fef2f2', color: '#b91c1c' }
      case 'MEDIUM':
        return { background: '#fffbeb', color: '#b45309' }
      case 'LOW':
      default:
        return { background: '#ecfdf5', color: '#047857' }
    }
  }

  const getImpactBadgeStyle = (impact) => {
    switch (impact) {
      case 'HIGH':
        return { background: '#ede9fe', color: '#6d28d9', fontWeight: '700' }
      case 'MEDIUM':
        return { background: '#e0f2fe', color: '#0369a1', fontWeight: '600' }
      case 'LOW':
      default:
        return { background: '#f1f5f9', color: '#475569', fontWeight: '500' }
    }
  }

  // Extract top 10 features for horizontal SHAP bar chart
  const rawFeatures = result?.shap?.features || []
  const chartData = rawFeatures
    .slice(0, 10)
    .map(f => {
      const shapVal = Number(f.shap_value || 0)
      const displayName = f.display_name || f.feature || 'Factor'
      return {
        name: displayName.length > 24 ? displayName.substring(0, 24) + '...' : displayName,
        full_name: displayName,
        shap_value: shapVal,
        actual_value: f.value !== undefined ? String(f.value) : 'N/A',
        direction: f.direction || (shapVal > 0 ? 'increases_churn' : 'decreases_churn'),
        impact: f.impact || (shapVal > 0 ? 'risk' : 'protective')
      }
    })
    .reverse()

  const CustomSHAPTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      const isRisk = data.shap_value > 0
      return (
        <div style={{
          background: '#ffffff',
          padding: '0.75rem 1rem',
          borderRadius: '8px',
          border: `1px solid ${isRisk ? '#fca5a5' : '#86efac'}`,
          boxShadow: '0 4px 12px rgba(0,0,0,0.12)',
          fontSize: '0.875rem'
        }}>
          <p style={{ fontWeight: '700', marginBottom: '0.25rem', color: '#1e293b' }}>{data.full_name}</p>
          <p style={{ margin: '0.2rem 0', color: '#475569' }}><strong>Actual Value:</strong> {data.actual_value}</p>
          <p style={{ margin: '0.2rem 0', color: isRisk ? '#dc2626' : '#16a34a', fontWeight: '700' }}>
            <strong>SHAP Value:</strong> {data.shap_value > 0 ? `+${data.shap_value.toFixed(4)}` : data.shap_value.toFixed(4)}
          </p>
          <p style={{ margin: '0.2rem 0', color: isRisk ? '#dc2626' : '#16a34a', fontWeight: '600' }}>
            <strong>Impact:</strong> {isRisk ? 'Increases Churn Risk ⚠️' : 'Decreases Churn Risk (Protective) 🛡️'}
          </p>
        </div>
      )
    }
    return null
  }

  return (
    <div className="single-customer-page">
      {/* Header Banner */}
      <div className="card card-royal mb-4">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h2 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span>🎯</span> Telecom Customer Retention Decision Support System
            </h2>
            <p style={{ margin: '0.5rem 0 0 0', color: 'var(--color-gray-600)' }}>
              Translate AI predictions and SHAP/LIME drivers into actionable retention interventions, strategy bundles, and counterfactual simulations.
            </p>
          </div>
          { /*<span style={{
            background: 'var(--color-primary-light)',
            color: 'white',
            padding: '0.35rem 0.85rem',
            borderRadius: '20px',
            fontSize: '0.8rem',
            fontWeight: '600'
          }}>
            Dataset: {dataset === 'ds1' ? 'Telco Enterprise' : 'Telecom Optimized'}
          </span>*/}
        </div>
      </div>

      {/* Input Form Grid */}
      <div className="grid grid-2" style={{ gap: '1.5rem' }}>
        {/* Account Details */}
        <div className="card" style={{ height: 'fit-content' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            marginBottom: '1.5rem',
            paddingBottom: '1rem',
            borderBottom: '2px solid var(--color-gray-200)'
          }}>
            <span style={{ fontSize: '1.5rem' }}>📋</span>
            <h3 style={{ margin: 0 }}>Account Information</h3>
          </div>

          <div className="form-group">
            <label>
              Tenure (months)
              <Tooltip content="Customer relationship duration. Early tenure (<12 mo) exhibits higher vulnerability; mature accounts build habitual stickiness.">
                <span style={{ marginLeft: '0.25rem', color: 'var(--color-gray-400)', cursor: 'help' }}>ⓘ</span>
              </Tooltip>
            </label>
            <input
              type="number"
              name="tenure"
              value={formData.tenure}
              onChange={handleChange}
              min="0"
              max="72"
              placeholder="e.g., 12"
            />
          </div>

          <div className="form-group">
            <label>
              Contract Term
              <Tooltip content="Month-to-month contracts have high voluntary churn ease. 1-year and 2-year terms provide commitment lock-in.">
                <span style={{ marginLeft: '0.25rem', color: 'var(--color-gray-400)', cursor: 'help' }}>ⓘ</span>
              </Tooltip>
            </label>
            <select name="contract" value={formData.contract} onChange={handleChange}>
              <option value="Month-to-month">Month-to-month</option>
              <option value="One year">One year</option>
              <option value="Two year">Two year</option>
            </select>
          </div>

          <div className="form-group">
            <label>
              Internet Service Tier
              <Tooltip content="Fiber optic subscribers experience high bandwidth expectations and competitive market deals.">
                <span style={{ marginLeft: '0.25rem', color: 'var(--color-gray-400)', cursor: 'help' }}>ⓘ</span>
              </Tooltip>
            </label>
            <select name="internet_service" value={formData.internet_service} onChange={handleChange}>
              <option value="DSL">DSL Internet</option>
              <option value="Fiber optic">Fiber Optic</option>
              <option value="No">No Internet Service</option>
            </select>
          </div>
        </div>

        {/* Billing Information */}
        <div className="card" style={{ height: 'fit-content' }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            marginBottom: '1.5rem',
            paddingBottom: '1rem',
            borderBottom: '2px solid var(--color-gray-200)'
          }}>
            <span style={{ fontSize: '1.5rem' }}>💳</span>
            <h3 style={{ margin: 0 }}>Billing & Payment</h3>
          </div>

          <div className="form-group">
            <label>
              Monthly Charges ($)
              <Tooltip content="Current monthly invoice. High recurring costs create acute price sensitivity.">
                <span style={{ marginLeft: '0.25rem', color: 'var(--color-gray-400)', cursor: 'help' }}>ⓘ</span>
              </Tooltip>
            </label>
            <input
              type="number"
              name="monthly_charges"
              value={formData.monthly_charges}
              onChange={handleChange}
              min="18"
              max="120"
              step="0.1"
              placeholder="e.g., 70.00"
            />
          </div>

          <div className="form-group">
            <label>
              Total Charges ($)
              <Tooltip content="Cumulative lifetime billing. Used to measure spending trajectory and sunk investment.">
                <span style={{ marginLeft: '0.25rem', color: 'var(--color-gray-400)', cursor: 'help' }}>ⓘ</span>
              </Tooltip>
            </label>
            <input
              type="number"
              name="total_charges"
              value={formData.total_charges}
              onChange={handleChange}
              min="0"
              max="10000"
              step="0.1"
              placeholder="e.g., 1000.00"
            />
          </div>

          <div className="form-group">
            <label>
              Payment Method
              <Tooltip content="Electronic check exhibits high manual payment friction and missed cycles. Autopay provides smooth renewal continuity.">
                <span style={{ marginLeft: '0.25rem', color: 'var(--color-gray-400)', cursor: 'help' }}>ⓘ</span>
              </Tooltip>
            </label>
            <select name="payment_method" value={formData.payment_method} onChange={handleChange}>
              <option value="Electronic check">Electronic check</option>
              <option value="Mailed check">Mailed check</option>
              <option value="Bank transfer (automatic)">Bank transfer (automatic)</option>
              <option value="Credit card (automatic)">Credit card (automatic)</option>
            </select>
          </div>
        </div>
      </div>

      <div className="mt-4" style={{ textAlign: 'center' }}>
        <button
          className="btn btn-royal"
          onClick={handleSubmit}
          disabled={loading}
          style={{
            padding: '1rem 3rem',
            fontSize: '1.125rem',
            minWidth: '280px',
            boxShadow: '0 4px 15px rgba(37, 99, 235, 0.25)'
          }}
        >
          {loading ? (
            <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
              <span className="spinner" style={{
                width: '18px',
                height: '18px',
                border: '2px solid white',
                borderTop: '2px solid transparent',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }}></span>
              Evaluating Churn Drivers & Solutions...
            </span>
          ) : '🚀 Generate Retention Decision Intelligence'}
        </button>
      </div>

      {error && (
        <div className="card mt-4" style={{ background: 'var(--risk-high-bg)', border: '2px solid var(--risk-high)' }}>
          <p style={{ color: 'var(--risk-high)', fontWeight: '600' }}>❌ Evaluation Error: {error}</p>
        </div>
      )}

      {/* =========================================================================
          RESULT SECTION: BUSINESS EXPLANATION LAYER (APPEARS FIRST)
          ========================================================================= */}
      {result && (
        <div className="mt-4" style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

          {/* 1 & 2. Top-Level Churn Metrics Bar */}
          <div className="grid grid-3" style={{ gap: '1rem' }}>
            <div className="result-card" style={{ background: '#ffffff', borderLeft: '5px solid var(--color-primary)' }}>
              <p className="result-label">
                <Tooltip content="Calibrated model prediction representing true statistical probability of cancellation.">
                  Predicted Churn Probability
                </Tooltip>
              </p>
              <p className="result-value" style={{
                color: result.probability >= 0.70 ? '#dc2626' : result.probability >= result.threshold ? '#d97706' : '#16a34a'
              }}>
                {(result.probability * 100).toFixed(1)}%
              </p>
              <span style={{ fontSize: '0.8rem', color: '#64748b' }}>Decision Threshold: {result.threshold.toFixed(3)}</span>
            </div>

            <div className="result-card" style={{ background: '#ffffff', borderLeft: '5px solid #8b5cf6' }}>
              <p className="result-label">Account Risk Classification</p>
              <div style={{ marginTop: '0.35rem' }}>
                <span className={`badge ${getRiskBadgeClass(result.risk_level)}`} style={{ fontSize: '1rem', padding: '0.35rem 1rem' }}>
                  {result.risk_level.toUpperCase()} RISK
                </span>
              </div>
              <span style={{ fontSize: '0.8rem', color: '#64748b', display: 'block', marginTop: '0.4rem' }}>
                {result.probability >= result.threshold ? 'Flags immediate retention outreach' : 'Normal lifecycle monitoring'}
              </span>
            </div>

            <div className="result-card" style={{ background: '#ffffff', borderLeft: '5px solid #059669' }}>
              <p className="result-label">
                <Tooltip content="Concordance rate between SHAP and LIME explanations. High confidence confirms both methodologies agree on risk drivers.">
                  Explanation Confidence
                </Tooltip>
              </p>
              <div style={{ marginTop: '0.35rem' }}>
                <span style={{
                  padding: '0.3rem 0.8rem',
                  borderRadius: '9999px',
                  fontSize: '0.9rem',
                  fontWeight: '700',
                  background: result.explanation_confidence === 'HIGH' ? '#ecfdf5' : result.explanation_confidence === 'MEDIUM' ? '#fffbeb' : '#fef2f2',
                  color: result.explanation_confidence === 'HIGH' ? '#047857' : result.explanation_confidence === 'MEDIUM' ? '#b45309' : '#b91c1c',
                  border: `1px solid ${result.explanation_confidence === 'HIGH' ? '#86efac' : result.explanation_confidence === 'MEDIUM' ? '#fde68a' : '#fca5a5'}`
                }}>
                  {result.explanation_confidence} CONFIDENCE
                </span>
              </div>
              <span style={{ fontSize: '0.8rem', color: '#64748b', display: 'block', marginTop: '0.4rem' }}>
                SHAP & LIME Directional Agreement
              </span>
            </div>
          </div>

          {/* 3. Executive Telecom Retention Decision Summary (10-Second Management Card) */}
          <div className="card card-royal" style={{ border: '2px solid var(--color-primary-light)', background: 'linear-gradient(to right, #ffffff, #f8fafc)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid #e2e8f0', paddingBottom: '0.75rem', marginBottom: '1rem' }}>
              <h3 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#1e293b' }}>
                <span>👔</span> Executive Telecom Management Summary
              </h3>
              <span style={{ fontSize: '0.8rem', fontWeight: '700', color: '#64748b', textTransform: 'uppercase' }}>
                10-Second Decision Overview
              </span>
            </div>

            <div className="grid grid-3" style={{ gap: '1.25rem', marginBottom: '1rem' }}>
              <div style={{ background: '#f8fafc', padding: '0.85rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <p style={{ margin: 0, fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', fontWeight: '700' }}>Primary Churn Driver</p>
                <p style={{ margin: '0.35rem 0 0 0', fontWeight: '700', color: '#991b1b', fontSize: '1rem' }}>
                  {result.management_summary?.primary_churn_driver || 'Account Term Structure'}
                </p>
              </div>

              <div style={{ background: '#f8fafc', padding: '0.85rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <p style={{ margin: 0, fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', fontWeight: '700' }}>Recommended Strategy</p>
                <p style={{ margin: '0.35rem 0 0 0', fontWeight: '700', color: '#1e40af', fontSize: '1rem' }}>
                  {result.management_summary?.recommended_strategy || 'Contract Upgrade + Autopay'}
                </p>
              </div>

              <div style={{ background: '#f8fafc', padding: '0.85rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                <p style={{ margin: 0, fontSize: '0.75rem', color: '#64748b', textTransform: 'uppercase', fontWeight: '700' }}>Retention Priority & Cost</p>
                <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.35rem' }}>
                  <span style={{
                    ...getPriorityBadgeStyle(result.management_summary?.priority),
                    padding: '0.2rem 0.6rem',
                    borderRadius: '4px',
                    fontSize: '0.8rem',
                    fontWeight: '700'
                  }}>
                    {result.management_summary?.priority}
                  </span>
                  <span style={{
                    ...getCostBadgeStyle(result.management_summary?.estimated_action_cost),
                    padding: '0.2rem 0.6rem',
                    borderRadius: '4px',
                    fontSize: '0.8rem',
                    fontWeight: '700'
                  }}>
                    {result.management_summary?.estimated_action_cost} COST
                  </span>
                </div>
              </div>
            </div>

            {/* Dynamic Natural Language Narrative */}
            <div style={{ background: '#f1f5f9', padding: '1rem', borderRadius: '8px', borderLeft: '4px solid var(--color-primary)' }}>
              <p style={{ margin: 0, fontSize: '0.95rem', lineHeight: '1.6', color: '#334155' }}>
                {result.customer_summary}
              </p>
            </div>
          </div>

          {/* 4. Top 5 Major Churn Drivers (Business Aggregated) */}
          <div className="card" style={{ background: '#ffffff' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
              <h4 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#991b1b' }}>
                <span>🚨</span> Why is This Customer at Risk? (Top Major Churn Drivers)
              </h4>
              <span style={{ fontSize: '0.8rem', color: '#64748b' }}>Combined SHAP + LIME Evidence</span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              {result.major_churn_drivers && result.major_churn_drivers.length > 0 ? (
                result.major_churn_drivers.map((driver, idx) => (
                  <div key={idx} style={{
                    display: 'flex',
                    flexDirection: 'column',
                    padding: '0.9rem 1.15rem',
                    borderRadius: '8px',
                    background: '#fff5f5',
                    borderLeft: '4px solid #ef4444',
                    border: '1px solid #fee2e2'
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                        <span style={{
                          width: '24px',
                          height: '24px',
                          borderRadius: '50%',
                          background: '#ef4444',
                          color: 'white',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          fontSize: '0.75rem',
                          fontWeight: '700'
                        }}>
                          {idx + 1}
                        </span>
                        <strong style={{ fontSize: '1rem', color: '#1e293b' }}>{driver.factor}</strong>
                        {driver.actual_value !== 'N/A' && (
                          <span style={{ fontSize: '0.85rem', color: '#64748b', background: '#ffffff', padding: '0.15rem 0.5rem', borderRadius: '4px', border: '1px solid #e2e8f0' }}>
                            Value: {driver.actual_value}
                          </span>
                        )}
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{
                          fontSize: '0.75rem',
                          fontWeight: '700',
                          padding: '0.2rem 0.5rem',
                          borderRadius: '4px',
                          background: driver.is_actionable ? '#dbeafe' : '#f1f5f9',
                          color: driver.is_actionable ? '#1e40af' : '#475569'
                        }}>
                          {driver.is_actionable ? '✅ Actionable' : 'ℹ️ Lifecycle / Non-Actionable'}
                        </span>
                        <span style={{
                          fontSize: '0.75rem',
                          fontWeight: '700',
                          padding: '0.2rem 0.5rem',
                          borderRadius: '4px',
                          background: '#fee2e2',
                          color: '#991b1b'
                        }}>
                          {driver.impact_level}
                        </span>
                      </div>
                    </div>

                    <p style={{ margin: '0.4rem 0 0 0', fontSize: '0.875rem', color: '#4b5563', lineHeight: '1.5' }}>
                      {driver.business_explanation}
                    </p>
                  </div>
                ))
              ) : (
                <p style={{ margin: 0, color: '#047857' }}>No significant churn risk drivers detected. Account profile is stable.</p>
              )}
            </div>
          </div>

          {/* 5. Top Retention Recommendations (Cards #1 to #10) */}
          <div className="card" style={{ background: '#ffffff' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div>
                <h4 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#1e293b' }}>
                  <span>💡</span> Recommended Retention Interventions
                </h4>
                <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.85rem', color: '#64748b' }}>
                  Ranked by transparent Action Score = (Risk Contribution × Churn Probability × Relevance × Impact) / Cost
                </p>
              </div>
              <span style={{ fontSize: '0.85rem', fontWeight: '600', color: 'var(--color-primary)' }}>
                {result.recommendations?.length || 0} Tailored Actions Identified
              </span>
            </div>

            {/* Active Inline Simulation Banner */}
            {activeSimulation && (
              <div style={{
                background: '#ecfdf5',
                border: '1px solid #10b981',
                borderRadius: '8px',
                padding: '0.85rem 1.25rem',
                margin: '1rem 0',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: '0.5rem'
              }}>
                <div>
                  <strong style={{ color: '#065f46', fontSize: '0.95rem' }}>
                    ⚡ Real ML Counterfactual Simulation: {activeSimulation.title}
                  </strong>
                  <p style={{ margin: '0.2rem 0 0 0', fontSize: '0.85rem', color: '#047857' }}>
                    Before: <strong>{(activeSimulation.before_probability * 100).toFixed(1)}%</strong> → Simulated After: <strong>{(activeSimulation.simulated_probability * 100).toFixed(1)}%</strong>
                  </p>
                </div>
                <div style={{
                  background: '#059669',
                  color: 'white',
                  fontWeight: '700',
                  padding: '0.35rem 0.85rem',
                  borderRadius: '9999px',
                  fontSize: '0.9rem'
                }}>
                  Risk Reduction: -{activeSimulation.risk_reduction_pct} pp
                </div>
              </div>
            )}

            {/* Recommendation Cards */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1rem' }}>
              {result.recommendations && result.recommendations.length > 0 ? (
                result.recommendations.map((rec, i) => (
                  <div key={i} style={{
                    padding: '1.25rem',
                    borderRadius: '10px',
                    border: '1px solid #e2e8f0',
                    borderLeft: `6px solid ${rec.priority === 'CRITICAL' ? '#dc2626' : rec.priority === 'HIGH' ? '#ef4444' : rec.priority === 'MEDIUM' ? '#f59e0b' : '#3b82f6'}`,
                    background: '#fafafa',
                    boxShadow: '0 2px 5px rgba(0,0,0,0.03)'
                  }}>
                    {/* Header Row */}
                    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.5rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                        <span style={{
                          fontWeight: '800',
                          fontSize: '1.05rem',
                          color: '#1e293b'
                        }}>
                          #{rec.rank}
                        </span>
                        <h4 style={{ margin: 0, fontSize: '1.05rem', color: '#1e293b' }}>{rec.title}</h4>
                        <span style={{
                          fontSize: '0.75rem',
                          color: '#64748b',
                          background: '#e2e8f0',
                          padding: '0.15rem 0.5rem',
                          borderRadius: '4px',
                          fontWeight: '600'
                        }}>
                          {rec.factor}
                        </span>
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{
                          ...getPriorityBadgeStyle(rec.priority),
                          fontSize: '0.75rem',
                          fontWeight: '800',
                          padding: '0.2rem 0.6rem',
                          borderRadius: '9999px'
                        }}>
                          {rec.priority} PRIORITY
                        </span>
                      </div>
                    </div>

                    {/* Why & Action Text */}
                    <p style={{ margin: '0.4rem 0', color: '#475569', fontSize: '0.9rem' }}>
                      <strong style={{ color: '#334155' }}>Why:</strong> {rec.why}
                    </p>
                    <p style={{ margin: '0.4rem 0', color: '#1e293b', fontSize: '0.95rem', fontWeight: '600' }}>
                      <strong style={{ color: '#2563eb' }}>Recommended Action:</strong> {rec.action}
                    </p>
                    <p style={{ margin: '0.4rem 0', color: '#64748b', fontSize: '0.85rem' }}>
                      <strong>Objective:</strong> {rec.business_objective}
                    </p>

                    {/* Metadata & Simulation Button Bar */}
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      flexWrap: 'wrap',
                      gap: '0.75rem',
                      marginTop: '0.85rem',
                      paddingTop: '0.75rem',
                      borderTop: '1px solid #e2e8f0'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap', fontSize: '0.8rem' }}>
                        <span>👤 <strong>Owner:</strong> {rec.owner}</span>
                        <span>⏱️ <strong>Timeframe:</strong> {rec.timeframe}</span>
                        <span style={{ ...getCostBadgeStyle(rec.cost), padding: '0.15rem 0.45rem', borderRadius: '4px', fontWeight: '700' }}>
                          Cost: {rec.cost}
                        </span>
                        <span style={{ ...getImpactBadgeStyle(rec.expected_impact), padding: '0.15rem 0.45rem', borderRadius: '4px' }}>
                          Impact: {rec.expected_impact}
                        </span>
                      </div>

                      {/* Interactive Simulation Button */}
                      {rec.simulation && (
                        <button
                          className="btn"
                          onClick={() => handleSimulateAction(rec)}
                          disabled={simulatingActionId === rec.action_id}
                          style={{
                            padding: '0.35rem 0.85rem',
                            fontSize: '0.8rem',
                            fontWeight: '700',
                            background: '#047857',
                            color: 'white',
                            borderRadius: '6px',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '0.35rem'
                          }}
                        >
                          <span>⚡</span>
                          {simulatingActionId === rec.action_id ? 'Simulating...' : `Simulate Impact (-${rec.simulation.risk_reduction_pct} pp)`}
                        </button>
                      )}
                    </div>
                  </div>
                ))
              ) : (
                <p>No immediate intervention required.</p>
              )}
            </div>
          </div>

          {/* 6. Retention Strategy Bundles (Strategy A, B, C) */}
          {result.strategy_bundles && result.strategy_bundles.length > 0 && result.probability >= result.threshold && (
            <div className="card" style={{ background: '#ffffff' }}>
              <h4 style={{ margin: '0 0 0.5rem 0', display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#1e293b' }}>
                <span>📦</span> Retention Strategy Bundles & Counterfactual ML Impact
              </h4>
              <p style={{ margin: '0 0 1rem 0', fontSize: '0.85rem', color: '#64748b' }}>
                Simulated by passing hypothetical retained customer attributes through the trained model pipeline:
              </p>

              <div className="grid grid-3" style={{ gap: '1rem' }}>
                {result.strategy_bundles.map((bundle, bIdx) => (
                  <div key={bIdx} style={{
                    padding: '1rem',
                    borderRadius: '8px',
                    border: '1px solid #e2e8f0',
                    background: bIdx === 1 ? '#f0fdf4' : '#fafafa',
                    boxShadow: '0 2px 4px rgba(0,0,0,0.04)',
                    display: 'flex',
                    flexDirection: 'column',
                    justifyContent: 'space-between'
                  }}>
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                        <strong style={{ color: '#1e293b', fontSize: '0.95rem' }}>{bundle.name}</strong>
                        <span style={{ ...getCostBadgeStyle(bundle.cost), padding: '0.15rem 0.5rem', borderRadius: '4px', fontSize: '0.75rem', fontWeight: '700' }}>
                          {bundle.cost} COST
                        </span>
                      </div>

                      <ul style={{ paddingLeft: '1.2rem', margin: '0.5rem 0', fontSize: '0.85rem', color: '#475569' }}>
                        {bundle.tactics.map((tac, tIdx) => (
                          <li key={tIdx} style={{ marginBottom: '0.25rem' }}>{tac}</li>
                        ))}
                      </ul>
                    </div>

                    <div style={{ marginTop: '0.75rem', paddingTop: '0.75rem', borderTop: '1px solid #e2e8f0' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#334155' }}>
                        <span>Before Churn:</span>
                        <strong>{(bundle.baseline_probability * 100).toFixed(1)}%</strong>
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', color: '#047857', marginTop: '0.2rem' }}>
                        <span>After Churn:</span>
                        <strong>{(bundle.simulated_probability * 100).toFixed(1)}%</strong>
                      </div>
                      <div style={{
                        marginTop: '0.5rem',
                        padding: '0.25rem 0.5rem',
                        borderRadius: '4px',
                        background: '#10b981',
                        color: 'white',
                        textAlign: 'center',
                        fontSize: '0.85rem',
                        fontWeight: '700'
                      }}>
                        Risk Reduction: -{bundle.risk_reduction_pct} pp
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* 7. Protective Factors Card */}
          <div className="card" style={{ borderLeft: '5px solid #10b981', background: '#f0fdf4' }}>
            <h4 style={{ color: '#065f46', marginTop: 0, marginBottom: '0.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <span>🛡️</span> What is Helping Retain This Customer? (Protective Factors)
            </h4>
            <p style={{ margin: '0 0 0.75rem 0', fontSize: '0.85rem', color: '#047857' }}>
              Attributes actively reducing churn probability and stabilizing the account relationship:
            </p>

            {result.protective_factors && result.protective_factors.length > 0 ? (
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '0.75rem' }}>
                {result.protective_factors.map((p, i) => (
                  <div key={i} style={{
                    background: '#ffffff',
                    padding: '0.65rem 0.85rem',
                    borderRadius: '6px',
                    border: '1px solid #bbf7d0'
                  }}>
                    <strong style={{ color: '#064e3b', fontSize: '0.9rem' }}>✓ {p.factor}</strong>
                    {p.actual_value !== 'N/A' && (
                      <span style={{ fontSize: '0.8rem', color: '#059669', marginLeft: '0.4rem' }}>({p.actual_value})</span>
                    )}
                    <p style={{ margin: '0.25rem 0 0 0', fontSize: '0.8rem', color: '#475569' }}>
                      {p.explanation}
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ color: '#064e3b', margin: 0, fontSize: '0.9rem' }}>No significant protective factors detected.</p>
            )}
          </div>

          {/* 8. SHAP + LIME Agreement Card */}
          <div className="card" style={{ background: '#ffffff', border: '1px solid #e2e8f0' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '0.5rem' }}>
              <h4 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#1e293b' }}>
                <span>⚖️</span> Dual Explainability Concordance (SHAP + LIME Agreement)
              </h4>
              <span style={{
                fontSize: '0.8rem',
                fontWeight: '700',
                padding: '0.2rem 0.6rem',
                borderRadius: '9999px',
                background: result.explanation_confidence === 'HIGH' ? '#ecfdf5' : '#fffbeb',
                color: result.explanation_confidence === 'HIGH' ? '#047857' : '#b45309'
              }}>
                {result.explanation_confidence} CONFIDENCE
              </span>
            </div>
            <p style={{ margin: 0, fontSize: '0.9rem', color: '#475569' }}>
              {result.agreement_summary || 'Both SHAP and LIME algorithms converge on the core churn drivers for this account.'}
            </p>
          </div>

          {/* =========================================================================
              9. TECHNICAL MODEL EXPLANATION (COLLAPSIBLE ACCORDION)
              ========================================================================= */}
          <div className="card" style={{ background: '#f8fafc', border: '1px solid #cbd5e1' }}>
            <div
              onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                cursor: 'pointer',
                userSelect: 'none'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '1.25rem' }}>⚙️</span>
                <h4 style={{ margin: 0, color: '#334155' }}>
                  Technical Model Explanation (Data Science & Audit Layer)
                </h4>
              </div>
              <button
                className="btn btn-secondary"
                style={{ padding: '0.35rem 0.75rem', fontSize: '0.85rem' }}
              >
                {showTechnicalDetails ? '▲ Hide Technical Details' : '▼ Expand Technical Details (SHAP / LIME)'}
              </button>
            </div>

            {showTechnicalDetails && (
              <div className="mt-4" style={{ borderTop: '1px solid #e2e8f0', paddingTop: '1rem' }}>
                <p style={{ fontSize: '0.85rem', color: '#64748b', marginBottom: '1rem' }}>
                  Detailed feature contribution charts for machine learning evaluators, model risk governance, and regulatory compliance auditing.
                </p>

                {/* SHAP Feature Contribution Chart */}
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <h5 style={{ margin: 0, fontSize: '0.95rem', color: '#1e293b' }}>
                      📊 SHAP Feature Attribution (Tree / Kernel Explainer)
                    </h5>
                    <div style={{ display: 'flex', gap: '1rem', fontSize: '0.8rem' }}>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                        <span style={{ width: '12px', height: '12px', borderRadius: '3px', background: '#ef4444' }}></span>
                        Increases Churn Risk
                      </span>
                      <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                        <span style={{ width: '12px', height: '12px', borderRadius: '3px', background: '#10b981' }}></span>
                        Decreases Churn Risk (Protective)
                      </span>
                    </div>
                  </div>

                  {chartData.length > 0 ? (
                    <div style={{ height: '340px', marginTop: '1rem' }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData} layout="vertical" margin={{ top: 10, right: 30, left: 40, bottom: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                          <XAxis type="number" tickFormatter={(val) => val.toFixed(2)} />
                          <YAxis type="category" dataKey="name" width={170} tick={{ fontSize: 12 }} />
                          <RechartsTooltip content={<CustomSHAPTooltip />} />
                          <ReferenceLine x={0} stroke="#94a3b8" />
                          <Bar dataKey="shap_value" radius={[4, 4, 4, 4]}>
                            {chartData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.shap_value >= 0 ? '#ef4444' : '#10b981'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  ) : (
                    <p>SHAP explanation unavailable.</p>
                  )}
                </div>

                {/* LIME Tabular Weights Table */}
                {result.lime?.features && result.lime.features.length > 0 && (
                  <div className="mt-4">
                    <h5 style={{ margin: '0 0 0.5rem 0', fontSize: '0.95rem', color: '#1e293b' }}>
                      🔍 LIME Local Linear Surrogate Contributions
                    </h5>
                    <div className="table-container" style={{ maxHeight: '250px', overflowY: 'auto' }}>
                      <table>
                        <thead>
                          <tr>
                            <th>Feature Name</th>
                            <th>Business Domain</th>
                            <th>LIME Weight</th>
                            <th>Direction</th>
                          </tr>
                        </thead>
                        <tbody>
                          {result.lime.features.slice(0, 8).map((lf, lIdx) => (
                            <tr key={lIdx}>
                              <td><strong>{lf.display_name}</strong></td>
                              <td>{lf.category}</td>
                              <td style={{ color: lf.lime_weight > 0 ? '#dc2626' : '#16a34a', fontWeight: '700' }}>
                                {lf.lime_weight > 0 ? `+${lf.lime_weight.toFixed(4)}` : lf.lime_weight.toFixed(4)}
                              </td>
                              <td>
                                {lf.lime_weight > 0 ? '⚠️ Increases Churn' : '🛡️ Protective'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Time to Churn Footer */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.85rem', color: '#64748b', padding: '0.5rem 0' }}>
            <span>
              Expected Time-to-Churn (Cox Proportional Hazards): <strong>{result.survival_months?.toFixed(1) || 12.0} months</strong>
            </span>
            <span>
              Model Calibrated Decision Threshold: <strong>{result.threshold.toFixed(4)}</strong>
            </span>
          </div>

        </div>
      )}
    </div>
  )
}

export default SingleCustomer
