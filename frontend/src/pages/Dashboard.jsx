import React, { useState, useEffect } from 'react'
import Tooltip from '../components/Tooltip'
import '../styles/theme.css'

const API_URL = 'http://localhost:8000'

function Dashboard() {
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expandedHistoryId, setExpandedHistoryId] = useState(null)

  const handleRefresh = async () => {
    setLoading(true)
    setError(null)
    try {
      const historyRes = await fetch(`${API_URL}/api/dashboard/history`)
      if (!historyRes.ok) throw new Error('Failed to fetch prediction history')
      const historyData = await historyRes.json()
      setHistory(historyData.history || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    let ignore = false
    const load = async () => {
      try {
        const historyRes = await fetch(`${API_URL}/api/dashboard/history`)
        if (!historyRes.ok) throw new Error('Failed to fetch prediction history')
        const historyData = await historyRes.json()
        if (!ignore) {
          setHistory(historyData.history || [])
        }
      } catch (err) {
        if (!ignore) setError(err.message)
      } finally {
        if (!ignore) setLoading(false)
      }
    }
    load()
    return () => {
      ignore = true
    }
  }, [])

  if (loading) {
    return (
      <div className="dashboard-page">
        <div className="card card-royal mb-4">
          <h2>📊 Dashboard</h2>
          <p>View prediction history and explainability.</p>
        </div>
        <div className="flex-center" style={{ padding: '4rem 0' }}>
          <div className="spinner"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="dashboard-page">
      <div className="card card-royal mb-4">
        <h2>📊 Dashboard</h2>
        <p>View prediction history and explainability.</p>
      </div>

      {error && (
        <div className="card mb-4" style={{ background: 'var(--risk-high-bg)', border: '2px solid var(--risk-high)' }}>
          <p style={{ color: 'var(--risk-high)', fontWeight: '600' }}>❌ Error: {error}</p>
        </div>
      )}

      <div className="card">
        <div className="flex-between">
          <h3>📜 Prediction History & Explainability</h3>
          <button className="btn btn-secondary" onClick={handleRefresh}>
            🔄 Refresh
          </button>
        </div>
        {history.length === 0 ? (
          <p className="mt-4" style={{ color: 'var(--color-gray-500)' }}>No prediction history available.</p>
        ) : (
          <div className="table-container mt-4">
            <table>
              <thead>
                <tr>
                  <th>Source</th>
                  <th>Customer ID</th>
                  <th>
                    <Tooltip content="Specific individual SHAP drivers that pushed this customer's risk score up.">
                      Top Factors
                    </Tooltip>
                  </th>
                  <th>Recommended Action</th>
                  <th>Timestamp</th>
                  <th>Explainability</th>
                </tr>
              </thead>
              <tbody>
                {history.slice(0, 50).map((item, index) => {
                  const rowKey = item.id ? `${item.type}-${item.id}` : `${item.type}-${item.customer_id}-${index}`
                  const isExpanded = expandedHistoryId === rowKey
                  const isBatch = item.type === 'batch'
                  return (
                    <React.Fragment key={rowKey}>
                      <tr style={{ background: isExpanded ? '#f8fafc' : 'transparent', borderBottom: isExpanded ? 'none' : '1px solid var(--color-gray-200)' }}>
                        <td>
                          <span style={{
                            display: 'inline-block',
                            padding: '0.2rem 0.5rem',
                            borderRadius: '4px',
                            fontSize: '0.75rem',
                            fontWeight: '600',
                            background: isBatch ? '#ede9fe' : '#e0f2fe',
                            color: isBatch ? '#6d28d9' : '#0369a1'
                          }}>
                            {isBatch ? 'Batch' : 'Single'}
                          </span>
                        </td>
                        <td style={{ fontWeight: '600', color: '#1e293b' }}>{item.customer_id}</td>
                        <td style={{ maxWidth: '200px', fontSize: '0.85rem' }}>
                          {Array.isArray(item.top_factors) 
                            ? item.top_factors.slice(0, 2).join(', ') 
                            : (typeof item.top_factors === 'string' ? item.top_factors.split(', ').slice(0, 2).join(', ') : 'N/A')}
                        </td>
                        <td style={{ maxWidth: '220px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '0.85rem' }}>
                          {item.recommended_action || 'N/A'}
                        </td>
                        <td style={{ fontSize: '0.8rem', color: '#64748b', whiteSpace: 'nowrap' }}>
                          {item.timestamp ? item.timestamp.split('.')[0] : 'N/A'}
                        </td>
                        <td>
                          <button
                            className={`btn btn-sm ${isExpanded ? 'btn-primary' : 'btn-outline'}`}
                            onClick={() => setExpandedHistoryId(prev => prev === rowKey ? null : rowKey)}
                            style={{ padding: '0.25rem 0.55rem', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}
                          >
                            <span>{isExpanded ? '▲ Close' : '🔍 Explain'}</span>
                          </button>
                        </td>
                      </tr>

                      {isExpanded && (
                        <tr style={{ background: '#f8fafc' }}>
                          <td colSpan={6} style={{ padding: '1.25rem 1.5rem', borderBottom: '2px solid #cbd5e1' }}>
                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                              {/* Plain English Explanation */}
                              {item.explanation ? (
                                <div style={{ background: '#ffffff', padding: '0.9rem 1.1rem', borderRadius: '8px', border: '1px solid #e2e8f0', borderLeft: '4px solid #3b82f6' }}>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
                                    <span style={{ fontSize: '1rem' }}>💡</span>
                                    <strong style={{ color: '#1e40af', fontSize: '0.9rem' }}>Plain-English Explanation (SHAP Generated)</strong>
                                    <span style={{ marginLeft: 'auto', fontSize: '0.75rem', background: '#dbeafe', color: '#1e40af', padding: '0.15rem 0.45rem', borderRadius: '4px', fontWeight: '600' }}>
                                      {item.type.toUpperCase()} PREDICTION
                                    </span>
                                  </div>
                                  <p style={{ margin: 0, whiteSpace: 'pre-line', fontSize: '0.875rem', color: '#334155', lineHeight: '1.5' }}>
                                    {item.explanation}
                                  </p>
                                </div>
                              ) : (
                                <div style={{ background: '#fffbeb', padding: '0.75rem 1rem', borderRadius: '6px', border: '1px solid #fef3c7', color: '#92400e', fontSize: '0.85rem' }}>
                                  ℹ️ Detailed text explanation not stored for this record (created prior to full explainability migration).
                                </div>
                              )}

                              <div className="grid grid-2" style={{ gap: '1rem' }}>
                                {/* Top SHAP Drivers */}
                                <div style={{ background: '#ffffff', padding: '0.9rem 1.1rem', borderRadius: '8px', border: '1px solid #fee2e2' }}>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.6rem' }}>
                                    <span>🔴</span>
                                    <strong style={{ color: '#991b1b', fontSize: '0.875rem' }}>Top Risk Factors (SHAP +)</strong>
                                  </div>
                                  {item.shap?.top_risk_factors?.length > 0 ? (
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                                      {item.shap.top_risk_factors.slice(0, 4).map((f, fi) => (
                                        <div key={fi} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.825rem' }}>
                                          <span style={{ color: '#334155' }}>{f.display_name}</span>
                                          <span style={{ color: '#dc2626', fontWeight: '700' }}>+{f.shap_value}</span>
                                        </div>
                                      ))}
                                    </div>
                                  ) : (
                                    <div style={{ fontSize: '0.825rem', color: '#475569' }}>
                                      {Array.isArray(item.top_factors) && item.top_factors.length > 0 
                                        ? item.top_factors.join(', ')
                                        : 'Standard profile factors.'}
                                    </div>
                                  )}
                                </div>

                                {/* Protective Factors */}
                                <div style={{ background: '#ffffff', padding: '0.9rem 1.1rem', borderRadius: '8px', border: '1px solid #dcfce7' }}>
                                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.6rem' }}>
                                    <span>🟢</span>
                                    <strong style={{ color: '#166534', fontSize: '0.875rem' }}>Top Protective Factors (SHAP -)</strong>
                                  </div>
                                  {item.shap?.top_protective_factors?.length > 0 ? (
                                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                                      {item.shap.top_protective_factors.slice(0, 4).map((f, fi) => (
                                        <div key={fi} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.825rem' }}>
                                          <span style={{ color: '#334155' }}>{f.display_name}</span>
                                          <span style={{ color: '#16a34a', fontWeight: '700' }}>{f.shap_value}</span>
                                        </div>
                                      ))}
                                    </div>
                                  ) : (
                                    <div style={{ fontSize: '0.825rem', color: '#64748b' }}>
                                      {item.protective_factors?.length > 0 ? (
                                        item.protective_factors.map((p, pi) => (
                                          <div key={pi} style={{ color: '#16a34a', fontWeight: '600' }}>{p.factor || p}</div>
                                        ))
                                      ) : 'No significant protective factors.'}
                                    </div>
                                  )}
                                </div>
                              </div>

                              {/* Recommendations */}
                              {item.recommendations?.length > 0 && (
                                <div style={{ background: '#ffffff', padding: '0.9rem 1.1rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                                  <strong style={{ color: '#1e40af', fontSize: '0.875rem', display: 'block', marginBottom: '0.5rem' }}>
                                    🎯 Retention Plan & Strategy
                                  </strong>
                                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                                    {item.recommendations.map((rec, ri) => (
                                      <div key={ri} style={{ fontSize: '0.825rem', padding: '0.4rem 0.6rem', background: '#f8fafc', borderRadius: '4px', borderLeft: '3px solid #2563eb' }}>
                                        <strong>{rec.strategy || rec.action}</strong>: {rec.details || rec.action}
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}

export default Dashboard
