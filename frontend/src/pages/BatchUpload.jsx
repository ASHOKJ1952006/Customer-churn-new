import { useState } from 'react'
import Tooltip from '../components/Tooltip'
import '../styles/theme.css'

const API_URL = 'http://localhost:8000'

function BatchUpload({ dataset }) {
  const [file, setFile] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [expandedCustomer, setExpandedCustomer] = useState(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0])
    }
  }

  const handleChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!file) {
      setError('Please select a file')
      return
    }

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('dataset', dataset)

    try {
      const response = await fetch(`${API_URL}/api/predict/batch`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Batch prediction failed')
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleDownload = async (batchId) => {
    try {
      const response = await fetch(`${API_URL}/api/predict/batch/download/${batchId}`)
      if (!response.ok) {
        throw new Error('Download failed')
      }
      
      const blob = await response.blob()
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `batch_results_${batchId}.csv`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      document.body.removeChild(a)
    } catch (err) {
      setError(err.message)
    }
  }


  return (
    <div className="batch-upload-page">
      <div className="card card-royal mb-4">
        <h2>📋 Batch Upload</h2>
        <p>Upload a CSV file to predict churn risk for multiple customers.</p>
      </div>

      <div className="card">
        <div 
          className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            id="file-upload"
            accept=".csv, .xlsx, .xls"
            onChange={handleChange}
            style={{ display: 'none' }}
          />
          <label htmlFor="file-upload" className="upload-label">
            <div className="upload-icon">📁</div>
            <p className="upload-text">
              {file ? file.name : 'Drag & drop CSV or Excel file here, or click to select'}
            </p>
            <p className="upload-hint">Supported formats: .csv, .xlsx, .xls</p>
          </label>
        </div>

        {file && (
          <div className="mt-4">
            <button 
              className="btn btn-royal" 
              onClick={handleSubmit}
              disabled={loading}
              style={{ width: '100%', padding: '1rem', fontSize: '1.125rem' }}
            >
              {loading ? 'Processing...' : '🚀 Process Batch'}
            </button>
          </div>
        )}
      </div>

      {error && (
        <div className="card mt-4" style={{ background: 'var(--risk-high-bg)', border: '2px solid var(--risk-high)' }}>
          <p style={{ color: 'var(--risk-high)', fontWeight: '600' }}>❌ Error: {error}</p>
        </div>
      )}

      {result && (
        <div className="card card-royal mt-4">
          <h3>📊 Batch Results</h3>
          
          <div className="grid grid-3 mt-4">
            <div className="summary-card">
              <p className="summary-label">Total Customers</p>
              <p className="summary-value">{result.summary.total}</p>
            </div>
            <div className="summary-card">
              <p className="summary-label">High Risk</p>
              <p className="summary-value" style={{ color: 'var(--risk-high)' }}>{result.summary.high}</p>
            </div>
            <div className="summary-card">
              <p className="summary-label">Medium Risk</p>
              <p className="summary-value" style={{ color: '#856404' }}>{result.summary.medium}</p>
            </div>
          </div>

          {/* Organizational Decision Support Aggregations (Requirement 24) */}
          {(result.summary.top_organizational_drivers || result.summary.top_retention_strategies) && (
            <div className="grid grid-2 mt-4" style={{ gap: '1.25rem' }}>
              {/* Top Organizational Churn Drivers */}
              <div className="card" style={{ background: '#ffffff', border: '1px solid #fee2e2', borderLeft: '5px solid #ef4444' }}>
                <h4 style={{ color: '#991b1b', marginTop: 0, marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span>🚨</span> Top Organizational Churn Drivers
                </h4>
                <p style={{ margin: '0 0 0.75rem 0', fontSize: '0.85rem', color: '#64748b' }}>
                  Most frequent structural risk drivers across the analyzed customer cohort:
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                  {result.summary.top_organizational_drivers?.map((d, i) => (
                    <div key={i}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginBottom: '0.2rem' }}>
                        <span style={{ fontWeight: '600', color: '#1e293b' }}>{d.driver}</span>
                        <span style={{ color: '#991b1b', fontWeight: '700' }}>{d.percentage}% ({d.count})</span>
                      </div>
                      <div style={{ width: '100%', height: '6px', background: '#fee2e2', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${Math.min(d.percentage, 100)}%`, height: '100%', background: '#ef4444', borderRadius: '3px' }}></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Top Retention Strategies */}
              <div className="card" style={{ background: '#ffffff', border: '1px solid #dbeafe', borderLeft: '5px solid #2563eb' }}>
                <h4 style={{ color: '#1e40af', marginTop: 0, marginBottom: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span>🎯</span> Priority Retention Campaigns
                </h4>
                <p style={{ margin: '0 0 0.75rem 0', fontSize: '0.85rem', color: '#64748b' }}>
                  Recommended high-impact organizational campaigns for this cohort:
                </p>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
                  {result.summary.top_retention_strategies?.map((s, i) => (
                    <div key={i}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.875rem', marginBottom: '0.2rem' }}>
                        <span style={{ fontWeight: '600', color: '#1e293b' }}>{s.strategy}</span>
                        <span style={{ color: '#2563eb', fontWeight: '700' }}>{s.percentage}% ({s.count})</span>
                      </div>
                      <div style={{ width: '100%', height: '6px', background: '#dbeafe', borderRadius: '3px', overflow: 'hidden' }}>
                        <div style={{ width: `${Math.min(s.percentage, 100)}%`, height: '100%', background: '#2563eb', borderRadius: '3px' }}></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          <div className="mt-4">
            <div className="flex-between">
              <h4>📋 Customer Results & Individual Explanations</h4>
              <button 
                className="btn btn-secondary"
                onClick={() => handleDownload(result.batch_id)}
              >
                📥 Download CSV
              </button>
            </div>
            
            <div className="table-container mt-4">
              <table>
                <thead>
                  <tr>
                    <th>Customer ID</th>
                    <th>
                      <Tooltip content="Specific individual SHAP drivers that pushed this customer's risk score up.">
                        Top Factors
                      </Tooltip>
                    </th>
                    <th>Primary Recommendation</th>
                    <th>Explainability</th>
                  </tr>
                </thead>
                <tbody>
                  {result.results.slice(0, 50).map((item, index) => {
                    const isExpanded = expandedCustomer === item.customer_id
                    return (
                      <>
                        <tr key={index} style={{ background: isExpanded ? '#f8fafc' : 'transparent', borderBottom: isExpanded ? 'none' : '1px solid var(--color-gray-200)' }}>
                          <td style={{ fontWeight: '600', color: '#1e293b' }}>{item.customer_id}</td>
                          <td style={{ maxWidth: '220px', fontSize: '0.875rem' }}>
                            {Array.isArray(item.top_factors) ? item.top_factors.slice(0, 2).join(', ') : item.top_factors}
                          </td>
                          <td style={{ maxWidth: '240px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontSize: '0.875rem' }}>
                            {item.recommended_action}
                          </td>
                          <td>
                            <button
                              className={`btn btn-sm ${isExpanded ? 'btn-primary' : 'btn-outline'}`}
                              onClick={() => setExpandedCustomer(prev => prev === item.customer_id ? null : item.customer_id)}
                              style={{ padding: '0.25rem 0.6rem', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.3rem' }}
                            >
                              <span>{isExpanded ? '▲ Close' : '🔍 Explain'}</span>
                            </button>
                          </td>
                        </tr>

                        {isExpanded && (
                          <tr key={`exp-${index}`} style={{ background: '#f8fafc' }}>
                            <td colSpan={4} style={{ padding: '1rem 1.5rem', borderBottom: '2px solid #cbd5e1' }}>
                              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                                {/* Plain English Explanation */}
                                {item.explanation && (
                                  <div style={{ background: '#ffffff', padding: '0.9rem 1.1rem', borderRadius: '8px', border: '1px solid #e2e8f0', borderLeft: '4px solid #3b82f6' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.4rem' }}>
                                      <span style={{ fontSize: '1rem' }}>💡</span>
                                      <strong style={{ color: '#1e40af', fontSize: '0.9rem' }}>Plain-English Churn Assessment (SHAP Derived)</strong>
                                    </div>
                                    <p style={{ margin: 0, whiteSpace: 'pre-line', fontSize: '0.875rem', color: '#334155', lineHeight: '1.5' }}>
                                      {item.explanation}
                                    </p>
                                  </div>
                                )}

                                <div className="grid grid-2" style={{ gap: '1rem' }}>
                                  {/* Top SHAP Risk Drivers */}
                                  <div style={{ background: '#ffffff', padding: '0.9rem 1.1rem', borderRadius: '8px', border: '1px solid #fee2e2' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: '0.6rem' }}>
                                      <span>🔴</span>
                                      <strong style={{ color: '#991b1b', fontSize: '0.875rem' }}>Top Churn Risk Drivers (SHAP +)</strong>
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
                                      <p style={{ fontSize: '0.825rem', color: '#64748b', margin: 0 }}>
                                        {Array.isArray(item.top_factors) ? item.top_factors.join(', ') : 'No high risk drivers.'}
                                      </p>
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
                                      <p style={{ fontSize: '0.825rem', color: '#64748b', margin: 0 }}>
                                        No significant protective factors detected.
                                      </p>
                                    )}
                                  </div>
                                </div>

                                {/* Dynamic Retention Recommendations */}
                                {item.recommendations?.length > 0 && (
                                  <div style={{ background: '#ffffff', padding: '0.9rem 1.1rem', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                                    <strong style={{ color: '#1e293b', fontSize: '0.875rem', display: 'block', marginBottom: '0.5rem' }}>
                                      🎯 Actionable Retention Plan
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
                      </>
                    )
                  })}
                </tbody>
              </table>
            </div>
            
            {result.results.length > 50 && (
              <p className="mt-4" style={{ color: 'var(--color-gray-500)', fontSize: 'var(--font-size-sm)' }}>
                Showing first 50 of {result.results.length} results. Download CSV for complete list.
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default BatchUpload
