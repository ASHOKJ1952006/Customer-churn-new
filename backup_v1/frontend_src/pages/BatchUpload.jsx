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

  const getRiskBadgeClass = (riskLevel) => {
    switch (riskLevel) {
      case 'High': return 'badge-high'
      case 'Medium': return 'badge-medium'
      case 'Low': return 'badge-low'
      default: return ''
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
            accept=".csv"
            onChange={handleChange}
            style={{ display: 'none' }}
          />
          <label htmlFor="file-upload" className="upload-label">
            <div className="upload-icon">📁</div>
            <p className="upload-text">
              {file ? file.name : 'Drag & drop CSV file here, or click to select'}
            </p>
            <p className="upload-hint">Supported format: .csv</p>
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
              <h4>📋 Customer Results</h4>
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
                    <th>Probability</th>
                    <th>
                      <Tooltip content="Low risk means the model's prediction is below the decision threshold — action probably isn't needed. Medium/High mean the model believes this customer is likely to churn and flags them for retention outreach.">
                        Risk Level
                      </Tooltip>
                    </th>
                    <th>
                      <Tooltip content="These are the specific factors that pushed this customer's risk score up the most, based on their profile — not general trends, but what mattered for THIS customer specifically.">
                        Top Factors
                      </Tooltip>
                    </th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {result.results.slice(0, 50).map((item, index) => (
                    <tr key={index}>
                      <td>{item.customer_id}</td>
                      <td>{(item.probability * 100).toFixed(1)}%</td>
                      <td>
                        <span className={`badge ${getRiskBadgeClass(item.risk_level)}`}>
                          {item.risk_level}
                        </span>
                      </td>
                      <td>{item.top_factors.slice(0, 2).join(', ')}</td>
                      <td style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {item.recommended_action}
                      </td>
                    </tr>
                  ))}
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
