import { useState, useEffect } from 'react'
import Tooltip from '../components/Tooltip'
import '../styles/theme.css'

const API_URL = 'http://localhost:8000'

function Dashboard() {
  const [activeTab, setActiveTab] = useState('history')
  const [history, setHistory] = useState([])
  const [riskQueue, setRiskQueue] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const [historyRes, queueRes] = await Promise.all([
        fetch(`${API_URL}/api/dashboard/history`),
        fetch(`${API_URL}/api/dashboard/risk-queue`)
      ])

      if (!historyRes.ok || !queueRes.ok) {
        throw new Error('Failed to fetch dashboard data')
      }

      const historyData = await historyRes.json()
      const queueData = await queueRes.json()

      setHistory(historyData.history || [])
      setRiskQueue(queueData.queue || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleStatusUpdate = async (id, status, notes) => {
    try {
      const response = await fetch(`${API_URL}/api/dashboard/risk-queue/${id}`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ status, notes }),
      })

      if (!response.ok) {
        throw new Error('Failed to update status')
      }

      // Refresh data
      fetchData()
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

  const getStatusBadgeClass = (status) => {
    switch (status) {
      case 'pending': return 'badge-medium'
      case 'contacted': return 'badge-low'
      case 'resolved': return 'badge-high'
      default: return ''
    }
  }

  if (loading) {
    return (
      <div className="dashboard-page">
        <div className="card card-royal mb-4">
          <h2>📊 Dashboard</h2>
          <p>View prediction history and manage risk queue.</p>
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
        <p>View prediction history and manage risk queue.</p>
      </div>

      {error && (
        <div className="card mb-4" style={{ background: 'var(--risk-high-bg)', border: '2px solid var(--risk-high)' }}>
          <p style={{ color: 'var(--risk-high)', fontWeight: '600' }}>❌ Error: {error}</p>
        </div>
      )}

      <div className="card">
        <div className="flex">
          <button 
            className={`nav-btn ${activeTab === 'history' ? 'active' : ''}`}
            onClick={() => setActiveTab('history')}
          >
            📜 Prediction History
          </button>
          <button 
            className={`nav-btn ${activeTab === 'risk-queue' ? 'active' : ''}`}
            onClick={() => setActiveTab('risk-queue')}
          >
            ⚠️ Risk Queue
          </button>
        </div>
      </div>

      {activeTab === 'history' && (
        <div className="card mt-4">
          <h3>📜 Prediction History</h3>
          {history.length === 0 ? (
            <p className="mt-4" style={{ color: 'var(--color-gray-500)' }}>No prediction history available.</p>
          ) : (
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
                    <th>Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {history.slice(0, 50).map((item, index) => (
                    <tr key={index}>
                      <td>{item.customer_id}</td>
                      <td>{(item.probability * 100).toFixed(1)}%</td>
                      <td>
                        <span className={`badge ${getRiskBadgeClass(item.risk_level)}`}>
                          {item.risk_level}
                        </span>
                      </td>
                      <td>
                        {Array.isArray(item.top_factors) 
                          ? item.top_factors.slice(0, 2).join(', ') 
                          : (typeof item.top_factors === 'string' ? item.top_factors.split(', ').slice(0, 2).join(', ') : 'N/A')}
                      </td>
                      <td style={{ maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {item.recommended_action || 'N/A'}
                      </td>
                      <td>{item.timestamp || 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {activeTab === 'risk-queue' && (
        <div className="card mt-4">
          <div className="flex-between">
            <h3>⚠️ Risk Queue</h3>
            <button className="btn btn-secondary" onClick={fetchData}>
              🔄 Refresh
            </button>
          </div>
          
          {riskQueue.length === 0 ? (
            <p className="mt-4" style={{ color: 'var(--color-gray-500)' }}>No items in risk queue.</p>
          ) : (
            <div className="table-container mt-4">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Customer ID</th>
                    <th>Probability</th>
                    <th>
                      <Tooltip content="Low risk means the model's prediction is below the decision threshold — action probably isn't needed. Medium/High mean the model believes this customer is likely to churn and flags them for retention outreach.">
                        Risk Level
                      </Tooltip>
                    </th>
                    <th>
                      <Tooltip content="Track your team's outreach: 'Pending' = flagged but not yet contacted, 'Contacted' = outreach has happened, 'Resolved' = this customer's risk has been addressed (retained or confirmed departing).">
                        Status
                      </Tooltip>
                    </th>
                    <th>Notes</th>
                    {/* <th>Actions</th> */}
                  </tr>
                </thead>
                <tbody>
                  {riskQueue.map((item, index) => (
                    <tr key={index}>
                      <td>{item.id}</td>
                      <td>{item.customer_id}</td>
                      <td>{(item.probability * 100).toFixed(1)}%</td>
                      <td>
                        <span className={`badge ${getRiskBadgeClass(item.risk_level)}`}>
                          {item.risk_level}
                        </span>
                      </td>
                      <td>
                        <span className={`badge ${getStatusBadgeClass(item.status)}`}>
                          {item.status}
                        </span>
                      </td>
                      <td style={{ maxWidth: '150px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {item.notes || '-'}
                      </td>
                      <td>
                        {/* <div className="flex" style={{ gap: '0.5rem' }}>
                          <button 
                            className="btn btn-outline"
                            style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem' }}
                            onClick={() => handleStatusUpdate(item.id, 'contacted', 'Customer contacted')}
                          >
                            Contact
                          </button>
                          <button 
                            className="btn btn-outline"
                            style={{ padding: '0.25rem 0.5rem', fontSize: '0.875rem' }}
                            onClick={() => handleStatusUpdate(item.id, 'resolved', 'Issue resolved')}
                          >
                            Resolve
                          </button>
                        </div> */}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default Dashboard
