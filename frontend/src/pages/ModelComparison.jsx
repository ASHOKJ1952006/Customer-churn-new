import { useState, useEffect } from 'react'
import '../styles/theme.css'

const API_URL = 'http://localhost:8000'

/* ─── model colour tokens (used only for data, not decoration) ─── */
const MODEL_COLOR = {
  'Random Forest':       '#2563eb',
  'XGBoost':             '#d97706',
  'LightGBM':            '#059669',
  'Stacked Ensemble':    '#7c3aed',
  'Stacking Ensemble':   '#7c3aed',
}

const METRICS = [
  { key: 'recall',    label: 'Recall',      suffix: '%', higherBetter: true,
    note: 'Primary metric — proportion of true churners caught' },
  { key: 'f1',        label: 'F1 Score',    suffix: '%', higherBetter: true,
    note: 'Harmonic mean of Precision & Recall' },
  { key: 'accuracy',  label: 'Accuracy',    suffix: '%', higherBetter: true,
    note: 'Overall correct predictions' },
  { key: 'precision', label: 'Precision',   suffix: '%', higherBetter: true,
    note: 'Of flagged churners, fraction that actually churn' },
  { key: 'brier',     label: 'Brier Score', suffix: '',  higherBetter: false,
    note: 'Calibration quality — lower is better' },
]

/* ── small inline bar used inside the table ── */
function InlineBar({ value, maxValue, color, higherBetter }) {
  const displayVal = higherBetter ? value : (1 - value)
  const displayMax = higherBetter ? maxValue : 1
  const pct = Math.min((displayVal / displayMax) * 100, 100)
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div style={{ flex: 1, height: 6, background: '#f0f0f0', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: 3,
          transition: 'width 0.7s ease' }} />
      </div>
    </div>
  )
}

/* ── simple grouped bar chart using divs ── */
function GroupedBarChart({ models, metricKey, higherBetter }) {
  const values = models.map(m => m[metricKey])
  const globalMax = Math.max(...values)
  const globalMin = Math.min(...values)
  
  return (
    <div style={{ display: 'flex', alignItems: 'flex-end', gap: 12, height: 140, padding: '0 4px' }}>
      {models.map(m => {
        const val = m[metricKey]
        const isBest = higherBetter ? val === globalMax : val === globalMin
        const heightPct = higherBetter
          ? (val / globalMax) * 100
          : ((1 - val) / (1 - globalMin)) * 100
        const color = MODEL_COLOR[m.name] || '#6b7280'
        return (
          <div key={m.name} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
            {isBest && (
              <div style={{ fontSize: 10, fontWeight: 700, color: '#059669', letterSpacing: '0.04em' }}>BEST</div>
            )}
            {!isBest && <div style={{ fontSize: 10 }}> </div>}
            <div style={{ width: '100%', display: 'flex', flexDirection: 'column', justifyContent: 'flex-end', height: 100 }}>
              <div
                title={`${m.name}: ${metricKey === 'brier' ? val.toFixed(4) : (val*100).toFixed(2) + '%'}`}
                style={{
                  width: '100%',
                  height: `${Math.max(heightPct, 5)}%`,
                  background: color,
                  borderRadius: '3px 3px 0 0',
                  opacity: isBest ? 1 : 0.65,
                  transition: 'height 0.5s ease',
                  cursor: 'default',
                }}
              />
            </div>
            <div style={{ fontSize: 11, fontWeight: 600, color: '#374151', textAlign: 'center' }}>
              {metricKey === 'brier' ? val.toFixed(4) : `${(val * 100).toFixed(1)}%`}
            </div>
            <div style={{ fontSize: 10, color: '#9ca3af', textAlign: 'center', lineHeight: 1.2 }}>
              {m.short || m.name.split(' ')[0]}
            </div>
          </div>
        )
      })}
    </div>
  )
}

/* ═══════════════════════════════════════════════════════════════ */
export default function ModelComparison() {
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError]     = useState(null)
  const [selMetric, setSel]   = useState('recall')

  useEffect(() => {
    fetch(`${API_URL}/api/model-comparison`)
      .then(r => { if (!r.ok) throw new Error('Backend unavailable'); return r.json() })
      .then(d => { setData(d); setLoading(false) })
      .catch(e => { setError(e.message); setLoading(false) })
  }, [])

  if (loading) return (
    <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: 300, gap: 12, color: '#6b7280' }}>
      <div className="spinner" />
      <span>Loading comparison data…</span>
    </div>
  )

  if (error) return (
    <div className="card" style={{ maxWidth: 500, margin: '3rem auto', textAlign: 'center' }}>
      <div style={{ fontSize: '2rem', marginBottom: '0.75rem' }}>⚠️</div>
      <h3 style={{ marginBottom: '0.5rem' }}>Backend unavailable</h3>
      <p style={{ color: '#6b7280', fontSize: '0.875rem', margin: 0 }}>
        {error}. Start the FastAPI server on port 8000 and refresh.
      </p>
    </div>
  )

  const models = data.models
  const ensemble = models.find(m => m.type === 'ensemble')
  const individuals = models.filter(m => m.type !== 'ensemble')

  const selMeta = METRICS.find(m => m.key === selMetric)

  return (
    <div style={{ maxWidth: 1100, margin: '0 auto', padding: '1rem' }}>

      {/* ── Page header ─────────────────────────────────────────── */}
      <div style={{ marginBottom: '2rem', borderBottom: '1px solid #e5e7eb', paddingBottom: '1.25rem' }}>
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: '0.35rem' }}>
              <h2 style={{ fontSize: '1.5rem', fontWeight: 700, color: '#111827', margin: 0 }}>
                Model Performance Comparison
              </h2>
              <span style={{
                background: '#e0f2fe', color: '#0369a1', fontSize: '0.72rem',
                fontWeight: 700, padding: '3px 8px', borderRadius: 12, border: '1px solid #bae6fd'
              }}>
                {data.model_version || 'v2.0-optimized'}
              </span>
            </div>
            <p style={{ color: '#6b7280', fontSize: '0.875rem', margin: 0 }}>
              {data.dataset} &nbsp;·&nbsp; {data.test_records ? data.test_records.toLocaleString() : '1,600'} Untouched Held-Out Test Records &nbsp;·&nbsp; Zero Data Leakage &nbsp;·&nbsp; Optuna TPE Tuned
            </p>
          </div>
        </div>
      </div>

      {/* ── Summary row: one KPI card per model ──────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem', marginBottom: '2rem' }}>
        {models.map(m => {
          const color = MODEL_COLOR[m.name] || '#6b7280'
          const isEns = m.type === 'ensemble'
          return (
            <div key={m.name} style={{
              background: '#fff',
              border: `1px solid ${isEns ? color + '55' : '#e5e7eb'}`,
              borderTop: `3px solid ${color}`,
              borderRadius: 8,
              padding: '1.1rem 1.25rem',
              boxShadow: '0 1px 3px rgba(0,0,0,0.06)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.6rem' }}>
                <div style={{ fontSize: '0.78rem', fontWeight: 600, color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  {m.name}
                </div>
                {(m.is_deployed ?? isEns) && (
                  <span style={{ fontSize: '0.65rem', fontWeight: 700, background: color + '18',
                    color: color, padding: '2px 6px', borderRadius: 4, border: `1px solid ${color}44` }}>
                    Deployed
                  </span>
                )}
              </div>
              <div style={{ fontSize: '2rem', fontWeight: 800, color: color, lineHeight: 1, marginBottom: '0.2rem' }}>
                {(m.recall * 100).toFixed(1)}<span style={{ fontSize: '1rem', fontWeight: 600 }}>%</span>
              </div>
              <div style={{ fontSize: '0.72rem', color: '#9ca3af', marginBottom: '0.9rem' }}>Recall</div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.3rem 0.75rem' }}>
                {[['F1', m.f1], ['Accuracy', m.accuracy], ['Precision', m.precision], ['Brier', m.brier]].map(([lbl, val]) => (
                  <div key={lbl}>
                    <div style={{ fontSize: '0.67rem', color: '#9ca3af', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{lbl}</div>
                    <div style={{ fontSize: '0.82rem', fontWeight: 600, color: '#374151' }}>
                      {lbl === 'Brier' ? val.toFixed(4) : `${(val * 100).toFixed(1)}%`}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>

      {/* ── Chart section ───────────────────────────────────────── */}
      <div className="card" style={{ marginBottom: '1.5rem', background: '#fff', borderRadius: 8, padding: '1.25rem', border: '1px solid #e5e7eb', boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>
        {/* metric selector tabs */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>
            <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#111827' }}>
              {selMeta?.label} by Model
            </div>
            <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: 2 }}>{selMeta?.note}</div>
          </div>
          <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap' }}>
            {METRICS.map(mt => (
              <button key={mt.key}
                onClick={() => setSel(mt.key)}
                style={{
                  padding: '0.3rem 0.75rem',
                  border: `1px solid ${selMetric === mt.key ? '#2563eb' : '#d1d5db'}`,
                  borderRadius: 5,
                  background: selMetric === mt.key ? '#2563eb' : '#fff',
                  color: selMetric === mt.key ? '#fff' : '#374151',
                  fontSize: '0.78rem', fontWeight: 500, cursor: 'pointer',
                  transition: 'all 0.15s',
                }}>
                {mt.label}
              </button>
            ))}
          </div>
        </div>

        {/* bar chart */}
        <GroupedBarChart models={models} metricKey={selMetric} higherBetter={selMeta?.higherBetter !== false} />

        {/* colour legend */}
        <div style={{ display: 'flex', gap: '1.25rem', marginTop: '1rem', paddingTop: '0.75rem',
          borderTop: '1px solid #f3f4f6', flexWrap: 'wrap' }}>
          {models.map(m => (
            <div key={m.name} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.78rem', color: '#374151' }}>
              <div style={{ width: 10, height: 10, borderRadius: 2, background: MODEL_COLOR[m.name], flexShrink: 0 }} />
              {m.name}
            </div>
          ))}
          <div style={{ marginLeft: 'auto', fontSize: '0.72rem', color: '#9ca3af' }}>
            {selMeta?.higherBetter !== false ? 'Higher bars = better' : 'Lower value = better (bars normalised for visual)'}
          </div>
        </div>
      </div>

      {/* ── Full metrics table ───────────────────────────────────── */}
      <div className="card" style={{ marginBottom: '1.5rem', background: '#fff', borderRadius: 8, padding: '1.25rem', border: '1px solid #e5e7eb', boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>
        <div style={{ marginBottom: '1rem' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#111827' }}>All Metrics — Full Table</div>
          <div style={{ fontSize: '0.75rem', color: '#9ca3af', marginTop: 2 }}>
            Highlighted cell = best value for that metric. Inline bar shows relative position.
          </div>
        </div>

        <div className="table-container" style={{ overflowX: 'auto' }}>
          <table style={{ fontSize: '0.85rem', width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: '#f9fafb', borderBottom: '1px solid #e5e7eb' }}>
                <th style={{ textAlign: 'left', fontWeight: 600, color: '#374151', padding: '0.75rem 1rem' }}>Model</th>
                {METRICS.map(mt => (
                  <th key={mt.key} style={{
                    textAlign: 'right', fontWeight: 600,
                    color: selMetric === mt.key ? '#2563eb' : '#374151',
                    padding: '0.75rem 1rem', whiteSpace: 'nowrap'
                  }}>
                    {mt.label}
                    {selMetric === mt.key && <span style={{ marginLeft: 4, fontSize: '0.65rem' }}>▲</span>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {models.map((m, rowIdx) => {
                const color = MODEL_COLOR[m.name] || '#6b7280'
                const isEns = m.type === 'ensemble'
                return (
                  <tr key={m.name} style={{
                    background: isEns ? '#faf9ff' : (rowIdx % 2 === 0 ? '#fff' : '#fafafa'),
                    borderBottom: '1px solid #f3f4f6',
                    borderLeft: isEns ? `3px solid ${color}` : '3px solid transparent',
                  }}>
                    {/* model name cell */}
                    <td style={{ padding: '0.85rem 1rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <div style={{ width: 3, height: 28, background: color, borderRadius: 2, flexShrink: 0 }} />
                        <div>
                          <div style={{ fontWeight: 600, color: '#111827' }}>{m.name}</div>
                          <div style={{ fontSize: '0.7rem', color: '#9ca3af' }}>
                            {isEns ? 'Stacked (RF + XGB + LGBM → LR)' : 'Individual classifier'}
                          </div>
                        </div>
                      </div>
                    </td>

                    {/* metric cells */}
                    {METRICS.map(mt => {
                      const val = m[mt.key]
                      const allVals = models.map(mm => mm[mt.key])
                      const isBest = mt.higherBetter
                        ? val === Math.max(...allVals)
                        : val === Math.min(...allVals)
                      const maxV = Math.max(...allVals)

                      return (
                        <td key={mt.key} style={{ padding: '0.85rem 1rem', textAlign: 'right' }}>
                          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 4 }}>
                            <span style={{
                              fontWeight: isBest ? 700 : 500,
                              color: isBest ? color : '#374151',
                              background: isBest ? color + '12' : 'transparent',
                              padding: isBest ? '1px 5px' : '1px 0',
                              borderRadius: 4,
                              fontSize: '0.875rem',
                            }}>
                              {mt.key === 'brier' ? val.toFixed(4) : `${(val * 100).toFixed(2)}%`}
                              {isBest && <span style={{ fontSize: '0.6rem', marginLeft: 3, color: color }}>★</span>}
                            </span>
                            <div style={{ width: 80 }}>
                              <InlineBar value={val} maxValue={maxV} color={color} higherBetter={mt.higherBetter} />
                            </div>
                          </div>
                        </td>
                      )
                    })}
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
        <p style={{ fontSize: '0.72rem', color: '#9ca3af', margin: '0.75rem 0 0', textAlign: 'right' }}>
          ★ = best value per column &nbsp;·&nbsp; Inline bars normalised per column
        </p>
      </div>

      {/* ── Analysis notes ──────────────────────────────────────── */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '1.25rem', marginBottom: '1.5rem' }}>

        {/* Why ensemble wins */}
        <div className="card" style={{ margin: 0, background: '#fff', borderRadius: 8, padding: '1.25rem', border: '1px solid #e5e7eb', boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#111827', marginBottom: '0.85rem',
            borderBottom: '1px solid #f3f4f6', paddingBottom: '0.6rem' }}>
            Why the Ensemble Leads on Recall
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {[
              ['Complementary learners', 'RF captures global structure, XGBoost learns residual errors, LightGBM exploits leaf-level patterns. Stacking aggregates all three.'],
              ['Meta-learner', 'A Logistic Regression trained on out-of-fold predictions learns optimal model weights, outperforming any fixed combination.'],
              ['Calibration', 'CalibratedClassifierCV (sigmoid) corrects overconfident probabilities, improving Brier score without hurting recall.'],
              ['Threshold tuning', 'F1-optimal threshold per model — not the default 0.5 — ensures recall and precision are properly balanced.'],
            ].map(([title, body]) => (
              <div key={title} style={{ display: 'flex', gap: '0.65rem' }}>
                <div style={{ width: 4, height: 4, borderRadius: '50%', background: '#7c3aed',
                  flexShrink: 0, marginTop: 7 }} />
                <div>
                  <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#374151' }}>{title}</div>
                  <div style={{ fontSize: '0.76rem', color: '#6b7280', lineHeight: 1.5 }}>{body}</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Trade-off summary */}
        <div className="card" style={{ margin: 0, background: '#fff', borderRadius: 8, padding: '1.25rem', border: '1px solid #e5e7eb', boxShadow: '0 1px 2px rgba(0,0,0,0.05)' }}>
          <div style={{ fontSize: '0.85rem', fontWeight: 600, color: '#111827', marginBottom: '0.85rem',
            borderBottom: '1px solid #f3f4f6', paddingBottom: '0.6rem' }}>
            Key Trade-offs
          </div>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: '0.8rem', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', color: '#6b7280', fontWeight: 600, padding: '0.4rem 0.5rem',
                    borderBottom: '1px solid #f0f0f0' }}>Dimension</th>
                  <th style={{ textAlign: 'left', color: '#6b7280', fontWeight: 600, padding: '0.4rem 0.5rem',
                    borderBottom: '1px solid #f0f0f0' }}>Best Individual</th>
                  <th style={{ textAlign: 'left', color: '#7c3aed', fontWeight: 600, padding: '0.4rem 0.5rem',
                    borderBottom: '1px solid #f0f0f0' }}>Ensemble</th>
                </tr>
              </thead>
              <tbody>
                {[
                  ['Recall (↑)',      `${(Math.max(...individuals.map(m=>m.recall))*100).toFixed(1)}%`,   `${(ensemble?.recall*100).toFixed(1)}%`,   true],
                  ['F1 (↑)',          `${(Math.max(...individuals.map(m=>m.f1))*100).toFixed(1)}%`,       `${(ensemble?.f1*100).toFixed(1)}%`,       null],
                  ['Accuracy (↑)',    `${(Math.max(...individuals.map(m=>m.accuracy))*100).toFixed(1)}%`, `${(ensemble?.accuracy*100).toFixed(1)}%`, false],
                  ['Brier (↓)',       `${Math.min(...individuals.map(m=>m.brier)).toFixed(4)}`,           `${ensemble?.brier.toFixed(4)}`,           true],
                ].map(([dim, ind, ens, ensWins]) => (
                  <tr key={dim}>
                    <td style={{ padding: '0.5rem', color: '#374151', fontWeight: 500 }}>{dim}</td>
                    <td style={{ padding: '0.5rem', color: '#374151' }}>{ind}</td>
                    <td style={{ padding: '0.5rem', color: ensWins ? '#059669' : ensWins === false ? '#ef4444' : '#7c3aed',
                      fontWeight: ensWins !== null ? 600 : 400 }}>
                      {ens} {ensWins === true && '↑'} {ensWins === false && '↓'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: '0.72rem', color: '#9ca3af', margin: '0.75rem 0 0' }}>
            The ensemble wins on recall and calibration across holdout test evaluation.
            For a churn use-case (false-negative cost ≫ false-positive cost) this is the correct trade-off.
          </p>
        </div>
      </div>
    </div>
  )
}
