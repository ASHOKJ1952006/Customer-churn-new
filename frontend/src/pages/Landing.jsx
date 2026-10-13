import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import Tooltip from '../components/Tooltip';
import '../styles/theme.css';

const API_URL = 'http://localhost:8000';

const Landing = () => {
  const navigate = useNavigate();
  const [modelStats, setModelStats] = useState({
    recall: '69.8%',
    accuracy: '78.4%',
    f1: '0.632',
    brier: '0.137'
  });

  useEffect(() => {
    fetch(`${API_URL}/api/model-comparison`)
      .then(res => res.ok ? res.json() : null)
      .then(data => {
        if (data && data.models) {
          const deployed = data.models.find(m => m.is_deployed) || data.models.find(m => m.type === 'ensemble') || data.models[0];
          if (deployed) {
            setModelStats({
              recall: `${(deployed.recall * 100).toFixed(1)}%`,
              accuracy: `${(deployed.accuracy * 100).toFixed(1)}%`,
              f1: deployed.f1.toFixed(3),
              brier: deployed.brier.toFixed(3)
            });
          }
        }
      })
      .catch(() => {});
  }, []);

  const navCards = [
    {
      id: 'single',
      title: 'Single Customer',
      description: 'Predict churn risk for an individual customer with detailed explanations',
      icon: '👤',
      route: '/single'
    },
    {
      id: 'batch',
      title: 'Batch Upload',
      description: 'Upload a CSV file to predict churn for multiple customers at once',
      icon: '📊',
      route: '/batch'
    },
    {
      id: 'whatif',
      title: 'What-If Simulator',
      description: 'Explore how changing specific factors affects churn risk',
      icon: '🔮',
      route: '/whatif'
    },
    {
      id: 'dashboard',
      title: 'Dashboard',
      description: 'View prediction history and manage high-risk customers',
      icon: '📈',
      route: '/dashboard'
    },
    {
      id: 'model-comparison',
      title: 'Model Comparison',
      description: 'Compare Random Forest, XGBoost, LightGBM and the Stacked Ensemble across key metrics',
      icon: '🏆',
      route: '/model-comparison'
    }
  ];

  const stats = [
    { 
      value: modelStats.recall, 
      label: 'Recall', 
      icon: '🎯',
      tooltip: 'Of all customers who actually churn, this is the percentage the model successfully catches. We prioritize this over raw accuracy, since missing a real churner is more costly than a false alarm.'
    },
    { 
      value: modelStats.accuracy, 
      label: 'Accuracy', 
      icon: '✓',
      tooltip: 'The percentage of all predictions (churn and no-churn) the model gets right. Can be misleading on its own when churn is rare, which is why we report it alongside recall.'
    },
    { 
      value: modelStats.f1, 
      label: 'F1 Score', 
      icon: '⚡',
      tooltip: 'A single number balancing precision (avoiding false alarms) and recall (catching real churners). Higher is better, max is 1.0.'
    },
    { 
      value: modelStats.brier, 
      label: 'Brier Score', 
      icon: '📏',
      tooltip: 'Measures how trustworthy the model\'s probability numbers are — lower is better. A well-calibrated model\'s "70% risk" predictions really do churn about 70% of the time.'
    }
  ];

  const steps = [
    {
      number: '1',
      title: 'Enter Customer Details',
      description: 'Input customer information manually or upload a CSV file'
    },
    {
      number: '2',
      title: 'Get Instant Risk Score',
      description: 'Receive an immediate churn probability with risk level classification'
    },
    {
      number: '3',
      title: 'See Why, In Plain Language',
      description: 'Understand which factors drive the risk with clear explanations'
    },
    {
      number: '4',
      title: 'Get Recommended Action',
      description: 'Receive specific retention strategies to prevent churn'
    }
  ];

  return (
    <div className="landing-page">
      {/* Header Section */}
      <header className="landing-header">
        <div className="container">
          <h1 className="landing-title">Telecom Customer Churn Predictor</h1>
          <p className="landing-subtitle">
            Predict which customers are likely to churn and why, with actionable retention recommendations
          </p>
        </div>
      </header>

      {/* Key Stats Section
      <section className="stats-section">
        <div className="container">
          <div className="stats-grid">
            {stats.map((stat, index) => (
              <div key={index} className="stat-card">
                <div className="stat-icon">{stat.icon}</div>
                <div className="stat-value">{stat.value}</div>
                <div className="stat-label">
                  <Tooltip content={stat.tooltip}>{stat.label}</Tooltip>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section> */}

      {/* Navigation Cards Section */}
      <section className="nav-section">
        <div className="container">
          <h2 className="section-title">Choose Your Analysis</h2>
          <div className="nav-grid">
            {navCards.map((card) => (
              <div
                key={card.id}
                className="nav-card"
                onClick={() => navigate(card.route)}
                role="button"
                tabIndex={0}
                onKeyPress={(e) => {
                  if (e.key === 'Enter') navigate(card.route);
                }}
              >
                <div className="nav-card-icon">{card.icon}</div>
                <h3 className="nav-card-title">{card.title}</h3>
                <p className="nav-card-description">{card.description}</p>
                <div className="nav-card-arrow">→</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="how-it-works-section">
        <div className="container">
          <h2 className="section-title">How It Works</h2>
          <div className="steps-container">
            {steps.map((step, index) => (
              <div key={index} className="step-item">
                <div className="step-number">{step.number}</div>
                <div className="step-content">
                  <h3 className="step-title">{step.title}</h3>
                  <p className="step-description">{step.description}</p>
                </div>
                {index < steps.length - 1 && <div className="step-connector">↓</div>}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="landing-footer">
        <div className="container">
          <p>Built with machine learning to help telecom companies reduce customer churn</p>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
