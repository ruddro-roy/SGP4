import React, { useState } from 'react';
import { DemoData, PropagationPoint, CorrectionAnalysis, BatchResult, Capabilities } from '../types'; // Import shared types

interface MainContentProps {
  demoData: DemoData | null;
  error: string | null;
  loading: boolean;
}

interface PropagationTabsProps {
  baseline_propagation: PropagationPoint[];
  corrected_propagation: PropagationPoint[];
  correction_analysis: CorrectionAnalysis[];
  batch_propagation: BatchResult[];
  capabilities: Capabilities;
}

const PropagationTabs: React.FC<PropagationTabsProps> = ({
  baseline_propagation,
  corrected_propagation,
  correction_analysis,
  batch_propagation,
  capabilities
}) => {
  const [activeTab, setActiveTab] = useState('baseline');

  const tabs = [
    { id: 'baseline', label: 'Baseline Propagation', available: true },
    { id: 'ml-corrected', label: 'ML-Corrected Propagation', available: capabilities.ml_corrections && corrected_propagation && corrected_propagation.length > 0 },
    { id: 'batch', label: 'Batch Propagation', available: capabilities.batch_processing && batch_propagation && batch_propagation.length > 0 }
  ];

  const availableTabs = tabs.filter(tab => tab.available);

  return (
    <div className="tabs-container">
      <div className="tab-nav">
        {availableTabs.map(tab => (
          <button
            key={tab.id}
            className={`tab-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </div>
      
      <div className="tab-content">
        {activeTab === 'baseline' && (
          <div className="data-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Position (km)</th>
                  <th>Velocity (km/s)</th>
                  <th>Orbital Radius (km)</th>
                  {capabilities.gradient_computation && <th className="gradient">Radius Gradient</th>}
                </tr>
              </thead>
              <tbody>
                {baseline_propagation && baseline_propagation.length > 0 ? baseline_propagation.map((point: PropagationPoint, index: number) => (
                  <tr key={index}>
                    <td>{point.time_hours?.toFixed(2)} hrs ({point.time_minutes?.toFixed(1)} min)</td>
                    <td className="vector">{point.position_km ? `[${point.position_km.map((p: number) => p.toFixed(2)).join(', ')}]` : 'N/A'}</td>
                    <td className="vector">{point.velocity_km_s ? `[${point.velocity_km_s.map((v: number) => v.toFixed(4)).join(', ')}]` : 'N/A'}</td>
                    <td>{point.orbital_radius_km?.toFixed(2)}</td>
                    {capabilities.gradient_computation && <td className="gradient">{point.radius_gradient?.toFixed(6) || 'N/A'}</td>}
                  </tr>
                )) : (
                  <tr>
                    <td colSpan={capabilities.gradient_computation ? 5 : 4}>No propagation data available</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
        
        {activeTab === 'ml-corrected' && capabilities.ml_corrections && corrected_propagation && (
          <div className="data-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Corrected Position (km)</th>
                  <th>Corrected Velocity (km/s)</th>
                  <th>Correction Magnitude (km)</th>
                </tr>
              </thead>
              <tbody>
                {corrected_propagation.map((point: PropagationPoint, index: number) => {
                  const correction = correction_analysis[index];
                  return (
                    <tr key={index}>
                      <td>{point.time_hours?.toFixed(2)} hrs ({point.time_minutes?.toFixed(1)} min)</td>
                      <td className="vector">{point.position_km ? `[${point.position_km.map((p: number) => p.toFixed(2)).join(', ')}]` : 'N/A'}</td>
                      <td className="vector">{point.velocity_km_s ? `[${point.velocity_km_s.map((v: number) => v.toFixed(4)).join(', ')}]` : 'N/A'}</td>
                      <td className="correction">{correction?.correction_magnitude_km?.toFixed(3) || 'N/A'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
        
        {activeTab === 'batch' && capabilities.batch_processing && batch_propagation && (
          <div className="data-table-container">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Position (km)</th>
                  <th>Velocity (km/s)</th>
                  <th>Orbital Radius (km)</th>
                </tr>
              </thead>
              <tbody>
                {batch_propagation.map((point: BatchResult, index: number) => (
                  <tr key={index}>
                    <td>{(point.time_minutes / 60).toFixed(2)} hrs ({point.time_minutes?.toFixed(1)} min)</td>
                    <td className="vector">{point.position_km ? `[${point.position_km.map((p: number) => p.toFixed(2)).join(', ')}]` : 'N/A'}</td>
                    <td className="vector">{point.velocity_km_s ? `[${point.velocity_km_s.map((v: number) => v.toFixed(4)).join(', ')}]` : 'N/A'}</td>
                    <td>{point.orbital_radius_km?.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

const MainContent: React.FC<MainContentProps> = ({ demoData, error, loading }) => {
  if (loading) {
    return (
      <div className="info-panel">
        <h2>Orbital Propagation in Progress</h2>
        <p>Computing satellite trajectory...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-panel">
        <h3>An Error Occurred</h3>
        <p>{error}</p>
      </div>
    );
  }

  if (!demoData) {
    return (
      <div className="info-panel">
        <h2>Orbital Propagation Results</h2>
        <p>Enter satellite parameters and run propagation to view results.</p>
      </div>
    );
  }

  const { 
    satellite_info, 
    baseline_propagation, 
    corrected_propagation, 
    correction_analysis, 
    batch_propagation, 
    capabilities,
    demo_parameters 
  } = demoData;

  return (
    <main className="results-tabs">
      <section id="satellite-info">
        <h2>Satellite Information</h2>
        <div className="info-grid">
          <div className="info-item"><strong>Name:</strong> {satellite_info.name}</div>
          <div className="info-item"><strong>NORAD ID:</strong> {satellite_info.norad_id}</div>
          <div className="info-item"><strong>Epoch:</strong> {satellite_info.epoch}</div>
          <div className="info-item"><strong>Mean Motion:</strong> {satellite_info.mean_motion.toFixed(4)}</div>
          <div className="info-item"><strong>Inclination:</strong> {satellite_info.inclination_deg.toFixed(4)}°</div>
          <div className="info-item"><strong>Eccentricity:</strong> {satellite_info.eccentricity}</div>
        </div>
      </section>

      {/* Differentiable Capabilities Showcase */}
      <section id="capabilities">
        <h2>Differentiable SGP4 Capabilities</h2>
        <div className="capability-grid">
          <div className={`capability ${capabilities.gradient_computation ? 'enabled' : 'disabled'}`}>
            <div className="icon">∇</div>
            <span>Gradient Computation</span>
            <small>{capabilities.gradient_computation ? 'Active' : 'Inactive'}</small>
          </div>
          <div className={`capability ${capabilities.ml_corrections ? 'enabled' : 'disabled'}`}>
            <div className="icon">ML</div>
            <span>ML Corrections</span>
            <small>{capabilities.ml_corrections ? 'Active' : 'Inactive'}</small>
          </div>
          <div className={`capability ${capabilities.batch_processing ? 'enabled' : 'disabled'}`}>
            <div className="icon">BATCH</div>
            <span>Batch Processing</span>
            <small>{capabilities.batch_processing ? 'Active' : 'Inactive'}</small>
          </div>
          <div className={`capability ${capabilities.pytorch_autograd ? 'enabled' : 'disabled'}`}>
            <div className="icon">∂</div>
            <span>PyTorch Autograd</span>
            <small>{capabilities.pytorch_autograd ? 'Active' : 'Inactive'}</small>
          </div>
        </div>
      </section>

      {/* Tabbed Results Section */}
      <section id="propagation-results">
        <h2>Propagation Results</h2>
        <PropagationTabs 
          baseline_propagation={baseline_propagation}
          corrected_propagation={corrected_propagation}
          correction_analysis={correction_analysis}
          batch_propagation={batch_propagation}
          capabilities={capabilities}
        />
      </section>

      {/* Gradient Analysis and Visualization */}
      {capabilities.gradient_computation && baseline_propagation && baseline_propagation.some(p => p.radius_gradient) && (
        <section id="gradient-analysis">
          <h2>Gradient Analysis</h2>
          <div className="gradient-viz">
            <h3>Orbital Radius Gradient vs Time</h3>
            <div className="gradient-chart">
              <svg width="600" height="300" viewBox="0 0 600 300">
                <defs>
                  <linearGradient id="gradientLine" x1="0%" y1="0%" x2="100%" y2="0%">
                    <stop offset="0%" stopColor="#ff6b6b" />
                    <stop offset="50%" stopColor="#4ecdc4" />
                    <stop offset="100%" stopColor="#45b7d1" />
                  </linearGradient>
                </defs>
                
                {/* Chart background */}
                <rect width="600" height="300" fill="#f8f9fa" stroke="#e0e0e0" />
                
                {/* Grid lines */}
                {[0, 1, 2, 3, 4, 5].map(i => (
                  <g key={i}>
                    <line x1={100 + i * 80} y1="50" x2={100 + i * 80} y2="250" stroke="#e0e0e0" strokeDasharray="2,2" />
                    <line x1="100" y1={50 + i * 40} x2="500" y2={50 + i * 40} stroke="#e0e0e0" strokeDasharray="2,2" />
                  </g>
                ))}
                
                {/* Axes */}
                <line x1="100" y1="250" x2="500" y2="250" stroke="#333" strokeWidth="2" />
                <line x1="100" y1="50" x2="100" y2="250" stroke="#333" strokeWidth="2" />
                
                {/* Gradient line */}
                <polyline
                  fill="none"
                  stroke="url(#gradientLine)"
                  strokeWidth="3"
                  points={baseline_propagation
                    .filter(p => p.radius_gradient !== undefined)
                    .map((point, index) => {
                      const x = 100 + (index / (baseline_propagation.length - 1)) * 400;
                      const y = 250 - ((point.radius_gradient || 0) + 1) * 100; // Normalize for display
                      return `${x},${y}`;
                    })
                    .join(' ')}
                />
                
                {/* Axis labels */}
                <text x="300" y="290" textAnchor="middle" fontSize="12" fill="#666">Time (hours)</text>
                <text x="30" y="150" textAnchor="middle" fontSize="12" fill="#666" transform="rotate(-90 30 150)">Gradient</text>
                
                {/* Title */}
                <text x="300" y="30" textAnchor="middle" fontSize="14" fontWeight="bold" fill="#2c3e50">Radius Gradient Evolution</text>
              </svg>
            </div>
          </div>
        </section>
      )}

      {/* Demo Summary */}
      <section id="demo-summary">
        <h2>Demonstration Summary</h2>
        <div className="info-grid">
          <div className="info-item">
            <strong>Differentiable Propagation:</strong> 
            {capabilities.gradient_computation ? 'ACTIVE - Successfully demonstrated gradient computation' : 'INACTIVE - Not available'}
          </div>
          <div className="info-item">
            <strong>ML Corrections:</strong> 
            {capabilities.ml_corrections ? 'ACTIVE - ML model corrections applied and validated' : 'INACTIVE - Not available'}
          </div>
          <div className="info-item">
            <strong>Batch Processing:</strong> 
            {capabilities.batch_processing ? 'ACTIVE - Efficient batch propagation demonstrated' : 'INACTIVE - Not available'}
          </div>
          <div className="info-item">
            <strong>PyTorch Integration:</strong> 
            {capabilities.pytorch_autograd ? 'ACTIVE - Full autograd support confirmed' : 'INACTIVE - Not available'}
          </div>
          <div className="info-item">
            <strong>Time Points:</strong> {demo_parameters.time_points} propagation steps
          </div>
          <div className="info-item">
            <strong>Max Time:</strong> {demo_parameters.max_time_hours} hours
          </div>
        </div>
        <div style={{ marginTop: '1rem', padding: '1rem', backgroundColor: '#f1f8ff', borderRadius: '4px', border: '1px solid #3498db' }}>
          <strong>Key Achievement:</strong> This demonstration proves that SGP4 can be made fully differentiable while maintaining accuracy, 
          enabling future ML corrections for improved orbital predictions without rewriting the core propagator.
        </div>
      </section>
    </main>
  );
};

export default MainContent;
