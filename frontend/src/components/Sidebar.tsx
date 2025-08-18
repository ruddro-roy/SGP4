import React from 'react';

interface SidebarProps {
  noradId: string;
  setNoradId: (id: string) => void;
  timePoints: string;
  setTimePoints: (points: string) => void;
  maxTimeHours: string;
  setMaxTimeHours: (hours: string) => void;
  runDemo: () => void;
  loading: boolean;
}

const Sidebar: React.FC<SidebarProps> = ({ noradId, setNoradId, timePoints, setTimePoints, maxTimeHours, setMaxTimeHours, runDemo, loading }) => (
  <aside className="control-panel">
    <div className="sidebar-header">
      <h1>Differentiable SGP4</h1>
      <p>Orbital Propagation Parameters</p>
    </div>
    <div className="control-group">
      <label htmlFor="noradId">Satellite NORAD ID</label>
      <input
        id="noradId"
        type="text"
        value={noradId}
        onChange={(e) => setNoradId(e.target.value)}
        placeholder="e.g., 25544 for ISS"
      />
    </div>
    <div className="control-group">
      <label htmlFor="timePoints">Time Points</label>
      <input
        id="timePoints"
        type="number"
        value={timePoints}
        onChange={(e) => setTimePoints(e.target.value)}
        placeholder="Number of propagation steps"
      />
    </div>
    <div className="control-group">
      <label htmlFor="maxTimeHours">Max Time (Hours)</label>
      <input
        id="maxTimeHours"
        type="number"
        value={maxTimeHours}
        onChange={(e) => setMaxTimeHours(e.target.value)}
        placeholder="Total propagation duration"
      />
    </div>
    <button className="run-demo-btn" onClick={runDemo} disabled={loading}>
      {loading ? 'Propagating...' : 'Run Propagation'}
    </button>
  </aside>
);

export default Sidebar;
