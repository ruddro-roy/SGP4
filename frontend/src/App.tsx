import React, { useState } from 'react';
import './App.css';
import Sidebar from './components/Sidebar.tsx';
import MainContent from './components/MainContent.tsx';
import { DemoData } from './types';

function App() {
  const [noradId, setNoradId] = useState('25544'); // Default to ISS
  const [timePoints, setTimePoints] = useState('100');
  const [maxTimeHours, setMaxTimeHours] = useState('6');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [demoData, setDemoData] = useState<DemoData | null>(null);

  const runDemo = async () => {
    setLoading(true);
    setError(null);
    setDemoData(null);
    try {
      const params = new URLSearchParams({
        norad_id: noradId,
        time_points: timePoints,
        max_time_hours: maxTimeHours,
      });
      const response = await fetch(`http://127.0.0.1:5001/differentiable-sgp4/demo?${params}`);
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setDemoData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Differentiable SGP4</h1>
        <p>Orbital Propagation Analysis</p>
      </header>
      <div className="main-container">
        <Sidebar
          noradId={noradId} setNoradId={setNoradId}
          timePoints={timePoints} setTimePoints={setTimePoints}
          maxTimeHours={maxTimeHours} setMaxTimeHours={setMaxTimeHours}
          runDemo={runDemo} loading={loading}
        />
        <MainContent demoData={demoData} error={error} loading={loading} />
      </div>
    </div>
  );
}

export default App;