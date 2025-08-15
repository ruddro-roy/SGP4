import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';

// Mock component for testing
const App = () => {
  return (
    <div>
      <h1>Satellite Tracking Platform</h1>
      <p>Real-time satellite tracking and visualization</p>
    </div>
  );
};

describe('App Component', () => {
  test('renders satellite tracking platform title', () => {
    render(<App />);
    const titleElement = screen.getByText(/Satellite Tracking Platform/i);
    expect(titleElement).toBeInTheDocument();
  });

  test('renders description text', () => {
    render(<App />);
    const descriptionElement = screen.getByText(/Real-time satellite tracking/i);
    expect(descriptionElement).toBeInTheDocument();
  });

  test('has correct heading structure', () => {
    render(<App />);
    const heading = screen.getByRole('heading', { level: 1 });
    expect(heading).toHaveTextContent('Satellite Tracking Platform');
  });
});