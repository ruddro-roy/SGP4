export interface SatelliteInfo {
  name: string;
  norad_id: number;
  epoch: string;
  mean_motion: number;
  inclination_deg: number;
  eccentricity: number;
}

export interface PropagationPoint {
  time_minutes: number;
  time_hours: number;
  position_km: [number, number, number];
  velocity_km_s: [number, number, number];
  orbital_radius_km: number;
  altitude_km?: number;
  radius_gradient?: number;
}

export interface CorrectionAnalysis {
  time_minutes: number;
  correction_magnitude_km: number;
  correction_vector_km: [number, number, number];
}

export interface BatchResult {
  time_minutes: number;
  position_km: [number, number, number];
  velocity_km_s: [number, number, number];
  orbital_radius_km: number;
}

export interface Capabilities {
  gradient_computation: boolean;
  ml_corrections: boolean;
  batch_processing: boolean;
  pytorch_autograd: boolean;
}

export interface DemoData {
  satellite_info: SatelliteInfo;
  baseline_propagation: PropagationPoint[];
  corrected_propagation: PropagationPoint[];
  correction_analysis: CorrectionAnalysis[];
  batch_propagation: BatchResult[];
  capabilities: Capabilities;
  demo_parameters: {
    time_points: number;
    max_time_hours: number;
    differentiable: boolean;
    ml_corrections_enabled: boolean;
  };
  timestamp: string;
}
