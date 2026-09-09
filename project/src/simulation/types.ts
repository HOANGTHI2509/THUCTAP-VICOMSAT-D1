export type NoiseKind = "spiky" | "hole" | "burst";

export interface Trip {
  id: string;
  vehicleId: string;
  vehicleName: string;
  tripName: string;
  noiseLabel: string;
  noiseKind: NoiseKind;
  durationSec: number;
  startTimeStr: string;
}

export interface TripPoint {
  t: number;
  speed: number;
  rawFuel: number;
  adaptiveKalman: number;
  traditionalKalman: number;
  mlKalman?: number;
  aiSmoothTracking?: number;
  smoothTracking?: number;
  cnn1DFuel?: number;
  noise: number;
  isSpike: boolean;
  vehicleId?: string;
  aiState?: string;
  lat?: number;
  lng?: number;
  address?: string;
}

export interface Vehicle {
  id: string;
  name: string;
  trips: Trip[];
}
