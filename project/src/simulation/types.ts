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
  filteredFuel: number;
  noise: number;
  isSpike: boolean;
}

export interface Vehicle {
  id: string;
  name: string;
  trips: Trip[];
}
