import type { NoiseKind, Trip, TripPoint } from "./types";

const STEP_SEC = 1;

interface EventSpec {
  startSec: number;
  endSec: number;
  amp: number;
  kind: "spike" | "hole" | "burst";
}

function planEvents(trip: Trip, seed: number): EventSpec[] {
  const d = trip.durationSec;
  const events: EventSpec[] = [];
  const rand = mulberry32(seed);

  const count =
    trip.noiseKind === "spiky" ? 28 : trip.noiseKind === "hole" ? 10 : 8;

  for (let i = 0; i < count; i++) {
    const start = Math.floor(rand() * (d - 120)) + 20;
    if (trip.noiseKind === "spiky") {
      events.push({ startSec: start, endSec: start + 2, amp: 0, kind: "spike" });
    } else if (trip.noiseKind === "hole") {
      const len = 3 + Math.floor(rand() * 5);
      const amp = -6 - rand() * 8;
      events.push({ startSec: start, endSec: start + len, amp, kind: "hole" });
    } else {
      const len = 8 + Math.floor(rand() * 20);
      const amp = 10 + rand() * 14;
      events.push({ startSec: start, endSec: start + len, amp, kind: "burst" });
    }
  }
  events.sort((a, b) => a.startSec - b.startSec);
  return events;
}

function baseSpeed(t: number, duration: number): number {
  const phase = t / duration;
  if (phase < 0.08) return (phase / 0.08) * 55;
  if (phase > 0.92) return (1 - (phase - 0.92) / 0.08) * 55;

  const stopCycle = Math.sin(phase * 13) > 0.72;
  const longStop = Math.sin(phase * 4.6 + 0.8) > 0.9;
  if (longStop) return 0;
  if (stopCycle) return 0;

  const cruise = 48 + Math.sin(phase * 14) * 12 + Math.sin(phase * 37) * 4;
  const accelerationWave = Math.sin(phase * 9 - 0.7) * 18;
  return Math.max(0, cruise + accelerationWave);
}

function hashStr(s: string): number {
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return h >>> 0;
}

function mulberry32(a: number): () => number {
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const SPIKE_THRESHOLD = 3.5;

interface TripCache {
  seed: number;
  events: EventSpec[];
  rand: () => number;
}

const cacheMap = new Map<string, TripCache>();

function getCache(trip: Trip): TripCache {
  let c = cacheMap.get(trip.id);
  if (!c) {
    const seed = hashStr(trip.id);
    c = { seed, events: planEvents(trip, seed), rand: mulberry32(seed ^ 0x9e3779b9) };
    cacheMap.set(trip.id, c);
  }
  return c;
}

export function samplePoint(trip: Trip, t: number, prev: TripPoint | null): TripPoint {
  const cache = getCache(trip);
  const { events, rand } = cache;

  const previousSpeed = prev?.speed ?? 0;
  const speed = Math.max(0, baseSpeed(t, trip.durationSec));
  const speedChange = speed - previousSpeed;
  const brakingNoise = speedChange < -16 ? -Math.min(7, Math.abs(speedChange) * 0.22) : 0;
  const baseFuelDrain = 0.012 + (speed / 90) * 0.03;

  const baseFuel = prev ? prev.adaptiveKalman - baseFuelDrain : 78;

  let noise = (rand() - 0.5) * 1.6;
  if (trip.noiseKind === "spiky") {
    noise += (rand() - 0.5) * 2.2;
  }

  let eventFlag = false;
  for (const e of events) {
    if (t < e.startSec || t > e.endSec) continue;
    if (e.kind === "spike") {
      noise += (rand() - 0.5) * 22;
      eventFlag = true;
    } else if (e.kind === "hole") {
      noise += e.amp + (rand() - 0.5) * 3;
      eventFlag = true;
    } else if (e.kind === "burst") {
      noise += Math.sin((t - e.startSec) * 3) * e.amp + (rand() - 0.5) * 6;
      eventFlag = true;
    }
  }

  const rawFuel = Math.max(0, Math.min(100, baseFuel + noise));

  const filtered = prev ? prev.adaptiveKalman - baseFuelDrain : baseFuel;

  return {
    t,
    speed,
    rawFuel,
    adaptiveKalman: Math.max(0, filtered),
    traditionalKalman: Math.max(0, filtered),
    noise: rawFuel - filtered,
    isSpike: Math.abs(noise) > SPIKE_THRESHOLD || eventFlag,
  };
}

export { STEP_SEC };

export type { NoiseKind };
