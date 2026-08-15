import numpy as np
import pandas as pd
import os
import random

def generate_synthetic_segment(segment_id, n_points=300):
    # TimeGap
    time_gaps = np.ones(n_points) * 5.0
    long_gap_idx = random.sample(range(10, n_points-10), int(n_points * 0.02))
    time_gaps[long_gap_idx] = np.random.uniform(60, 120, len(long_gap_idx))
    
    # Movement State
    is_moving = np.zeros(n_points, dtype=bool)
    current_state = random.choice([True, False])
    state_duration = 0
    for i in range(n_points):
        if state_duration <= 0:
            current_state = not current_state
            state_duration = random.randint(15, 60)
        is_moving[i] = current_state
        state_duration -= 1

    speeds = np.zeros(n_points)
    for i in range(n_points):
        if is_moving[i]:
            # Add some continuity to speed
            speeds[i] = speeds[i-1] + np.random.uniform(-5, 5) if i>0 and is_moving[i-1] else np.random.uniform(20, 50)
            speeds[i] = np.clip(speeds[i], 5, 80)
    
    # Accelerations
    accels = np.append([0.0], np.diff(speeds) / (time_gaps[1:] * 60))
    
    # Clean Fuel
    clean_fuel = np.zeros(n_points)
    clean_fuel[0] = random.uniform(50, 400)
    
    # Track events for auditing
    event_labels = ["Normal"] * n_points
    
    i = 1
    while i < n_points:
        # Consumption based on speed
        if speeds[i] == 0:
            consumption = np.random.uniform(-0.1, 0.0) # idle
        elif speeds[i] < 30:
            consumption = np.random.uniform(-0.8, -0.2) # slow
        else:
            consumption = np.random.uniform(-1.5, -0.8) # fast
            
        delta = consumption
        
        # Determine if Refuel or Drain starts
        event_started = False
        if not is_moving[i] and random.random() < 0.005 and i < n_points - 5:
            duration = random.randint(2, 5) # Multi-step event
            is_refuel = random.random() < 0.6
            
            total_change = random.uniform(20, 150) if is_refuel else -random.uniform(15, 50)
            
            # Randomize the shape of the change instead of strictly linear
            # Generate random proportions that sum to 1.0
            proportions = np.random.uniform(0.1, 1.0, duration)
            proportions /= proportions.sum()
            
            for j in range(duration):
                if i+j < n_points:
                    step_change = total_change * proportions[j]
                    clean_fuel[i+j] = clean_fuel[i+j-1] + step_change + np.random.uniform(-0.5, 0.5)
                    event_labels[i+j] = "Refuel" if is_refuel else "Drain"
            
            i += duration
            continue
            
        clean_fuel[i] = clean_fuel[i-1] + delta
        clean_fuel[i] = max(0, min(600, clean_fuel[i]))
        i += 1

    # Generate Noise
    noisy_fuel = clean_fuel.copy()
    
    # Sloshing logic
    phase = 0.0
    for i in range(n_points):
        if is_moving[i]:
            # Amplitude based on acceleration and speed
            amp = 1.5
            if abs(accels[i]) > 0.03:
                amp = random.uniform(3.0, 5.0) # High acceleration -> large slosh
            elif speeds[i] > 40:
                amp = random.uniform(2.0, 3.5) # High speed -> medium slosh
                
            phase += random.uniform(0.5, 1.5)
            slosh = amp * np.sin(phase) + np.random.normal(0, 0.5)
            noisy_fuel[i] += slosh
            if event_labels[i] == "Normal" and amp > 2.5:
                event_labels[i] = "Sloshing"
        else:
            # Idle noise
            noisy_fuel[i] += np.random.normal(0, 0.2)
            
    # Spikes logic
    i = 5
    while i < n_points - 5:
        if random.random() < 0.01 and event_labels[i] == "Normal":
            spike_type = random.choice(["single_pos", "single_neg", "multi"])
            
            if spike_type == "single_pos":
                noisy_fuel[i] += random.uniform(20, 60)
                event_labels[i] = "Spike"
                i += 2
            elif spike_type == "single_neg":
                noisy_fuel[i] -= random.uniform(20, 60)
                event_labels[i] = "Spike"
                i += 2
            elif spike_type == "multi":
                duration = random.randint(2, 4)
                amp = random.uniform(20, 60) * random.choice([1, -1])
                for j in range(duration):
                    noisy_fuel[i+j] += amp + np.random.normal(0, 2)
                    event_labels[i+j] = "Spike"
                i += duration + 1
        else:
            i += 1
            
    # Compile DataFrame
    df = pd.DataFrame({
        'VehicleID': f"Synthetic_{segment_id}",
        'SegmentID': segment_id,
        'TimeGapMinutes': time_gaps,
        'Speed': speeds,
        'Acceleration': accels,
        'MovementState': ['Moving' if m else 'Stopped' for m in is_moving],
        'CleanFuel': clean_fuel,
        'NoisyFuel': noisy_fuel,
        'EventLabel': event_labels
    })
    
    df['RollingStd'] = df['NoisyFuel'].rolling(window=5, min_periods=1).std().fillna(0)
    df['FuelResidual'] = df['NoisyFuel'].diff().fillna(0)
    
    return df

def generate_synthetic_dataset(num_segments=100, output_path='data/gru_dataset/synthetic_dataset.csv'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_dfs = []
    for i in range(num_segments):
        all_dfs.append(generate_synthetic_segment(i))
        
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(output_path, index=False)
    print(f"Generated {len(final_df)} synthetic points and saved to {output_path}")

if __name__ == '__main__':
    np.random.seed(42)
    random.seed(42)
    generate_synthetic_dataset()
