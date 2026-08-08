import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd

# Load data
df = pd.read_csv('data/processed/CarFuelHistory_Processed_Car1.csv')
df['FuelTime'] = pd.to_datetime(df['FuelTime'])
df['MovementState'] = df['MovementState'].str.upper()

# Ensure we have the exact sliding window statistics for w=5
df['RollingMedian'] = df['FuelLevel'].rolling(5).median().fillna(0).round(2)
df['RollingStd'] = df['FuelLevel'].rolling(5).std().fillna(0).round(3)
df['FuelLevel'] = df['FuelLevel'].round(2)

# Find a spot where speed changes from 0 to something > 5 so we see a transition
df['StateChange'] = (df['MovementState'] != df['MovementState'].shift()).astype(int)
transition_idx = df[(df['MovementState'] == 'MOVING') & (df['StateChange'] == 1)].index

if len(transition_idx) > 0:
    start = transition_idx[10] - 4 # 4 rows before transition
else:
    start = 500

snippet = df.iloc[start:start+10][['FuelTime', 'Speed', 'MovementState', 'FuelLevel', 'RollingMedian', 'RollingStd']]
snippet['FuelTime'] = snippet['FuelTime'].dt.strftime('%H:%M:%S')

# Print markdown
print(snippet.to_markdown(index=False))
