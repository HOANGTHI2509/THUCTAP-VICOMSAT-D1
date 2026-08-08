import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd

df = pd.read_csv('data/processed/CarFuelHistory_Processed_Car1.csv')
# find segment change
segment_change = df[df['SegmentID'] != df['SegmentID'].shift(1)]
print(segment_change.head(5)[['FuelTime', 'FuelLevel', 'Speed', 'TimeGapMinutes', 'SegmentID', 'QualityReason']])
