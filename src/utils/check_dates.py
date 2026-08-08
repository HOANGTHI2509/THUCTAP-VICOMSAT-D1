import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import pandas as pd
df = pd.read_excel('CarFuelHistory.xlsx')
print("Cars:", df['VehicleID'].unique())
print("Min Date:", df['FuelTime'].min())
print("Max Date:", df['FuelTime'].max())
