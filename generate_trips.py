import json
import glob
import os
import pandas as pd

files = sorted(glob.glob('project/public/data_*.json'))

out = 'import { Vehicle, Trip } from "./types";\n\n'
out += 'export const VEHICLES: Vehicle[] = [\n'

for f in files:
    vid = os.path.basename(f).replace('data_', '').replace('.json', '')
    
    with open(f, 'r') as fp:
        data = json.load(fp)
    duration = data[-1]['t'] if data else 0
    
    csv_file = f"data/processed/CarFuelHistory_Processed_{vid}.csv"
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, nrows=2)
        start_time = df['FuelTime'].iloc[0]
        start_time = start_time.replace(' ', 'T') + "Z"
    else:
        start_time = "2026-07-29T00:00:00Z"
    
    out += f'  {{\n'
    out += f'    id: "{vid}",\n'
    out += f'    name: "Xe {vid}",\n'
    out += f'    trips: [\n'
    out += f'      {{\n'
    out += f'        id: "{vid}",\n'
    out += f'        vehicleId: "{vid}",\n'
    out += f'        vehicleName: "Xe {vid}",\n'
    out += f'        tripName: "Dữ liệu thực tế",\n'
    out += f'        noiseLabel: "Real Data",\n'
    out += f'        noiseKind: "spiky",\n'
    out += f'        durationSec: {duration},\n'
    out += f'        startTimeStr: "{start_time}",\n'
    out += f'      }}\n'
    out += f'    ]\n'
    out += f'  }},\n'

out += '];\n\n'
out += 'export const DEFAULT_TRIP_ID = VEHICLES[0]?.trips[0]?.id;\n\n'
out += 'export function findTrip(id: string): Trip | undefined {\n'
out += '  return VEHICLES.flatMap(v => v.trips).find(t => t.id === id);\n'
out += '}\n'

with open('project/src/simulation/trips.ts', 'w', encoding='utf-8') as fw:
    fw.write(out)
print("trips.ts generated successfully!")
