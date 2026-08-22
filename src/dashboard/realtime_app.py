from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn
import pandas as pd
import io
import sqlite3
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataPoint(BaseModel):
    time: str 
    raw: float
    kalman: float
    adaptive: float
    ai_enhanced: float
    vehicle_id: Optional[str] = "UNKNOWN"
    speed: Optional[float] = 0.0
    lat: Optional[float] = 0.0
    lng: Optional[float] = 0.0
    address: Optional[str] = ""

DB_PATH = "fuel_data.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS fuel_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            vehicle_id TEXT,
            raw_fuel REAL,
            kalman REAL,
            adaptive REAL,
            ai_enhanced REAL,
            speed REAL,
            lat REAL,
            lng REAL,
            address TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

receive_enabled = True

@app.post("/api/toggle")
async def toggle_api(enabled: bool):
    global receive_enabled
    receive_enabled = enabled
    return {"status": "ok", "enabled": receive_enabled}

@app.post("/api/clear")
async def clear_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM fuel_records')
    conn.commit()
    conn.close()
    return {"status": "ok", "message": "All data cleared"}

@app.post("/api/push")
async def push_data(point: DataPoint):
    if not receive_enabled:
        return {"status": "ignored", "reason": "receiving paused"}
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Thêm cột nếu chưa có (để không phải reset database file)
    try:
        c.execute('ALTER TABLE fuel_records ADD COLUMN vehicle_id TEXT')
    except:
        pass
        
    c.execute('''
        INSERT INTO fuel_records (timestamp, vehicle_id, raw_fuel, kalman, adaptive, ai_enhanced, speed, lat, lng, address)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (point.time, point.vehicle_id, point.raw, point.kalman, point.adaptive, point.ai_enhanced, point.speed, point.lat, point.lng, point.address))
    conn.commit()
    conn.close()
    return {"status": "ok"}

@app.get("/api/data")
async def get_latest_data(limit: int = 500):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"SELECT * FROM fuel_records ORDER BY timestamp DESC LIMIT {limit}", conn)
    conn.close()
    
    if df.empty:
        return []
    
    # Reverse to chronological order
    df = df.iloc[::-1].reset_index(drop=True)
    
    res = []
    for _, row in df.iterrows():
        res.append({
            "time": row["timestamp"],
            "vehicle_id": row.get("vehicle_id", "UNKNOWN"),
            "raw": row["raw_fuel"],
            "kalman": row["kalman"],
            "adaptive": row["adaptive"],
            "ai_enhanced": row["ai_enhanced"],
            "speed": row.get("speed", 0),
            "lat": row.get("lat", 0),
            "lng": row.get("lng", 0),
            "address": row.get("address", "")
        })
    return res

@app.get("/api/history")
async def get_history(start_date: str, end_date: str):
    conn = sqlite3.connect(DB_PATH)
    if len(end_date) == 10:
        end_date += " 23:59:59"
    query = "SELECT * FROM fuel_records WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC"
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    conn.close()
    
    # Format dates to be prettier if needed, or just return as is
    return df.to_dict(orient="records")

@app.get("/api/export_history")
async def export_history(start_date: str, end_date: str):
    conn = sqlite3.connect(DB_PATH)
    if len(end_date) == 10:
        end_date += " 23:59:59"
    query = "SELECT * FROM fuel_records WHERE timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC"
    df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    conn.close()
    
    if df.empty:
        return {"error": "No data found for the selected date range"}
        
    # Drop the internal 'id' column
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        
    # Rename columns to match the pipeline and be user-friendly
    df = df.rename(columns={
        'timestamp': 'Thời gian (FuelTime)',
        'vehicle_id': 'Biển số xe',
        'raw_fuel': 'Nhiên liệu chưa làm sạch',
        'kalman': 'Nhiên liệu lọc Kalman',
        'adaptive': 'Nhiên liệu Adaptive Kalman',
        'ai_enhanced': 'Nhiên liệu đã làm sạch qua thuật toán AI',
        'speed': 'Speed',
        'lat': 'Vĩ độ',
        'lng': 'Kinh độ',
        'address': 'Địa chỉ'
    })
    
    # Reorder columns as user explicitly requested
    columns_order = [
        'Biển số xe',
        'Thời gian (FuelTime)', 
        'Nhiên liệu chưa làm sạch', 
        'Nhiên liệu đã làm sạch qua thuật toán AI', 
        'Kinh độ', 
        'Vĩ độ', 
        'Địa chỉ', 
        'Speed',
        'Nhiên liệu lọc Kalman',
        'Nhiên liệu Adaptive Kalman'
    ]
    # Only keep columns that exist in the dataframe (in case some are missing)
    columns_order = [c for c in columns_order if c in df.columns]
    df = df[columns_order]
        
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Fuel History')
    output.seek(0)
    
    headers = {
        'Content-Disposition': f'attachment; filename="fuel_data_{start_date[:10]}_to_{end_date[:10]}.xlsx"',
        'Access-Control-Expose-Headers': 'Content-Disposition'
    }
    return StreamingResponse(output, headers=headers, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

@app.get("/")
async def root():
    return HTMLResponse("<h2>🚀 API Server is running. The frontend is now the React App on port 5173.</h2>")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
