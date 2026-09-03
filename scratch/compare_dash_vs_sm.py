import sys, os
sys.path.insert(0, os.path.abspath("."))
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from src.core.filters.ai_state_filter import load_fuel_state_classifier, filter_with_ai_state
from src.core.filters.ai_enhanced_adaptive_realtime import filter_ai_enhanced_adaptive_realtime
from src.service.state_manager import StreamingStateManager

df = pd.read_csv('TienXuLy/21H-03221_processed.csv')
subset = df.iloc[200:280].copy()

# 1. Streamlit Dashboard
model, meta = load_fuel_state_classifier('models/fuel_state_classifier')
df_stream = filter_with_ai_state(subset, model=model, metadata=meta, mode='realtime')
clean_stream = filter_ai_enhanced_adaptive_realtime(df_stream, config={'source_col': 'FuelLevel'})
df_stream['Clean_Streamlit'] = clean_stream

# 2. StateManager (Realtime service)
sm = StreamingStateManager(model_dir='models/fuel_state_classifier')
sm_clean, sm_state = [], []
for idx, row in subset.iterrows():
    r = sm.process_point(
        vehicle_id='21H-03221',
        fuel_time=pd.to_datetime(row['FuelTime']),
        fuel_level=float(row['FuelLevel']),
        speed=float(row['Speed']),
    )
    sm_clean.append(r['clean_fuel_liters'])
    sm_state.append(r['ai_signal_state'])

df_stream['Clean_SM'] = sm_clean
df_stream['State_SM'] = sm_state

# 3. StateManager KHÔNG LOAD ĐƯỢC MODEL (như server hiện tại)
sm_no_model = StreamingStateManager(model_dir='models/rf_signal_state_causal_v3')
sm_nm_clean, sm_nm_state = [], []
for idx, row in subset.iterrows():
    r = sm_no_model.process_point(
        vehicle_id='21H-03221',
        fuel_time=pd.to_datetime(row['FuelTime']),
        fuel_level=float(row['FuelLevel']),
        speed=float(row['Speed']),
    )
    sm_nm_clean.append(r['clean_fuel_liters'])
    sm_nm_state.append(r['ai_signal_state'])

df_stream['Clean_NoModel'] = sm_nm_clean
df_stream['State_NoModel'] = sm_nm_state

for _, row in df_stream.iterrows():
    ft = str(row['FuelTime'])
    if '08:44' <= ft[-8:] <= '08:52':
        print(f"{ft} | Raw: {row['FuelLevel']:>6.1f} | Dash: {row['Clean_Streamlit']:>6.2f} ({row['AI_State']}) | SM_CoModel: {row['Clean_SM']:>6.2f} ({row['State_SM']}) | ServerHienTai: {row['Clean_NoModel']:>6.2f} ({row['State_NoModel']})")
