import os
import sys
sys.path.insert(0, os.path.abspath("."))
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from src.core.filters.ai_state_filter import load_fuel_state_classifier, filter_with_ai_state
from src.core.filters.ai_enhanced_adaptive_realtime import filter_ai_enhanced_adaptive_realtime as new_filter
from scratch.candidate_baseline_filter import filter_ai_enhanced_adaptive_realtime as cand_filter

df = pd.read_csv("TienXuLy/21C-04064_processed.csv")
model, metadata = load_fuel_state_classifier("models/fuel_state_classifier")
df_ai = filter_with_ai_state(df, model=model, metadata=metadata, mode="realtime")

t_new, t_cand = [], []
new_filter(df_ai, config={"trace_collector": t_new})
cand_filter(df_ai, config={"trace_collector": t_cand})

df_tn = pd.DataFrame(t_new)
df_tc = pd.DataFrame(t_cand)

idx = df_ai[df_ai["FuelTime"].str.contains("2026-08-11 11:00")].index[0]
cols = ["fuel_time", "raw_z", "x_after", "branch_selected", "update_mode", "R_effective", "Q_effective", "K"]

print("=== BAN MOI (CANDIDATE + RECOVERY) ===")
print(df_tn.iloc[idx-3:idx+15][cols].to_string())
print("\n=== BASELINE (CANDIDATE ONLY) ===")
print(df_tc.iloc[idx-3:idx+15][cols].to_string())

