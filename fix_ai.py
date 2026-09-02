import re

file_path = 'd:/THUCTAP_VICOMSAT/src/core/filters/ai_state_filter.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

idx = content.find('    stable_recenter_count: int = 4')
if idx != -1:
    imports = """from __future__ import annotations

import json
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.core.filters.fuel_state_filter import build_fuel_state_profile
from src.core.signal_labels import to_legacy_label, to_signal_label

TRANSIENT_LABELS = {"TRANSIENT_NOISE", "TRANSIENT_UP_NOISE", "TRANSIENT_DOWN_NOISE", "TRANSIENT_CLUSTER_NOISE"}
TRANSIENT_STATE = "TRANSIENT_NOISE"

@dataclass
class AIStateFilterConfig:
    stable_alpha: float = 0.12
    stable_deadband_ratio: float = 0.25
    consumption_alpha: float = 0.58
    sloshing_alpha: float = 0.08
    refuel_alpha: float = 1.0
    drain_candidate_alpha: float = 0.45
    drain_alpha: float = 0.92
    unknown_alpha: float = 0.25
"""
    new_content = imports + content[idx:]
    with open(file_path, 'w', encoding='utf-8') as fw:
        fw.write(new_content)
    print('SUCCESS')
