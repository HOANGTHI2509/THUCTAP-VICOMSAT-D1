import pandas as pd

def apply_median_filter(df: pd.DataFrame, window_size: int = 10, source_col: str = "FuelLevel") -> list[float]:
    """
    Applies a simple moving median filter with a given window size (default N=10).
    """
    if source_col not in df.columns:
        if "ShapeCleanFuel" in df.columns:
            source_col = "ShapeCleanFuel"
        elif "ProfileCleanFuel" in df.columns:
            source_col = "ProfileCleanFuel"
        else:
            source_col = "FuelLevel"
            
    fuels = pd.to_numeric(df[source_col], errors="coerce")
    
    # Use pandas rolling median with min_periods=1
    median_filtered = fuels.rolling(window=window_size, min_periods=1, center=False).median()
    
    return median_filtered.to_list()
