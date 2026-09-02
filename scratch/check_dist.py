import numpy as np

for name in ["train", "val"]:
    try:
        y = np.load(f"data/real_dataset/windows/y_{name}_N10.npy").reshape(-1)
        
        print(f"\n===== {name.upper()} =====")
        print(f"min       = {y.min():.10f}")
        print(f"max       = {y.max():.10f}")
        print(f"mean      = {y.mean():.10f}")
        print(f"std       = {y.std():.10f}")
        print(f"P95 |y|   = {np.percentile(np.abs(y), 95):.10f}")
        print(f"P99 |y|   = {np.percentile(np.abs(y), 99):.10f}")
        print(f"Max |y|   = {np.max(np.abs(y)):.10f}")
        
        print(f"|y| < .001 = {np.mean(np.abs(y) < .001)*100:.2f}%")
        print(f"|y| < .005 = {np.mean(np.abs(y) < .005)*100:.2f}%")
        print(f"|y| < .01  = {np.mean(np.abs(y) < .01)*100:.2f}%")
    except Exception as e:
        print(f"Could not process {name}: {e}")
