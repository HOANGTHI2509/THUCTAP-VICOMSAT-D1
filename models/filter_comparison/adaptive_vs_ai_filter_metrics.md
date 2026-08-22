# Adaptive Kalman vs AI Ensemble Filter

| Label | Count | Adaptive step | AI step | Adaptive abs residual | AI abs residual | Adaptive bias | AI bias | Adaptive move ratio | AI move ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| STABLE_JITTER | 9046 | 4.1553 | 4.1212 | 0.9489 | 1.1717 | 0.0101 | 0.0939 | 0.9616 | 0.9537 |
| CONSUMPTION | 691 | 39.1975 | 39.5687 | 4.9906 | 5.3889 | 2.9284 | 0.9813 | 0.9835 | 0.9928 |
| SLOSHING_NOISE | 3793 | 13.4527 | 12.0985 | 12.8782 | 15.1060 | 0.9718 | 3.0269 | 0.7982 | 0.7179 |
| REFUEL | 62 | 108.9803 | 133.9801 | 220.0581 | 64.0945 | -219.4290 | -64.0217 | 1.4680 | 1.8047 |
| DRAIN | 116 | 116.6591 | 118.7722 | 78.7017 | 49.0184 | 58.1862 | 17.4448 | 0.9693 | 0.9869 |

Notes:
- MeanAbsOutputStep: lower is smoother, especially for STABLE_JITTER/SLOSHING_NOISE/SPIKE.
- MeanAbsResidualToRaw: lower means closer to sensor, useful for REFUEL/DRAIN/CONSUMPTION.
- MeanSignedResidual: positive means output tends to stay above raw, often lagging on drops.
- MovementRatioVsRaw: output movement divided by raw movement; lower means stronger denoising.
