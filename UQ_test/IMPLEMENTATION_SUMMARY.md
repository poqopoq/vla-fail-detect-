# Implementation Summary: Per-Timestep logzpo for LIBERO Single-Demo HDF5

## ✅ Implementation Complete

Successfully implemented a complete pipeline for computing per-timestep logzpo scores and detecting failure anomalies in LIBERO single-demo HDF5 files.

### What Was Implemented

#### **Phase 1: Single-Demo HDF5 Data Loading**
- `load_single_demo()` — Loads one HDF5 file, extracts:
  - Observations: (T, 84) state vectors
  - Actions: (T, 7) action vectors  
  - Success label: final reward (0=failure, 1=success)
  - Trajectory length: T timesteps

- `load_multiple_demos()` — Loads multiple demos for calibration

#### **Phase 2: Per-Timestep logzpo Computation**
- `compute_logzpo_timeseries()` — **KEY FUNCTION**
  - Iterates over each timestep t in trajectory
  - Extracts observation at timestep t: shape (1, 84)
  - Calls `logpZO_UQ()` to compute logzpo score
  - Returns array of shape (T,) with per-timestep logzpo values
  
  **This is what you wanted**: logzpo as a timeseries, not a single scalar!

#### **Phase 3: Failure Detection via Threshold Band**
- `build_calibration_threshold()` — One-time calibration setup
  - Loads 25 successful demo trajectories
  - Computes per-timestep logzpo for each
  - Takes cumulative sum: cumsum_logzpo = np.cumsum(logzpo_timeseries)
  - Computes threshold at 90th percentile: `np.quantile(final_cumsums, 0.9)`
  - Returns threshold value (116015.08 for current calibration)

- `detect_anomalies()` — Detect when trajectory diverges
  - Checks where cumsum exceeds threshold
  - Returns: anomaly_detected (bool), first_idx (timestep), anomaly_percentage (%)

#### **Phase 4: Visualization**
- `plot_trajectory_with_detection()` — Generates 2-subplot figure:
  
  **Subplot 1**: Per-timestep logzpo
  - X: timesteps (scaled by 8)
  - Y: logzpo value per timestep
  - Color: blue (success) or red (failure)
  
  **Subplot 2**: Cumulative logzpo with threshold band
  - X: timesteps
  - Y: cumsum_logzpo (cumulative sum)
  - Blue shaded region: 0 to threshold (normal range)
  - Green dashed line: threshold value
  - Red triangle marker: first anomaly detection point (if triggered)
  - Statistics box: showing final cumsum, threshold, detection result

### Test Results

**Calibration Phase:**
```
Number of successful demos: 25
Mean cumsum logzpo: 95484.3328
Std cumsum logzpo: 16868.4651
Threshold (quantile 0.9): 116015.0818
```

**Single Demo Analysis:**
```
Loaded demo: 242 timesteps, 84-dim observations
Success label: 1 (Success)
Min logzpo: 203.3497
Max logzpo: 532.2134
Mean logzpo: 312.5874
Final cumsum: 75646.1613
Anomaly Detected: False ✓ (correct - successful trajectory stays below threshold)
```

### Key Features

✅ **Per-timestep computation** — logzpo is a timeseries, not scalar
✅ **Failure detection** — Peaks/divergences when trajectory exceeds threshold band
✅ **Visualization** — Clear plots showing anomaly markers and statistics
✅ **Scalable** — Process any single-demo HDF5 file independently
✅ **Modular** — Separate functions for each phase, easy to extend

### File Location

**Updated Script:** `/home/zhiyuanjia/FAIL-Detect/UQ_test/test_for_libero.py`

**Generated Plot:** `/home/zhiyuanjia/FAIL-Detect/UQ_test/test_libero_logzpo_plot.png`

### How to Use

```python
# 1. Load baseline model
baseline_model = elb.get_baseline_model('logpZO', 'square', 'diffusion').to(device)

# 2. Build calibration threshold (one-time)
threshold, stats = build_calibration_threshold(
    baseline_model, 
    '/path/to/multi_demo.hdf5',
    num_successful_demos=25
)

# 3. Load any single-demo HDF5
observations, actions, success_label, T = load_single_demo('/path/to/single_demo.hdf5')

# 4. Compute per-timestep logzpo
logzpo_timeseries = compute_logzpo_timeseries(baseline_model, observations)
cumsum_logzpo = np.cumsum(logzpo_timeseries)

# 5. Detect anomalies
anomaly_detected, first_idx, percentage = detect_anomalies(cumsum_logzpo, threshold)

# 6. Visualize
fig = plot_trajectory_with_detection(
    observations, logzpo_timeseries, cumsum_logzpo,
    threshold, success_label, anomaly_detected, first_idx
)
plt.show()
```

### Next Steps (Optional Enhancements)

1. **Batch Processing**: Loop over multiple single-demo files and generate comparison plots
2. **Threshold Persistence**: Save/load threshold to checkpoint to avoid recomputation
3. **Dynamic Threshold**: Upgrade to FunctionalPredictor from `timeseries_cp` for more sophisticated bands
4. **Per-Timestep Threshold**: Instead of just comparing final cumsum, use per-timestep comparison
5. **Failure Classification**: Use first_idx to classify failure type (early vs. late failure)

---

**Status**: ✅ **READY FOR USE**

Run the script with any single-demo HDF5 file to compute logzpo and detect failure peaks!
