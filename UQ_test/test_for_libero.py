"""
Per-timestep logzpo calculation and failure detection for LIBERO single-demo HDF5 files.

Workflow:
  Phase 1: Load single-demo HDF5 file
  Phase 2: Compute logzpo per timestep
  Phase 3: Setup failure detection threshold
  Phase 4: Visualize with anomaly detection
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import h5py
import eval_load_baseline as elb
from tqdm import tqdm
import argparse
import sys

# ============================================================================
# Configuration
# ============================================================================
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
task_name = 'square'
policy_type = 'diffusion'

# Default paths (can be overridden via command-line arguments)
DEFAULT_DEMO_PATH = '/home/zhiyuanjia/lerobot/test_traj/trajectories/libero_10_7/episode_001.hdf5'
DEFAULT_CALIBRATION_PATH = '/home/zhiyuanjia/LIBERO/datasets/libero_10/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5'

# ============================================================================
# Phase 1: Data Loading Functions
# ============================================================================

def load_single_demo(hdf5_path):
    """Load a single-demo HDF5 file.
    
    Returns:
        observations: (T, 84) array of states
        actions: (T, 7) array of actions
        success_label: scalar (0 or 1) - final reward
        trajectory_length: scalar T
    """
    with h5py.File(hdf5_path, 'r') as f:
        # Assuming first (and only) demo in file
        demo_key = list(f['data'].keys())[0]
        demo = f['data'][demo_key]
        
        observations = torch.tensor(demo['states'][:], dtype=torch.float32)
        actions = torch.tensor(demo['actions'][:], dtype=torch.float32)
        success_label = int(demo['rewards'][-1])
        trajectory_length = observations.shape[0]
    
    return observations, actions, success_label, trajectory_length


def load_multiple_demos(hdf5_path, max_demos=None):
    """Load multiple demos from an HDF5 file.
    
    Returns:
        List of (observations, success_label) tuples
    """
    demos_data = []
    with h5py.File(hdf5_path, 'r') as f:
        demo_keys = sorted(list(f['data'].keys()))
        if max_demos:
            demo_keys = demo_keys[:max_demos]
        
        for demo_key in demo_keys:
            demo = f['data'][demo_key]
            observations = torch.tensor(demo['states'][:], dtype=torch.float32)
            success_label = int(demo['rewards'][-1])
            demos_data.append((observations, success_label))
    
    return demos_data


# ============================================================================
# Phase 2: Per-timestep logzpo Computation
# ============================================================================

def compute_logzpo_timeseries(baseline_model, observations, task_name='square'):
    """Compute logzpo for each timestep in a trajectory.
    
    Args:
        baseline_model: Loaded logpZO model
        observations: (T, 84) tensor
        task_name: Task identifier
    
    Returns:
        logzpo_timeseries: (T,) array of per-timestep logzpo scores
    """
    T = observations.shape[0]
    logzpo_timeseries = np.zeros(T)
    
    with torch.no_grad():
        for t in range(T):
            # Extract single observation at timestep t
            obs_t = observations[t:t+1].to(device)  # Shape: (1, 84)
            
            # Compute logzpo using baseline model
            logzpo_val = elb.logpZO_UQ(baseline_model, obs_t, 
                                       action_pred=None, 
                                       task_name=task_name)
            
            # Extract scalar value and store
            logzpo_timeseries[t] = logzpo_val.cpu().numpy()[0]
    
    return logzpo_timeseries


# ============================================================================
# Phase 3: Failure Detection via Threshold Band
# ============================================================================

def build_calibration_threshold(baseline_model, calibration_hdf5_path, 
                               task_name='square', num_successful_demos=25, alpha=0.1):
    """Build per-timestep threshold from successful calibration trajectories.
    
    CPband approach: Learn threshold at each timestep from successful demos.
    If logzpo[i] exceeds threshold[i], flag as failure.
    
    Args:
        baseline_model: Loaded logpZO model
        calibration_hdf5_path: Path to HDF5 with multiple demos
        task_name: Task identifier
        num_successful_demos: Number of demos to use for calibration
        alpha: Quantile parameter (1-alpha for threshold)
    
    Returns:
        threshold_per_timestep: Array of thresholds (one per typical timestep)
        calibration_stats: Dict with calibration info
    """
    print(f"\n{'='*60}")
    print(f"Building per-timestep threshold from {calibration_hdf5_path}")
    print(f"{'='*60}")
    
    demos_data = load_multiple_demos(calibration_hdf5_path, max_demos=num_successful_demos)
    
    all_logzpo_trajectories = []
    for i, (observations, success_label) in enumerate(tqdm(demos_data, desc="Computing logzpo for calibration")):
        if success_label == 1:  # Only use successful demos
            observations = observations.to(device)
            logzpo_traj = compute_logzpo_timeseries(baseline_model, observations, task_name)
            all_logzpo_trajectories.append(logzpo_traj)
    
    # Compute per-timestep threshold using the median trajectory length
    median_length = int(np.median([len(traj) for traj in all_logzpo_trajectories]))
    
    # Pad/interpolate trajectories to same length for per-timestep quantile
    all_logzpo_padded = []
    for traj in all_logzpo_trajectories:
        if len(traj) >= median_length:
            all_logzpo_padded.append(traj[:median_length])
        else:
            # Pad with last value
            padded = np.pad(traj, (0, median_length - len(traj)), mode='edge')
            all_logzpo_padded.append(padded)
    
    all_logzpo_padded = np.array(all_logzpo_padded)
    
    # Per-timestep threshold: quantile at each timestep
    threshold_per_timestep = np.quantile(all_logzpo_padded, 1 - alpha, axis=0)
    
    calibration_stats = {
        'num_demos': len(all_logzpo_trajectories),
        'median_trajectory_length': median_length,
        'mean_logzpo': np.mean(all_logzpo_padded),
        'std_logzpo': np.std(all_logzpo_padded),
        'threshold_mean': np.mean(threshold_per_timestep),
        'threshold_max': np.max(threshold_per_timestep),
        'alpha': alpha,
        'quantile': 1 - alpha
    }
    
    print(f"\nCalibration Results:")
    print(f"  Number of successful demos: {calibration_stats['num_demos']}")
    print(f"  Median trajectory length: {calibration_stats['median_trajectory_length']}")
    print(f"  Mean logzpo across all timesteps: {calibration_stats['mean_logzpo']:.4f}")
    print(f"  Threshold mean: {calibration_stats['threshold_mean']:.4f}")
    print(f"  Threshold max: {calibration_stats['threshold_max']:.4f}")
    
    return threshold_per_timestep, calibration_stats


def detect_anomalies(logzpo_timeseries, threshold_per_timestep):
    """Detect failure using per-timestep CPband comparison.
    
    CPband Logic: For each timestep, if logzpo[i] > threshold[i], flag as failure.
    Return first timestep where this happens.
    
    Args:
        logzpo_timeseries: (T,) array of per-timestep logzpo scores
        threshold_per_timestep: (T_cal,) array of per-timestep thresholds
    
    Returns:
        anomaly_detected: Boolean
        first_idx: Timestep index of first anomaly (or -1 if none)
        anomaly_percentage: Percentage through episode (0-100)
    """
    # Handle different trajectory lengths
    T = len(logzpo_timeseries)
    T_threshold = len(threshold_per_timestep)
    
    # Compare logzpo against threshold at each timestep
    exceedance_mask = np.zeros(T, dtype=bool)
    for t in range(min(T, T_threshold)):
        if logzpo_timeseries[t] > threshold_per_timestep[t]:
            exceedance_mask[t] = True
    
    # Find first exceeding timestep
    if np.any(exceedance_mask):
        first_idx = np.argmax(exceedance_mask)
        anomaly_detected = True
        anomaly_percentage = 100.0 * first_idx / len(logzpo_timeseries)
    else:
        first_idx = -1
        anomaly_detected = False
        anomaly_percentage = -1
    
    return anomaly_detected, first_idx, anomaly_percentage


# ============================================================================
# Phase 4: Visualization
# ============================================================================

def plot_trajectory_with_detection(logzpo_timeseries, threshold_per_timestep,
                                  success_label, anomaly_detected, first_idx,
                                  trajectory_name='Single Demo'):
    """Plot per-timestep logzpo with CPband threshold detection.
    
    Args:
        logzpo_timeseries: (T,) array of per-timestep logzpo
        threshold_per_timestep: (T_cal,) array of per-timestep thresholds
        success_label: 0 or 1
        anomaly_detected: Boolean
        first_idx: Timestep of first anomaly (or -1)
        trajectory_name: Title for plot
    """
    T = len(logzpo_timeseries)
    T_threshold = len(threshold_per_timestep)
    timesteps = np.arange(T) * 8  # Scale by 8 to match plot_with_CP_band.py
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    color = 'blue' if success_label == 1 else 'red'
    label_text = 'Success' if success_label == 1 else 'Failure'
    
    # Plot per-timestep logzpo
    ax.plot(timesteps, logzpo_timeseries, color=color, linewidth=2.5, label=label_text, zorder=3)
    
    # Plot per-timestep threshold (align with logzpo length)
    threshold_plot = threshold_per_timestep[:min(T, T_threshold)]
    if len(threshold_plot) < T:
        # Extend threshold if trajectory is longer
        threshold_plot = np.pad(threshold_plot, (0, T - len(threshold_plot)), mode='edge')
    
    timesteps_threshold = np.arange(len(threshold_plot)) * 8
    ax.plot(timesteps_threshold, threshold_plot, color='green', linestyle='--', 
           linewidth=2, label='Threshold (CPband)', zorder=2)
    
    # Shade region above threshold
    ax.fill_between(timesteps_threshold, threshold_plot, np.max(logzpo_timeseries) * 1.1, 
                   color='red', alpha=0.1, label='Failure Region', zorder=1)
    
    # Mark anomaly if detected
    if anomaly_detected and first_idx >= 0:
        ax.axvline(x=timesteps[first_idx], color='red', linestyle=':', linewidth=3, 
                  label=f'Anomaly Detected at t={first_idx}', zorder=4)
        ax.plot(timesteps[first_idx], logzpo_timeseries[first_idx], 'r^', markersize=14, 
               markeredgewidth=2, markeredgecolor='darkred', zorder=5)
    
    ax.set_xlabel('Timestep', fontsize=12)
    ax.set_ylabel('logzpo Score', fontsize=12)
    ax.set_title(f'{trajectory_name} - Per-Timestep logzpo with CPband Detection', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(fontsize=11, loc='upper left')
    
    # Add statistics text box
    stats_text = f"""
    Trajectory Length: {T}
    Success Label: {label_text}
    Anomaly Detected: {'YES' if anomaly_detected else 'NO'}
    """
    if anomaly_detected and first_idx >= 0:
        stats_text += f"    Detection at t={first_idx} ({100*first_idx/T:.1f}%)\n    logzpo[{first_idx}]={logzpo_timeseries[first_idx]:.2f}\n"
    
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           family='monospace')
    
    plt.tight_layout()
    return fig


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Compute per-timestep logzpo and detect failures using CPband'
    )
    parser.add_argument(
        '--demo', 
        type=str, 
        default=DEFAULT_DEMO_PATH,
        help=f'Path to single-demo HDF5 file (default: {DEFAULT_DEMO_PATH})'
    )
    parser.add_argument(
        '--calibration',
        type=str,
        default=DEFAULT_CALIBRATION_PATH,
        help=f'Path to calibration HDF5 file (default: {DEFAULT_CALIBRATION_PATH})'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='/home/zhiyuanjia/FAIL-Detect/UQ_test/test_libero_logzpo_plot.png',
        help='Output plot file path'
    )
    
    args = parser.parse_args()
    demo_hdf5_path = args.demo
    calibration_hdf5_path = args.calibration
    output_path = args.output
    
    print(f"\nDevice: {device}")
    print(f"Task: {task_name}, Policy: {policy_type}\n")
    print(f"Demo file: {demo_hdf5_path}")
    print(f"Calibration file: {calibration_hdf5_path}\n")
    
    # Load baseline model
    print(f"{'='*60}")
    print("Loading baseline model...")
    print(f"{'='*60}")
    baseline_model = elb.get_baseline_model('logpZO', task_name, policy_type).to(device)
    baseline_model.eval()
    baseline_model.global_eps = None
    print("✓ Model loaded successfully\n")
    
    # Phase 3: Build per-timestep threshold using CPband
    threshold_per_timestep, calibration_stats = build_calibration_threshold(
        baseline_model, 
        calibration_hdf5_path,
        task_name=task_name,
        num_successful_demos=25,
        alpha=0.1
    )
    
    # Phase 1: Load single demo
    print(f"\n{'='*60}")
    print(f"Loading single demo from {demo_hdf5_path}")
    print(f"{'='*60}")
    observations, actions, success_label, T = load_single_demo(demo_hdf5_path)
    print(f"✓ Loaded demo: {T} timesteps, {observations.shape[1]}-dim observations")
    print(f"  Success label: {success_label} ({'Success' if success_label == 1 else 'Failure'})\n")
    
    # Phase 2: Compute per-timestep logzpo
    print(f"{'='*60}")
    print("Computing per-timestep logzpo...")
    print(f"{'='*60}")
    observations = observations.to(device)
    logzpo_timeseries = compute_logzpo_timeseries(baseline_model, observations, task_name)
    print(f"✓ Computed logzpo timeseries: {logzpo_timeseries.shape}")
    print(f"  Min logzpo: {logzpo_timeseries.min():.4f}")
    print(f"  Max logzpo: {logzpo_timeseries.max():.4f}")
    print(f"  Mean logzpo: {logzpo_timeseries.mean():.4f}\n")
    
    # Phase 3: Detect anomalies using per-timestep CPband
    print(f"{'='*60}")
    print("Detecting anomalies using CPband...")
    print(f"{'='*60}")
    anomaly_detected, first_idx, anomaly_percentage = detect_anomalies(logzpo_timeseries, threshold_per_timestep)
    
    print(f"Anomaly Detected: {anomaly_detected}")
    if anomaly_detected:
        print(f"  First anomaly at timestep: {first_idx}")
        print(f"  logzpo[{first_idx}] = {logzpo_timeseries[first_idx]:.4f}")
        print(f"  Threshold[{first_idx}] = {threshold_per_timestep[first_idx]:.4f}")
        print(f"  Percentage through episode: {anomaly_percentage:.1f}%")
    print()
    
    # Phase 4: Visualization
    print(f"{'='*60}")
    print("Generating visualization...")
    print(f"{'='*60}")
    fig = plot_trajectory_with_detection(
        logzpo_timeseries, threshold_per_timestep,
        success_label, anomaly_detected, first_idx,
        trajectory_name=f"LIBERO Single Demo (T={T})"
    )
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to {output_path}")
    plt.show()
    
    print(f"\n{'='*60}")
    print("✓ Analysis complete!")
    print(f"{'='*60}\n")

