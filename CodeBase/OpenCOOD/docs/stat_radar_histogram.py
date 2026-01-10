#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistics script for radar point cloud histogram information.

This script scans data/v2x-radar/v2x-radar-c directory and collects statistics
for the raw intensity and dopple fields from PCD files (without normalization)
for CAV -1 (Infra) and CAV 142 (Vehicle).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from packages.pypcd import pypcd


def collect_radar_statistics(data_root, cav_ids=['-1', '142']):
    """
    Collect radar point cloud statistics for specified CAVs.
    Reads raw intensity and dopple fields directly from PCD files (no normalization).
    
    Parameters
    ----------
    data_root : str
        Root directory path (e.g., 'data/v2x-radar/v2x-radar-c')
    cav_ids : list
        List of CAV IDs to process
    
    Returns
    -------
    stats_dict : dict
        Dictionary with statistics for each CAV ID
        Format: {cav_id: {'intensity': [...], 'dopple': [...]}}
    """
    data_root = Path(data_root)
    stats_dict = {cav_id: {'intensity': [], 'dopple': []} for cav_id in cav_ids}
    
    # Find all radar.pcd files
    radar_files = []
    for split_dir in ['train', 'validate', 'test']:
        split_path = data_root / split_dir
        if not split_path.exists():
            continue
        
        # Find all scenario directories
        for scenario_dir in split_path.iterdir():
            if not scenario_dir.is_dir():
                continue
            
            # Check each CAV directory
            for cav_id in cav_ids:
                cav_dir = scenario_dir / cav_id
                if not cav_dir.exists():
                    continue
                
                # Find all radar.pcd files
                for radar_file in cav_dir.glob('*_radar.pcd'):
                    radar_files.append((str(radar_file), cav_id))
    
    print(f"Found {len(radar_files)} radar files")
    
    # Process each radar file
    for idx, (radar_path, cav_id) in enumerate(radar_files):
        if (idx + 1) % 100 == 0:
            print(f"Processing {idx + 1}/{len(radar_files)} files...")
        
        try:
            # Read PCD file directly using pypcd (no normalization)
            pcd = pypcd.PointCloud.from_path(radar_path)
            # Extract raw intensity and dopple fields
            # Use exactly the same method as pcd_utils.read_radar
            # Use transpose exactly like pcd_utils does for x, y, z
            intensity_data = np.transpose(pcd.pc_data["intensity"])
            # Convert to list directly (no isfinite check to avoid structured array issues)
            stats_dict[cav_id]['intensity'].extend(intensity_data.flatten().tolist())
            # Use transpose exactly like pcd_utils does for x, y, z
            dopple_data = np.transpose(pcd.pc_data["dopple"])
            # Convert to list directly (no isfinite check to avoid structured array issues)
            stats_dict[cav_id]['dopple'].extend(dopple_data.flatten().tolist())
        
        except Exception as e:
            print(f"Error processing {radar_path}: {e}")
            continue
    
    return stats_dict


def compute_histogram_stats(data_array, dim_name):
    """
    Compute histogram statistics for a data array.
    Filters out NaN and infinite values before computing statistics.
    
    Parameters
    ----------
    data_array : np.ndarray or list
        Data array (1D)
    dim_name : str
        Dimension name for labeling
    
    Returns
    -------
    stats : dict
        Dictionary containing statistics
    """
    if len(data_array) == 0:
        return None
    
    data_array = np.array(data_array, dtype=np.float64)
    
    # Filter out NaN and infinite values
    valid_mask = np.isfinite(data_array)
    if not valid_mask.any():
        return None
    
    data_array = data_array[valid_mask]
    
    stats = {
        'count': len(data_array),
        'mean': float(np.mean(data_array)),
        'std': float(np.std(data_array)),
        'min': float(np.min(data_array)),
        'max': float(np.max(data_array)),
        'median': float(np.median(data_array)),
        'percentiles': {
            'p1': float(np.percentile(data_array, 1)),
            'p5': float(np.percentile(data_array, 5)),
            'p25': float(np.percentile(data_array, 25)),
            'p50': float(np.percentile(data_array, 50)),
            'p75': float(np.percentile(data_array, 75)),
            'p95': float(np.percentile(data_array, 95)),
            'p99': float(np.percentile(data_array, 99)),
        }
    }
    
    return stats


def print_statistics(stats_dict):
    """
    Print statistics in a formatted way.
    
    Parameters
    ----------
    stats_dict : dict
        Statistics dictionary from collect_radar_statistics
    """
    print("\n" + "="*80)
    print("RADAR POINT CLOUD STATISTICS (RAW FIELDS, NO NORMALIZATION)")
    print("="*80)
    
    for cav_id in sorted(stats_dict.keys()):
        print(f"\n{'='*80}")
        print(f"CAV {cav_id}")
        print(f"{'='*80}")
        
        for field_name in ['intensity', 'dopple']:
            data = stats_dict[cav_id][field_name]
            
            if len(data) == 0:
                print(f"\n{field_name} field: No data")
                continue
            
            stats = compute_histogram_stats(data, field_name)
            if stats is None:
                continue
            
            print(f"\n{field_name.upper()} Field Statistics (Raw Values):")
            print(f"  Count:        {stats['count']:,}")
            print(f"  Mean:         {stats['mean']:.6f}")
            print(f"  Std:          {stats['std']:.6f}")
            print(f"  Min:          {stats['min']:.6f}")
            print(f"  Max:          {stats['max']:.6f}")
            print(f"  Median:       {stats['median']:.6f}")
            print(f"  Percentiles:")
            print(f"    P1:         {stats['percentiles']['p1']:.6f}")
            print(f"    P5:         {stats['percentiles']['p5']:.6f}")
            print(f"    P25:        {stats['percentiles']['p25']:.6f}")
            print(f"    P50:        {stats['percentiles']['p50']:.6f}")
            print(f"    P75:        {stats['percentiles']['p75']:.6f}")
            print(f"    P95:        {stats['percentiles']['p95']:.6f}")
            print(f"    P99:        {stats['percentiles']['p99']:.6f}")


def plot_histograms(stats_dict, output_dir='docs/visualization'):
    """
    Plot histograms for each CAV and field (intensity and dopple).
    
    Parameters
    ----------
    stats_dict : dict
        Statistics dictionary from collect_radar_statistics
    output_dir : str
        Output directory for saving plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for cav_id in sorted(stats_dict.keys()):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Radar Point Cloud Histograms (Raw Fields) - CAV {cav_id}', 
                     fontsize=14, fontweight='bold')
        
        for field_idx, (field_name, ax) in enumerate(zip(['intensity', 'dopple'], axes)):
            data = stats_dict[cav_id][field_name]
            
            if len(data) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{field_name.upper()} (No data)')
                continue
            
            data_array = np.array(data)
            
            # Filter out NaN and infinite values first
            valid_mask = np.isfinite(data_array)
            if not valid_mask.any():
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{field_name.upper()} (No valid data)')
                continue
            
            data_array = data_array[valid_mask]
            
            # Compute statistics on full data (for display)
            stats = compute_histogram_stats(data_array, field_name)
            
            # Filter data to 2%-98% percentile range for plotting (avoid extreme values)
            p10 = np.percentile(data_array, 2)
            p90 = np.percentile(data_array, 98)
            plot_mask = (data_array >= p10) & (data_array <= p90)
            data_plot = data_array[plot_mask]
            
            if len(data_plot) == 0:
                ax.text(0.5, 0.5, 'No data in 2%-98% range', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{field_name.upper()} (No data in range)')
                continue
            
            # Plot histogram (only 2%-98% range)
            n_bins = 100
            counts, bins, patches = ax.hist(data_plot, bins=n_bins, alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Color bars by value (gradient)
            cm = plt.cm.get_cmap('viridis')
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            col = bin_centers - bin_centers.min()
            col /= col.max() if col.max() > 0 else 1
            for c, p in zip(col, patches):
                plt.setp(p, 'facecolor', cm(c))
            
            # Add statistics text (based on full data, but note the plot range)
            if stats:
                stats_text = (
                    f"Count: {stats['count']:,}\n"
                    f"Mean: {stats['mean']:.4f}\n"
                    f"Std: {stats['std']:.4f}\n"
                    f"Min: {stats['min']:.4f}\n"
                    f"Max: {stats['max']:.4f}\n"
                    f"Median: {stats['median']:.4f}\n"
                    f"Plot range: [{p10:.4f}, {p90:.4f}]"
                )
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_xlabel(f'{field_name.upper()} Value (Raw, 2%-98% range)', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(f'{field_name.upper()} Field Histogram (2%-98% percentile)', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'radar_histogram_raw_CAV_{cav_id}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved histogram plot to {output_path}")
        plt.close()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Statistics for radar point cloud histogram')
    parser.add_argument('--data_root', type=str, 
                       default='data/v2x-radar/v2x-radar-c',
                       help='Root directory of v2x-radar dataset')
    parser.add_argument('--cav_ids', nargs='+', default=['-1', '142'],
                       help='CAV IDs to process (default: -1 142)')
    parser.add_argument('--output_dir', type=str, default='docs/visualization',
                       help='Output directory for plots (default: docs/visualization)')
    parser.add_argument('--no_plot', action='store_true',
                       help='Skip plotting histograms')
    
    args = parser.parse_args()
    
    # Collect statistics
    print("Collecting radar point cloud statistics...")
    print(f"Data root: {args.data_root}")
    print(f"CAV IDs: {args.cav_ids}")
    
    stats_dict = collect_radar_statistics(args.data_root, args.cav_ids)
    
    # Print statistics
    print_statistics(stats_dict)
    
    # Plot histograms
    if not args.no_plot:
        print("\nGenerating histogram plots...")
        plot_histograms(stats_dict, args.output_dir)
    
    print("\nDone!")


if __name__ == "__main__":
    main()

