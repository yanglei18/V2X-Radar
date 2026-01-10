#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Statistics script for GT box distribution

This script analyzes the distribution of GT boxes for a specific CAV across all scenarios.
"""
import yaml
import numpy as np
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import argparse

def load_yaml(yaml_path):
    """Load YAML file"""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def count_gt_boxes_in_yaml(yaml_path):
    """Count GT boxes in a YAML file"""
    try:
        params = load_yaml(yaml_path)
        if 'vehicles' in params and params['vehicles'] is not None:
            return len(params['vehicles'])
        return 0
    except Exception as e:
        print(f"Error reading {yaml_path}: {e}")
        return None

def get_all_timestamps(scenario_dir, cav_id):
    """Get all timestamp files for a specific CAV in a scenario"""
    cav_dir = Path(scenario_dir) / str(cav_id)
    if not cav_dir.exists():
        return []
    
    # Find all YAML files (timestamp.yaml)
    yaml_files = sorted([f for f in cav_dir.iterdir() 
                        if f.is_file() and f.suffix == '.yaml' and f.stem.isdigit()])
    return yaml_files

def analyze_scenario(scenario_path, cav_id):
    """Analyze a single scenario and return GT box counts"""
    yaml_files = get_all_timestamps(scenario_path, cav_id)
    counts = []
    
    for yaml_file in yaml_files:
        count = count_gt_boxes_in_yaml(yaml_file)
        if count is not None:
            counts.append(count)
    
    return counts

def main():
    parser = argparse.ArgumentParser(description="Statistics for GT box distribution")
    parser.add_argument("--data_dir", type=str, default="data/v2x-radar/v2x-radar-c/train", help="Path to data directory containing scenarios")
    parser.add_argument("--cav_id", type=str, default="-1", help="CAV ID to analyze (default: '142')")
    parser.add_argument("--output_dir", type=str, default="docs", help="Output directory for statistics (default: 'docs')")
    
    args = parser.parse_args()
    
    # Get project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = Path(project_root) / args.data_dir
    output_dir = Path(project_root) / args.output_dir
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    print(f"Analyzing GT box distribution for CAV {args.cav_id}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print("="*60)
    
    # Get all scenario directories
    scenarios = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    print(f"Found {len(scenarios)} scenarios\n")
    
    # Collect statistics
    all_counts = []
    scenario_stats = {}
    
    for scenario_path in scenarios:
        scenario_name = scenario_path.name
        counts = analyze_scenario(scenario_path, args.cav_id)
        
        if len(counts) > 0:
            scenario_stats[scenario_name] = {
                'counts': counts,
                'mean': np.mean(counts),
                'min': np.min(counts),
                'max': np.max(counts),
                'std': np.std(counts),
                'total_frames': len(counts)
            }
            all_counts.extend(counts)
            print(f"{scenario_name}: {len(counts)} frames, "
                  f"GT count - mean: {scenario_stats[scenario_name]['mean']:.2f}, "
                  f"min: {scenario_stats[scenario_name]['min']}, "
                  f"max: {scenario_stats[scenario_name]['max']}")
    
    if len(all_counts) == 0:
        print("No data found!")
        return
    
    # Overall statistics
    print("\n" + "="*60)
    print("Overall Statistics:")
    print(f"Total frames analyzed: {len(all_counts)}")
    print(f"Total scenarios: {len(scenario_stats)}")
    print(f"Mean GT boxes per frame: {np.mean(all_counts):.2f}")
    print(f"Median GT boxes per frame: {np.median(all_counts):.2f}")
    print(f"Min GT boxes per frame: {np.min(all_counts)}")
    print(f"Max GT boxes per frame: {np.max(all_counts)}")
    print(f"Std GT boxes per frame: {np.std(all_counts):.2f}")
    
    # Distribution statistics
    unique_counts, counts_freq = np.unique(all_counts, return_counts=True)
    print(f"\nUnique GT box counts: {len(unique_counts)}")
    print(f"Most common count: {unique_counts[np.argmax(counts_freq)]} (appears {counts_freq[np.argmax(counts_freq)]} times)")
    
    # Save detailed statistics
    output_file = output_dir / f"gt_statistics_cav_{args.cav_id}.txt"
    with open(output_file, 'w') as f:
        f.write(f"GT Box Distribution Statistics for CAV {args.cav_id}\n")
        f.write("="*60 + "\n\n")
        f.write(f"Data directory: {data_dir}\n")
        f.write(f"Total scenarios: {len(scenario_stats)}\n")
        f.write(f"Total frames: {len(all_counts)}\n\n")
        
        f.write("Overall Statistics:\n")
        f.write(f"  Mean: {np.mean(all_counts):.2f}\n")
        f.write(f"  Median: {np.median(all_counts):.2f}\n")
        f.write(f"  Min: {np.min(all_counts)}\n")
        f.write(f"  Max: {np.max(all_counts)}\n")
        f.write(f"  Std: {np.std(all_counts):.2f}\n\n")
        
        f.write("Per-Scenario Statistics:\n")
        f.write("-"*60 + "\n")
        for scenario_name, stats in sorted(scenario_stats.items()):
            f.write(f"{scenario_name}:\n")
            f.write(f"  Frames: {stats['total_frames']}\n")
            f.write(f"  Mean: {stats['mean']:.2f}\n")
            f.write(f"  Min: {stats['min']}\n")
            f.write(f"  Max: {stats['max']}\n")
            f.write(f"  Std: {stats['std']:.2f}\n\n")
        
        f.write("Distribution (Count -> Frequency):\n")
        f.write("-"*60 + "\n")
        for count, freq in zip(unique_counts, counts_freq):
            f.write(f"  {count}: {freq} ({freq/len(all_counts)*100:.2f}%)\n")
    
    print(f"\nDetailed statistics saved to: {output_file}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Histogram of all GT counts
    axes[0, 0].hist(all_counts, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Number of GT Boxes')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Distribution of GT Boxes per Frame (CAV {args.cav_id})')
    axes[0, 0].axvline(np.mean(all_counts), color='r', linestyle='--', label=f'Mean: {np.mean(all_counts):.2f}')
    axes[0, 0].axvline(np.median(all_counts), color='g', linestyle='--', label=f'Median: {np.median(all_counts):.2f}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box plot
    axes[0, 1].boxplot(all_counts, vert=True)
    axes[0, 1].set_ylabel('Number of GT Boxes')
    axes[0, 1].set_title(f'Box Plot of GT Boxes (CAV {args.cav_id})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Per-scenario mean comparison
    scenario_names = sorted(scenario_stats.keys())
    scenario_means = [scenario_stats[name]['mean'] for name in scenario_names]
    axes[1, 0].bar(range(len(scenario_names)), scenario_means, alpha=0.7)
    axes[1, 0].set_xlabel('Scenario Index')
    axes[1, 0].set_ylabel('Mean GT Boxes per Frame')
    axes[1, 0].set_title(f'Mean GT Boxes per Scenario (CAV {args.cav_id})')
    axes[1, 0].set_xticks(range(0, len(scenario_names), max(1, len(scenario_names)//20)))
    axes[1, 0].set_xticklabels([scenario_names[i] if i < len(scenario_names) else '' 
                                for i in range(0, len(scenario_names), max(1, len(scenario_names)//20))],
                               rotation=45, ha='right')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 4. Cumulative distribution
    sorted_counts = np.sort(all_counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    axes[1, 1].plot(sorted_counts, cumulative, linewidth=2)
    axes[1, 1].set_xlabel('Number of GT Boxes')
    axes[1, 1].set_ylabel('Cumulative Probability')
    axes[1, 1].set_title(f'Cumulative Distribution (CAV {args.cav_id})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    fig_path = output_dir / f"gt_distribution_cav_{args.cav_id}.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {fig_path}")
    plt.close()
    
    print("\nAnalysis completed!")

if __name__ == "__main__":
    main()

