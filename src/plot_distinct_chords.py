#!/usr/bin/env python3
"""
Plot the distinct chords ratio over time for multiple walks.

Compares classical random walks vs quantum walks to see differences
in exploration patterns.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import glob
from midi_generator import calculate_distinct_chords_ratio


def load_walk_from_csv(csv_path):
    """
    Load a walk from CSV file and extract the chord sequence.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of chord names in order
    """
    df = pd.read_csv(csv_path)
    return df['current_chord'].tolist()


def plot_walks(classical_dir, qutrit_dir, window_size=10, num_walks=20, output_path='distinct_chords_comparison.pdf'):
    """
    Plot mean and standard deviation of distinct chords ratio for classical and qutrit walks.

    Args:
        classical_dir: Directory containing classical walk CSV files
        qutrit_dir: Directory containing qutrit walk CSV files
        window_size: Window size for calculating distinct chords ratio
        num_walks: Number of walks to analyze from each type
        output_path: Where to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Collect classical walks data
    # Try both patterns: files with .csv extension and without
    classical_pattern_csv = str(Path(classical_dir) / "*.csv")
    classical_pattern_noext = str(Path(classical_dir) / "*")

    classical_files = sorted(glob.glob(classical_pattern_csv))
    if len(classical_files) == 0:
        # Try without extension, but filter out MIDI files
        classical_files = sorted([f for f in glob.glob(classical_pattern_noext)
                                 if not f.endswith('.mid') and Path(f).is_file()])

    classical_files = classical_files[:num_walks]
    print(f"Found {len(classical_files)} classical walk files")
    if len(classical_files) == 0:
        print(f"  Searched patterns: {classical_pattern_csv} and {classical_pattern_noext}")
        print(f"  Directory exists: {Path(classical_dir).exists()}")
        if Path(classical_dir).exists():
            print(f"  Files in directory: {list(Path(classical_dir).glob('*'))[:5]}")

    classical_ratios = []
    for csv_file in classical_files:
        chord_sequence = load_walk_from_csv(csv_file)
        ratios = calculate_distinct_chords_ratio(chord_sequence, window_size)
        classical_ratios.append(ratios)

    # Calculate mean and std for classical walks
    if classical_ratios:
        classical_ratios = np.array(classical_ratios)
        classical_mean = np.mean(classical_ratios, axis=0)
        classical_std = np.std(classical_ratios, axis=0)
        steps = np.arange(len(classical_mean))

        ax1.plot(steps, classical_mean, linewidth=2, color='blue', label='Mean')
        ax1.fill_between(steps, classical_mean - classical_std, classical_mean + classical_std,
                         alpha=0.3, color='blue', label='±1 std dev')

    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel(f'Distinct Chords Ratio (window={window_size})', fontsize=11)
    ax1.set_title(f'Classical Random Walks (n={len(classical_files)})', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=10)

    # Collect qutrit walks data
    qutrit_pattern = str(Path(qutrit_dir) / "*.csv")
    qutrit_files = sorted(glob.glob(qutrit_pattern))[:num_walks]
    print(f"Found {len(qutrit_files)} qutrit walk files")
    if len(qutrit_files) == 0:
        print(f"  Warning: No files found with pattern: {qutrit_pattern}")

    qutrit_ratios = []
    for csv_file in qutrit_files:
        chord_sequence = load_walk_from_csv(csv_file)
        ratios = calculate_distinct_chords_ratio(chord_sequence, window_size)
        qutrit_ratios.append(ratios)

    # Calculate mean and std for qutrit walks
    if qutrit_ratios:
        qutrit_ratios = np.array(qutrit_ratios)
        qutrit_mean = np.mean(qutrit_ratios, axis=0)
        qutrit_std = np.std(qutrit_ratios, axis=0)
        steps = np.arange(len(qutrit_mean))

        ax2.plot(steps, qutrit_mean, linewidth=2, color='red', label='Mean')
        ax2.fill_between(steps, qutrit_mean - qutrit_std, qutrit_mean + qutrit_std,
                         alpha=0.3, color='red', label='±1 std dev')

    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel(f'Distinct Chords Ratio (window={window_size})', fontsize=11)
    ax2.set_title(f'Quantum Walks (LPR order, n={len(qutrit_files)})', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")

    # Also save as PNG
    png_path = Path(output_path).with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {png_path}")

    plt.close()


def plot_combined(classical_dir, qutrit_dir, window_size=10, num_walks=20, output_path='distinct_chords_combined.pdf'):
    """
    Plot both walk types on a single plot with mean and std dev.

    Args:
        classical_dir: Directory containing classical walk CSV files
        qutrit_dir: Directory containing qutrit walk CSV files
        window_size: Window size for calculating distinct chords ratio
        num_walks: Number of walks to analyze from each type
        output_path: Where to save the plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect classical walks data
    # Try both patterns: files with .csv extension and without
    classical_pattern_csv = str(Path(classical_dir) / "*.csv")
    classical_pattern_noext = str(Path(classical_dir) / "*")

    classical_files = sorted(glob.glob(classical_pattern_csv))
    if len(classical_files) == 0:
        # Try without extension, but filter out MIDI files
        classical_files = sorted([f for f in glob.glob(classical_pattern_noext)
                                 if not f.endswith('.mid') and Path(f).is_file()])

    classical_files = classical_files[:num_walks]
    print(f"Analyzing {len(classical_files)} classical walks")
    if len(classical_files) == 0:
        print(f"  Warning: No files found in {classical_dir}")

    classical_ratios = []
    for csv_file in classical_files:
        chord_sequence = load_walk_from_csv(csv_file)
        ratios = calculate_distinct_chords_ratio(chord_sequence, window_size)
        classical_ratios.append(ratios)

    # Calculate mean and std for classical walks
    if classical_ratios:
        classical_ratios = np.array(classical_ratios)
        classical_mean = np.mean(classical_ratios, axis=0)
        classical_std = np.std(classical_ratios, axis=0)
        steps = np.arange(len(classical_mean))

        ax.plot(steps, classical_mean, linewidth=2.5, color='blue', label='Classical (mean)')
        ax.fill_between(steps, classical_mean - classical_std, classical_mean + classical_std,
                        alpha=0.25, color='blue')

    # Collect qutrit walks data
    qutrit_pattern = str(Path(qutrit_dir) / "*.csv")
    qutrit_files = sorted(glob.glob(qutrit_pattern))[:num_walks]
    print(f"Analyzing {len(qutrit_files)} qutrit walks")
    if len(qutrit_files) == 0:
        print(f"  Warning: No files found with pattern: {qutrit_pattern}")

    qutrit_ratios = []
    for csv_file in qutrit_files:
        chord_sequence = load_walk_from_csv(csv_file)
        ratios = calculate_distinct_chords_ratio(chord_sequence, window_size)
        qutrit_ratios.append(ratios)

    # Calculate mean and std for qutrit walks
    if qutrit_ratios:
        qutrit_ratios = np.array(qutrit_ratios)
        qutrit_mean = np.mean(qutrit_ratios, axis=0)
        qutrit_std = np.std(qutrit_ratios, axis=0)
        steps = np.arange(len(qutrit_mean))

        ax.plot(steps, qutrit_mean, linewidth=2.5, color='red', label='Quantum (mean)')
        ax.fill_between(steps, qutrit_mean - qutrit_std, qutrit_mean + qutrit_std,
                        alpha=0.25, color='red')

    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel(f'Distinct Chords Ratio (window={window_size})', fontsize=12)
    ax.set_title(f'Classical vs Quantum Walks: Exploration Diversity (n={num_walks} each)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, loc='best')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot to {output_path}")

    png_path = Path(output_path).with_suffix('.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved combined plot to {png_path}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Plot distinct chords ratio for classical and quantum walks"
    )
    parser.add_argument('--classical-dir', type=str,
                       default='../results/ClassicalWalkResults',
                       help='Directory with classical walk CSV files')
    parser.add_argument('--qutrit-dir', type=str,
                       default='../results/QutritWalkSweepResults',
                       help='Directory with qutrit walk CSV files')
    parser.add_argument('--window-size', type=int, default=10,
                       help='Window size for distinct chords ratio (default: 10)')
    parser.add_argument('--num-walks', type=int, default=20,
                       help='Number of walks to plot from each type (default: 20)')
    parser.add_argument('--output', type=str, default='distinct_chords_comparison.pdf',
                       help='Output file path (default: distinct_chords_comparison.pdf)')
    parser.add_argument('--separate', action='store_true',
                       help='Plot as separate subplots (default: combined on same axes)')

    args = parser.parse_args()

    print(f"Window size: {args.window_size}")
    print(f"Number of walks per type: {args.num_walks}")
    print()

    if args.separate:
        plot_walks(
            args.classical_dir,
            args.qutrit_dir,
            window_size=args.window_size,
            num_walks=args.num_walks,
            output_path=args.output
        )
    else:
        plot_combined(
            args.classical_dir,
            args.qutrit_dir,
            window_size=args.window_size,
            num_walks=args.num_walks,
            output_path=args.output
        )


if __name__ == '__main__':
    main()
