"""
Statistical experiments for qutrit quantum walk on triadic tonnetz

This module runs the qutrit walk without MIDI playback and saves the results
to CSV for statistical analysis.
"""

import argparse
import csv
import random
import numpy as np
from QutritWalk import QutritWalkSimulator, Chord


# All 24 chords in order: majors then minors
# Use Chord.NOTE_NAMES directly to build clean names
ALL_CHORD_NAMES = []
for root_name in Chord.NOTE_NAMES:
    ALL_CHORD_NAMES.append(root_name)  # Major chords
for root_name in Chord.NOTE_NAMES:
    ALL_CHORD_NAMES.append(root_name + 'm')  # Minor chords


def run_experiment(num_steps: int, initial_chord: Chord, initial_spin: int = 0,
                   seed: int = None) -> list:
    """
    Run a single qutrit walk experiment

    Args:
        num_steps: Number of steps to run
        initial_chord: Starting chord
        initial_spin: Initial spin state (0=|↑⟩, 1=|→⟩, 2=|↓⟩)
        seed: Random seed for reproducibility

    Returns:
        List of dictionaries, one per step, containing:
            - step: Step number
            - current_chord: Current chord name
            - neighbor_L: Leittonwechsel neighbor name
            - neighbor_P: Parallel neighbor name
            - neighbor_R: Relative neighbor name
            - All 24 chord probabilities (C, Cm, C#, C#m, ..., B, Bm)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    simulator = QutritWalkSimulator(initial_chord, initial_spin)
    results = []

    for step in range(num_steps):
        result = simulator.step_and_sample()

        # Build row for CSV
        row = {
            'step': step,
            'current_chord': str(result['current'])
        }

        # Add neighbor names (empty for first step)
        if result['is_first']:
            row['neighbor_L'] = ''
            row['neighbor_P'] = ''
            row['neighbor_R'] = ''
        else:
            row['neighbor_L'] = str(result['neighbors'][0])
            row['neighbor_P'] = str(result['neighbors'][1])
            row['neighbor_R'] = str(result['neighbors'][2])

        # Add all 24 chord probabilities
        all_probs = result['all_probs']
        for chord_name in ALL_CHORD_NAMES:
            # Parse chord name back to (root, is_major)
            if chord_name.endswith('m'):
                # Minor chord - strip the 'm' to get root
                root_name = chord_name[:-1]
                is_major = False
            else:
                # Major chord
                root_name = chord_name
                is_major = True

            # Find the root index
            root_idx = Chord.NOTE_NAMES.index(root_name)
            key = (root_idx, is_major)
            row[chord_name] = all_probs.get(key, 0.0)

        results.append(row)

    return results


def save_to_csv(results: list, filename: str):
    """Save experiment results to CSV file"""
    if not results:
        return

    fieldnames = ['step', 'current_chord', 'neighbor_L', 'neighbor_P', 'neighbor_R'] + ALL_CHORD_NAMES

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} steps to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Run qutrit quantum walk experiments and save to CSV"
    )
    parser.add_argument('--steps', type=int, default=100,
                        help='Number of steps to run (default: 100)')
    parser.add_argument('--root', type=str, default='C',
                        help='Starting chord root note (default: C)')
    parser.add_argument('--minor', action='store_true',
                        help='Start with minor chord (default: major)')
    parser.add_argument('--seed', type=int,
                        help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default='qutrit_walk.csv',
                        help='Output CSV filename (default: qutrit_walk.csv)')
    parser.add_argument('--num-runs', type=int, default=1,
                        help='Number of independent runs to perform (default: 1)')

    args = parser.parse_args()

    # Parse starting chord
    try:
        root_idx = Chord.NOTE_NAMES.index(args.root)
    except ValueError:
        print(f"Error: Invalid root note '{args.root}'. Valid notes: {', '.join(Chord.NOTE_NAMES)}")
        return

    initial_chord = Chord(root_idx, not args.minor, 0)

    print(f"Running qutrit walk experiments")
    print(f"Initial chord: {initial_chord}")
    print(f"Steps per run: {args.steps}")
    print(f"Number of runs: {args.num_runs}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print()

    # Run experiments
    if args.num_runs == 1:
        # Single run - simple filename
        results = run_experiment(args.steps, initial_chord, seed=args.seed)
        save_to_csv(results, args.output)
    else:
        # Multiple runs - add run number to filename
        base_filename = args.output.rsplit('.', 1)[0]
        ext = args.output.rsplit('.', 1)[1] if '.' in args.output else 'csv'

        for run in range(args.num_runs):
            run_seed = args.seed + run if args.seed is not None else None
            results = run_experiment(args.steps, initial_chord, seed=run_seed)

            filename = f"{base_filename}_run{run+1}.{ext}"
            save_to_csv(results, filename)

    print("\nDone!")


if __name__ == "__main__":
    main()
