"""
Statistical experiments for qutrit quantum walk on triadic tonnetz

This module runs the qutrit walk without MIDI playback and saves the results
to CSV for statistical analysis.
"""

import argparse
import csv
import random
import numpy as np
from pathlib import Path
from QutritWalk import QutritWalkSimulator, Chord, parse_initial_state
from midi_generator import save_walk_to_midi, calculate_distinct_chords_ratio
from typing import Optional, List, Tuple


# All 24 chords in order: majors then minors
# Use Chord.NOTE_NAMES directly to build clean names
ALL_CHORD_NAMES = []
for root_name in Chord.NOTE_NAMES:
    ALL_CHORD_NAMES.append(root_name)  # Major chords
for root_name in Chord.NOTE_NAMES:
    ALL_CHORD_NAMES.append(root_name + 'm')  # Minor chords

# All 72 state basis names: chord_spin for real and imaginary parts
SPIN_NAMES = ['up', 'right', 'down']
ALL_STATE_NAMES = []
for chord_name in ALL_CHORD_NAMES:
    for spin_name in SPIN_NAMES:
        ALL_STATE_NAMES.append(f"{chord_name}_{spin_name}_re")
        ALL_STATE_NAMES.append(f"{chord_name}_{spin_name}_im")


def run_experiment(num_steps: int, initial_chord: Chord, initial_spin: int = 0,
                   initial_superposition: Optional[List[Tuple[Chord, int, complex]]] = None,
                   transform_order: str = "LPR",
                   seed: Optional[int] = None) -> list:
    """
    Run a single qutrit walk experiment

    Args:
        num_steps: Number of steps to run
        initial_chord: Starting chord (used if initial_superposition is None)
        initial_spin: Initial spin state (0=|↑⟩, 1=|→⟩, 2=|↓⟩)
        initial_superposition: Optional custom initial state
        transform_order: Transformation order (default: "LPR")
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

    simulator = QutritWalkSimulator(initial_chord, initial_spin, initial_superposition, transform_order)
    results = []

    for step in range(num_steps):
        result = simulator.step_and_sample()

        # Skip the initial state (step 0 before any evolution)
        if result['is_first']:
            continue

        # Build row for CSV
        row = {
            'step': step - 1,  # Adjust step number since we skipped step 0
            'current_chord': str(result['current'])
        }

        # Add neighbor names
        row['neighbor_L'] = str(result['neighbors'][0])
        row['neighbor_P'] = str(result['neighbors'][1])
        row['neighbor_R'] = str(result['neighbors'][2])

        # Add all 24 chord probabilities (marginal over spins)
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

        # Add all 72 complex amplitudes (full quantum state)
        full_state = result['full_state']
        for chord_name in ALL_CHORD_NAMES:
            # Parse chord name back to (root, is_major)
            if chord_name.endswith('m'):
                root_name = chord_name[:-1]
                is_major = False
            else:
                root_name = chord_name
                is_major = True
            root_idx = Chord.NOTE_NAMES.index(root_name)

            # Add amplitudes for all 3 spin states
            for spin_idx, spin_name in enumerate(SPIN_NAMES):
                key = (root_idx, is_major, spin_idx)
                amplitude = full_state.get(key, 0.0 + 0.0j)
                row[f"{chord_name}_{spin_name}_re"] = amplitude.real
                row[f"{chord_name}_{spin_name}_im"] = amplitude.imag

        results.append(row)

    return results


def save_to_csv(results: list, filename: str, save_midi: bool = True, minimal: bool = False):
    """
    Save experiment results to CSV file and optionally MIDI file

    Args:
        results: List of result dictionaries from the walk
        filename: CSV filename to save to
        save_midi: If True, also save a MIDI file with the chord sequence
        minimal: If True, only save step and current_chord (saves space for large batches)
    """
    if not results:
        return

    if minimal:
        # Minimal format: only essential columns
        fieldnames = ['step', 'current_chord']
    else:
        # Full format: all columns
        fieldnames = ['step', 'current_chord', 'neighbor_L', 'neighbor_P', 'neighbor_R'] + ALL_CHORD_NAMES + ALL_STATE_NAMES

    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved {len(results)} steps to {filename}")

    # Save MIDI file with the chord sequence
    if save_midi:
        # Extract chord names from results
        chord_sequence = [row['current_chord'] for row in results]

        # Generate MIDI filename from CSV filename
        midi_filename = Path(filename).with_suffix('.mid')
        save_walk_to_midi(chord_sequence, midi_filename)


def run_parameter_sweep(num_steps: int, num_runs: int, output_dir: str, seed: Optional[int] = None, save_midi: bool = True, minimal: bool = False):
    """
    Run systematic parameter sweep experiments

    Runs two experiment sets:
    1. Fix initial state (C:up), vary transformation order (all 6 permutations)
    2. Fix transformation order (LPR), vary initial state (4 configurations)

    Args:
        num_steps: Number of steps per walk
        num_runs: Number of independent runs per configuration
        output_dir: Directory to save results
        seed: Base random seed (each run gets seed+offset)
    """
    import os

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Initial chord is always C major
    initial_chord = Chord(0, True, 0)

    print("="*70)
    print("PARAMETER SWEEP EXPERIMENTS")
    print("="*70)
    print(f"Steps per walk: {num_steps}")
    print(f"Runs per configuration: {num_runs}")
    print(f"Output directory: {output_dir}")
    if seed is not None:
        print(f"Base seed: {seed}")
    print()

    # Experiment 1: Fix initial state, vary transformation order
    print("Experiment 1: Varying transformation order (fixed initial state C:up)")
    print("-"*70)

    transform_orders = ['LPR', 'PRL', 'RLP']

    for order in transform_orders:
        print(f"  Transform order: {order}")

        for run in range(num_runs):
            run_seed = seed + run if seed is not None else None

            results = run_experiment(
                num_steps,
                initial_chord,
                initial_spin=0,  # up
                initial_superposition=None,
                transform_order=order,
                seed=run_seed
            )

            filename = os.path.join(output_dir, f"exp1_order_{order}_run{run+1:02d}.csv")
            save_to_csv(results, filename, save_midi=save_midi, minimal=minimal)

        print(f"    ✓ Completed {num_runs} runs")

    print()
    print(f"Experiment 1 complete: {len(transform_orders)} orders × {num_runs} runs = {len(transform_orders) * num_runs} walks")
    print()

    # Experiment 2: Fix transformation order, vary initial state
    print("Experiment 2: Varying initial state (fixed transform order LPR)")
    print("-"*70)

    # Define the 4 initial states
    initial_states = [
        ("up", None, 0),  # Pure up state
        ("down", None, 2),  # Pure down state
        ("right", None, 1),  # Pure right state
        ("equal_superposition", parse_initial_state("C:up+right+down"), 0)  # Equal superposition
    ]

    for state_name, superposition, spin in initial_states:
        print(f"  Initial state: {state_name}")

        for run in range(num_runs):
            # Offset seed to avoid overlap with experiment 1
            run_seed = (seed + 1000 + run) if seed is not None else None

            results = run_experiment(
                num_steps,
                initial_chord,
                initial_spin=spin,
                initial_superposition=superposition,
                transform_order="LPR",
                seed=run_seed
            )

            filename = os.path.join(output_dir, f"exp2_state_{state_name}_run{run+1:02d}.csv")
            save_to_csv(results, filename, save_midi=save_midi, minimal=minimal)

        print(f"    ✓ Completed {num_runs} runs")

    print()
    print(f"Experiment 2 complete: {len(initial_states)} states × {num_runs} runs = {len(initial_states) * num_runs} walks")
    print()
    print("="*70)
    print(f"TOTAL: {(len(transform_orders) + len(initial_states)) * num_runs} walks saved to {output_dir}")
    print("="*70)




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
    parser.add_argument('--initial-state', type=str,
                        help='Initial quantum state specification (e.g., "C:up+down" for symmetric superposition)')
    parser.add_argument('--transform-order', type=str, default='LPR',
                        help='Transformation order for spin states: permutation of LPR (default: LPR)')
    parser.add_argument('--parameter-sweep', action='store_true',
                        help='Run systematic parameter sweep experiments (overrides other settings)')
    parser.add_argument('--sweep-output-dir', type=str, default='parameter_sweep',
                        help='Output directory for parameter sweep results (default: parameter_sweep)')
    parser.add_argument('--no-midi', action='store_true',
                        help='Skip generating MIDI files (only save CSV)')
    parser.add_argument('--minimal-csv', action='store_true',
                        help='Save only step and chord columns (smaller files for large batches)')

    args = parser.parse_args()

    # Handle parameter sweep mode
    if args.parameter_sweep:
        run_parameter_sweep(
            num_steps=args.steps,
            num_runs=args.num_runs,
            output_dir=args.sweep_output_dir,
            seed=args.seed,
            save_midi=not args.no_midi,
            minimal=args.minimal_csv
        )
        return

    # Handle initial state specification
    initial_superposition = None
    if args.initial_state:
        try:
            initial_superposition = parse_initial_state(args.initial_state)
            initial_chord = initial_superposition[0][0]  # Use first chord for display
        except ValueError as e:
            print(f"Error parsing initial state: {e}")
            return
    else:
        # Parse starting chord from --root and --minor
        try:
            root_idx = Chord.NOTE_NAMES.index(args.root)
        except ValueError:
            print(f"Error: Invalid root note '{args.root}'. Valid notes: {', '.join(Chord.NOTE_NAMES)}")
            return
        initial_chord = Chord(root_idx, not args.minor, 0)

    # Validate transform order
    transform_order = args.transform_order.upper()
    if sorted(transform_order) != ['L', 'P', 'R']:
        print(f"Error: Invalid transform order '{args.transform_order}'. Must be a permutation of LPR.")
        print(f"Valid options: LPR, LRP, PLR, PRL, RLP, RPL")
        return

    print(f"Running qutrit walk experiments")
    if initial_superposition:
        print(f"Initial state: Custom superposition")
    else:
        print(f"Initial chord: {initial_chord}")
    print(f"Transform order: {transform_order}")
    print(f"Steps per run: {args.steps}")
    print(f"Number of runs: {args.num_runs}")
    if args.seed is not None:
        print(f"Random seed: {args.seed}")
    print()

    # Run experiments
    save_midi = not args.no_midi

    if args.num_runs == 1:
        # Single run - simple filename
        results = run_experiment(args.steps, initial_chord,
                                initial_superposition=initial_superposition,
                                transform_order=transform_order, seed=args.seed)
        save_to_csv(results, args.output, save_midi=save_midi, minimal=args.minimal_csv)
    else:
        # Multiple runs - add run number to filename
        base_filename = args.output.rsplit('.', 1)[0]
        ext = args.output.rsplit('.', 1)[1] if '.' in args.output else 'csv'

        for run in range(args.num_runs):
            run_seed = args.seed + run if args.seed is not None else None
            results = run_experiment(args.steps, initial_chord,
                                   initial_superposition=initial_superposition,
                                   transform_order=transform_order, seed=run_seed)

            filename = f"{base_filename}_run{run+1}.{ext}"
            save_to_csv(results, filename, save_midi=save_midi, minimal=args.minimal_csv)

    print("\nDone!")


if __name__ == "__main__":
    main()
