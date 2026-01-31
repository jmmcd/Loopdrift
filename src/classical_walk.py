"""
Classical random walk on the triadic Tonnetz graph.

Each step, the walker randomly chooses one of three neighbors with equal probability (1/3 each).
This provides a baseline for comparison with quantum walks.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from midi_generator import save_walk_to_midi, calculate_distinct_chords_ratio


# Chord names (matching experiments_qutrit.py)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

ALL_CHORDS = []
for root_name in NOTE_NAMES:
    ALL_CHORDS.append(root_name)  # Major chords
for root_name in NOTE_NAMES:
    ALL_CHORDS.append(root_name + 'm')  # Minor chords


def get_neo_riemannian_neighbors(chord_name):
    """
    Get the three neo-Riemannian neighbors of a chord.

    Returns:
        (L_neighbor, P_neighbor, R_neighbor) tuple
    """
    # Parse chord
    if chord_name.endswith('m'):
        root_name = chord_name[:-1]
        is_major = False
    else:
        root_name = chord_name
        is_major = True

    root_idx = NOTE_NAMES.index(root_name)

    if is_major:
        # Major chord transformations
        L_root = (root_idx + 4) % 12  # Leading-tone exchange: go to relative minor
        L_chord = NOTE_NAMES[L_root] + 'm'

        P_root = root_idx  # Parallel: same root, opposite mode
        P_chord = NOTE_NAMES[P_root] + 'm'

        R_root = (root_idx + 9) % 12  # Relative: go to relative minor
        R_chord = NOTE_NAMES[R_root] + 'm'
    else:
        # Minor chord transformations
        L_root = (root_idx - 4) % 12  # Leading-tone exchange: go to relative major
        L_chord = NOTE_NAMES[L_root]

        P_root = root_idx  # Parallel: same root, opposite mode
        P_chord = NOTE_NAMES[P_root]

        R_root = (root_idx - 9) % 12  # Relative: go to relative major
        R_chord = NOTE_NAMES[R_root]

    return L_chord, P_chord, R_chord


def classical_walk(initial_chord='C', num_steps=200, seed=None):
    """
    Perform a classical random walk on the chord graph.

    At each step, randomly choose one of three neighbors with probability 1/3 each.

    Args:
        initial_chord: Starting chord (e.g., 'C', 'Am', etc.)
        num_steps: Number of steps to take
        seed: Random seed for reproducibility

    Returns:
        List of chord names representing the walk
    """
    if seed is not None:
        np.random.seed(seed)

    walk = [initial_chord]
    current = initial_chord

    for _ in range(num_steps):
        # Get three neighbors
        L, P, R = get_neo_riemannian_neighbors(current)

        # Choose one uniformly at random
        neighbors = [L, P, R]
        current = np.random.choice(neighbors)
        walk.append(current)

    return walk


def save_walk_to_csv(walk, output_path, minimal=False):
    """
    Save walk to CSV file.

    Args:
        walk: List of chord names
        output_path: Path to save CSV file
        minimal: If True, only save step and current_chord (saves space for large batches)
    """
    data = []

    for step, chord in enumerate(walk):
        if minimal:
            # Minimal format: just step and chord
            row = {
                'step': step,
                'current_chord': chord,
            }
        else:
            # Full format: step, chord, and neighbors
            L, P, R = get_neo_riemannian_neighbors(chord)
            row = {
                'step': step,
                'current_chord': chord,
                'neighbor_L': L,
                'neighbor_P': P,
                'neighbor_R': R,
            }

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved walk to {output_path}")
    print(f"Total steps: {len(walk) - 1} (from step 0 to {len(walk) - 1})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate classical random walk on Tonnetz chord graph"
    )
    parser.add_argument('--initial', type=str, default='C',
                       help='Initial chord (default: C)')
    parser.add_argument('--steps', type=int, default=200,
                       help='Number of steps (default: 200)')
    parser.add_argument('--output', type=str, default='classical_walk.csv',
                       help='Output CSV file (default: classical_walk.csv)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--num-walks', type=int, default=1,
                       help='Number of walks to generate (default: 1)')
    parser.add_argument('--minimal-csv', action='store_true',
                       help='Save only step and chord columns (smaller files for large batches)')

    args = parser.parse_args()

    # Validate initial chord
    if args.initial not in ALL_CHORDS:
        print(f"Error: '{args.initial}' is not a valid chord name")
        print(f"Valid chords: {', '.join(ALL_CHORDS[:12])} (major)")
        print(f"             {', '.join(ALL_CHORDS[12:])} (minor)")
        return

    num_walks = getattr(args, 'num_walks')

    if num_walks == 1:
        # Single walk - use original behavior
        print(f"Starting classical random walk from {args.initial}")
        print(f"Number of steps: {args.steps}")
        if args.seed is not None:
            print(f"Random seed: {args.seed}")

        # Perform walk
        walk = classical_walk(
            initial_chord=args.initial,
            num_steps=args.steps,
            seed=args.seed
        )

        # Save to CSV
        save_walk_to_csv(walk, args.output, minimal=args.minimal_csv)

        # Save to MIDI
        midi_path = Path(args.output).with_suffix('.mid')
        save_walk_to_midi(walk, midi_path)
    else:
        # Multiple walks
        print(f"Generating {num_walks} classical random walks from {args.initial}")
        print(f"Number of steps per walk: {args.steps}")
        if args.seed is not None:
            print(f"Initial random seed: {args.seed}")

        # Prepare output filename pattern
        output_path = Path(args.output)
        stem = output_path.stem
        suffix = output_path.suffix
        parent = output_path.parent

        for i in range(num_walks):
            # Generate unique seed for each walk if base seed provided
            if args.seed is not None:
                walk_seed = args.seed + i
            else:
                walk_seed = None

            # Perform walk
            walk = classical_walk(
                initial_chord=args.initial,
                num_steps=args.steps,
                seed=walk_seed
            )

            # Generate output filenames
            output_file = parent / f"{stem}_{i+1}{suffix}"
            midi_file = parent / f"{stem}_{i+1}.mid"

            # Save to CSV and MIDI
            print(f"\nWalk {i+1}/{num_walks}:")
            if walk_seed is not None:
                print(f"  Seed: {walk_seed}")
            save_walk_to_csv(walk, output_file, minimal=args.minimal_csv)
            save_walk_to_midi(walk, midi_file)

        print(f"\nCompleted {num_walks} walks")


if __name__ == '__main__':
    main()
