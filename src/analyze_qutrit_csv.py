#!/usr/bin/env python3
"""
Example script for reading and analyzing qutrit quantum walk CSV data
"""
import csv
import numpy as np
import argparse


def read_quantum_walk_csv(filename):
    """
    Read qutrit walk CSV and return structured data

    Returns:
        List of dicts with:
        - step: int
        - current_chord: str
        - neighbors: list of 3 strings (or empty for first step)
        - marginal_probs: dict of {chord_name: probability}
        - full_state: dict of {(chord_name, spin): complex amplitude}
    """
    results = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)

        # Get chord and state column names
        chord_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B',
                      'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'Bbm', 'Bm']
        spin_names = ['up', 'right', 'down']

        for row in reader:
            # Parse metadata
            step_data = {
                'step': int(row['step']),
                'current_chord': row['current_chord'],
                'neighbors': [row['neighbor_L'], row['neighbor_P'], row['neighbor_R']]
                            if row['neighbor_L'] else []
            }

            # Parse marginal probabilities
            marginal_probs = {}
            for chord in chord_names:
                marginal_probs[chord] = float(row[chord])
            step_data['marginal_probs'] = marginal_probs

            # Parse full quantum state
            full_state = {}
            for chord in chord_names:
                for spin in spin_names:
                    re = float(row[f'{chord}_{spin}_re'])
                    im = float(row[f'{chord}_{spin}_im'])
                    full_state[(chord, spin)] = complex(re, im)
            step_data['full_state'] = full_state

            results.append(step_data)

    return results


def analyze_state(step_data):
    """Print analysis of quantum state at one step"""
    print(f"\n{'='*70}")
    print(f"Step {step_data['step']}: {step_data['current_chord']}")
    print(f"{'='*70}")

    if step_data['neighbors']:
        print(f"Neighbors: {', '.join(step_data['neighbors'])}")
        print()

    # Show top marginal chord probabilities
    probs = sorted(step_data['marginal_probs'].items(), key=lambda x: x[1], reverse=True)
    print("Top chord probabilities (marginal over spins):")
    for chord, prob in probs[:5]:
        if prob > 0.001:
            print(f"  {chord:5s}: {prob:.6f}")

    # Verify state normalization
    total_prob = sum(abs(amp)**2 for amp in step_data['full_state'].values())
    print(f"\nState normalization: {total_prob:.9f} (should be 1.0)")

    # Show spin distribution
    spin_probs = {'up': 0.0, 'right': 0.0, 'down': 0.0}
    for (chord, spin), amp in step_data['full_state'].items():
        spin_probs[spin] += abs(amp)**2

    print(f"\nSpin distribution:")
    print(f"  ↑ (up):    {spin_probs['up']:.6f}")
    print(f"  → (right): {spin_probs['right']:.6f}")
    print(f"  ↓ (down):  {spin_probs['down']:.6f}")


def compute_entropy(step_data):
    """Compute von Neumann entropy of the state"""
    # Build density matrix (simplified: use marginal chord distribution)
    probs = np.array([p for p in step_data['marginal_probs'].values() if p > 1e-10])
    # S = -sum(p * log2(p))
    entropy = -np.sum(probs * np.log2(probs))
    return entropy


def main():
    parser = argparse.ArgumentParser(
        description="Analyze qutrit quantum walk CSV data"
    )
    parser.add_argument('csv_file', help='CSV file to analyze')
    parser.add_argument('--step', type=int, help='Analyze specific step (default: all)')
    parser.add_argument('--entropy', action='store_true',
                       help='Compute entropy over time')

    args = parser.parse_args()

    # Read data
    print(f"Reading {args.csv_file}...")
    data = read_quantum_walk_csv(args.csv_file)
    print(f"Loaded {len(data)} steps")

    if args.entropy:
        # Plot entropy over time
        print("\nEntropy (bits) over time:")
        print("Step | Entropy")
        print("-----|--------")
        for step_data in data:
            ent = compute_entropy(step_data)
            print(f"{step_data['step']:4d} | {ent:.4f}")

    elif args.step is not None:
        # Analyze specific step
        if 0 <= args.step < len(data):
            analyze_state(data[args.step])
        else:
            print(f"Error: Step {args.step} out of range (0-{len(data)-1})")

    else:
        # Analyze all steps
        for step_data in data:
            analyze_state(step_data)


if __name__ == "__main__":
    main()
