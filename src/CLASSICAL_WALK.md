# Classical Random Walk on Tonnetz

## Overview

The `classical_walk.py` script generates classical random walks on the triadic Tonnetz chord graph. At each step, the walker randomly selects one of three neo-Riemannian neighbors (L, P, R) with equal probability (1/3 each).

This provides a baseline for comparison with quantum walks, which can exhibit different exploration patterns due to quantum interference.

## Usage

### Basic Usage

Generate a random walk starting from C major:

```bash
python classical_walk.py
```

This creates `classical_walk.csv` with 200 steps.

### Custom Parameters

```bash
# Start from a different chord
python classical_walk.py --initial Am

# Different number of steps
python classical_walk.py --steps 500

# Custom output filename
python classical_walk.py --output my_walk.csv

# Set random seed for reproducibility
python classical_walk.py --seed 42

# Generate multiple walks at once
python classical_walk.py --num-walks 10 --output walk.csv
# This creates walk_1.csv, walk_2.csv, ..., walk_10.csv
```

### Combined Options

```bash
python classical_walk.py --initial G --steps 1000 --seed 123 --output g_walk.csv

# Generate multiple walks with different seeds
python classical_walk.py --num-walks 5 --seed 100 --steps 300 --output experiment.csv
# Creates experiment_1.csv (seed 100), experiment_2.csv (seed 101), etc.
```

## Examples

```bash
# Generate three walks with different random seeds (new method)
python classical_walk.py --num-walks 3 --seed 1 --output walk.csv
# Creates walk_1.csv (seed 1), walk_2.csv (seed 2), walk_3.csv (seed 3)

# Generate three walks with different random seeds (old method also works)
python classical_walk.py --seed 1 --output walk1.csv
python classical_walk.py --seed 2 --output walk2.csv
python classical_walk.py --seed 3 --output walk3.csv

# Generate 10 walks of 500 steps each
python classical_walk.py --num-walks 10 --steps 500 --seed 42 --output long_walk.csv
# Creates long_walk_1.csv through long_walk_10.csv

# Compare different starting positions (same seed)
python classical_walk.py --initial C --seed 42 --output walk_C.csv
python classical_walk.py --initial Am --seed 42 --output walk_Am.csv
python classical_walk.py --initial G --seed 42 --output walk_G.csv
```

## Output Format

The CSV file contains:
- `step`: Step number (0 to num_steps)
- `current_chord`: Current chord name
- `neighbor_L`: L transformation neighbor (Leading-tone exchange)
- `neighbor_P`: P transformation neighbor (Parallel)
- `neighbor_R`: R transformation neighbor (Relative)

Example:
```csv
step,current_chord,neighbor_L,neighbor_P,neighbor_R
0,C,Em,Cm,Am
1,Am,F,A,C
2,F,Am,Fm,Dm
...
```

## Visualization

The output can be visualized using `visualize_tonnetz.py`:

```bash
# Generate and visualize a classical walk
python classical_walk.py --seed 42
python visualize_tonnetz.py classical_walk.csv
```

This creates `classical_walk_viz.pdf` showing the trajectory on the Tonnetz.

## Neo-Riemannian Transformations

The three transformations are:

- **L (Leading-tone exchange)**: Moves between relative major/minor chords
  - C major → E minor
  - A minor → F major

- **P (Parallel)**: Changes mode while keeping the root
  - C major → C minor
  - A minor → A major

- **R (Relative)**: Moves to the relative major/minor
  - C major → A minor
  - A minor → C major

## Comparison with Quantum Walks

Key differences:

| Feature | Classical Walk | Quantum Walk |
|---------|---------------|--------------|
| **Transition** | Probabilistic (1/3 each) | Quantum superposition + measurement |
| **State** | Single chord at a time | Superposition of all 24 chords |
| **Interference** | No interference | Quantum interference effects |
| **Exploration** | Purely random | Can show directional bias |
| **Reproducibility** | Via random seed | Via initial quantum state |

## Valid Chord Names

**Major chords**: C, C#, D, D#, E, F, F#, G, G#, A, A#, B

**Minor chords**: Cm, C#m, Dm, D#m, Em, Fm, F#m, Gm, G#m, Am, A#m, Bm

## Python API

```python
from classical_walk import classical_walk, save_walk_to_csv

# Generate a walk
walk = classical_walk(initial_chord='C', num_steps=200, seed=42)

# Save to CSV
save_walk_to_csv(walk, 'my_walk.csv')
```

## Notes

- Each walk is memoryless - the next chord depends only on the current chord, not the history
- All three neighbors are equally likely at each step (uniform distribution)
- The graph is connected, so any chord can eventually reach any other chord
- Expected return time to starting chord: related to graph structure (mixing time)
