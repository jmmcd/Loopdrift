# Plot Distinct Chords Analysis

This script analyzes and visualizes the exploration diversity of chord walks using a windowed distinct chords ratio metric.

## Overview

The script reads multiple walk CSV files (classical or quantum), calculates the distinct chords ratio over time, and plots the mean with standard deviation shading to compare exploration patterns.

## Metric: Distinct Chords Ratio

**Definition**: `(number of distinct chords in window) / (window size)`

- **Range**: 0.0 to 1.0
- **Interpretation**:
  - 1.0 = Maximum diversity (all chords in window are different)
  - Lower values = More repetition (revisiting same chords)
- **Window**: Sliding window that looks at the last N steps

## Usage

### Basic Usage

Compare classical and quantum walks with default settings (window=10, n=20):

```bash
python plot_distinct_chords.py \
  --classical-dir ../results/ClassicalWalkResults \
  --qutrit-dir ../results/QutritWalkSweepResults
```

### Custom Parameters

```bash
# Different window size
python plot_distinct_chords.py --window-size 20

# Analyze more/fewer walks
python plot_distinct_chords.py --num-walks 50

# Custom output path
python plot_distinct_chords.py --output my_analysis.pdf

# Show as separate subplots instead of combined
python plot_distinct_chords.py --separate
```

### Full Example

```bash
python plot_distinct_chords.py \
  --classical-dir /path/to/classical/walks \
  --qutrit-dir /path/to/quantum/walks \
  --window-size 15 \
  --num-walks 30 \
  --output results/exploration_analysis.pdf
```

## Output

The script generates two files:
- **PDF**: High-resolution vector graphics (300 DPI)
- **PNG**: Raster image for quick viewing (150 DPI)

### Plot Types

1. **Combined (default)**: Both walk types on the same axes
   - Blue: Classical random walks
   - Red: Quantum walks
   - Shaded regions show ±1 standard deviation

2. **Separate (--separate flag)**: Side-by-side subplots
   - Left: Classical walks
   - Right: Quantum walks

## CSV Format

The script expects CSV files with at least these columns:
- `step`: Step number
- `current_chord`: Chord name at this step

Works with:
- Classical walk CSVs (from `classical_walk.py`)
- Quantum walk CSVs (from `experiments_qutrit.py`)

## Example Analysis

```bash
# Generate 20 classical walks
python classical_walk.py --num-walks 20 --steps 200 --seed 42 \
  --output results/ClassicalWalkResults/walk.csv

# Generate 20 quantum walks
python experiments_qutrit.py --num-runs 20 --steps 200 --seed 42 \
  --output results/QutritWalkResults/walk.csv --no-midi

# Analyze and compare
python plot_distinct_chords.py \
  --classical-dir results/ClassicalWalkResults \
  --qutrit-dir results/QutritWalkResults \
  --window-size 10 \
  --output results/comparison.pdf
```

## Interpretation

- **Higher ratio** = More exploratory (visiting diverse chords)
- **Lower ratio** = More repetitive (revisiting same chords)
- **Declining trend** = Walk is settling into a subset of chords
- **Stable ratio** = Consistent exploration pattern

Comparing classical vs quantum walks can reveal differences in:
- Initial exploration rate
- Long-term diversity
- Convergence behavior
- Overall exploration efficiency

## Python API

```python
from plot_distinct_chords import load_walk_from_csv, plot_combined
from midi_generator import calculate_distinct_chords_ratio

# Load and analyze a single walk
chord_sequence = load_walk_from_csv('walk.csv')
ratios = calculate_distinct_chords_ratio(chord_sequence, window_size=10)

# Generate comparison plot
plot_combined(
    classical_dir='results/ClassicalWalkResults',
    qutrit_dir='results/QutritWalkSweepResults',
    window_size=10,
    num_walks=20,
    output_path='comparison.pdf'
)
```

## Notes

- The script automatically filters out `.mid` files when searching for walks
- If walks have different lengths, the plot shows data up to the shortest walk
- Standard deviation shading helps visualize the variability across different walks
- Window size should be adjusted based on walk length (typically 5-10% of total steps)
