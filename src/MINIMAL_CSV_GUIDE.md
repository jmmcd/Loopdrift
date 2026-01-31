# Minimal CSV Format Guide

## Overview

For large batch experiments (hundreds or thousands of walks), the `--minimal-csv` flag dramatically reduces file sizes by saving only the essential chord sequence data.

## Space Savings

| Walk Type | Full CSV | Minimal CSV | Space Saved |
|-----------|----------|-------------|-------------|
| **Classical Walk** (200 steps) | 3.0 KB | 1.3 KB | **57%** (2.3x smaller) |
| **Quantum Walk** (200 steps) | 286 KB | 1.4 KB | **99.5%** (204x smaller) |

### For Large Batches

**1000 classical walks of 200 steps:**
- Full format: ~3 MB
- Minimal format: ~1.3 MB
- **Saves 1.7 MB**

**1000 quantum walks of 200 steps:**
- Full format: ~286 MB
- Minimal format: ~1.4 MB
- **Saves 285 MB** (99.5% reduction!)

## Usage

### Classical Walks

```bash
# Single walk with minimal CSV
python classical_walk.py --steps 200 --minimal-csv --output walk.csv

# Large batch (1000 walks)
python classical_walk.py --num-walks 1000 --steps 200 --minimal-csv \
  --seed 42 --output results/batch/walk.csv
```

### Quantum Walks

```bash
# Single walk with minimal CSV
python experiments_qutrit.py --steps 200 --minimal-csv --output walk.csv

# Large batch (1000 walks, no MIDI for extra space savings)
python experiments_qutrit.py --num-runs 1000 --steps 200 \
  --minimal-csv --no-midi --seed 42 --output results/batch/walk.csv
```

## CSV Format Comparison

### Full Format

**Classical Walk:**
```csv
step,current_chord,neighbor_L,neighbor_P,neighbor_R
0,C,Em,Cm,Am
1,Am,F,A,C
2,F,Am,Fm,Dm
```

**Quantum Walk:**
```csv
step,current_chord,neighbor_L,neighbor_P,neighbor_R,C,C#,D,D#,E,F,...(173 columns total)
0,C,Em,Cm,Am,0.0,0.0,0.0,0.0,0.0,0.0,...
1,Cm,D#,C,G#,0.296,0.0,0.0,0.197,0.049,0.049,...
```

### Minimal Format

**Both Classical and Quantum:**
```csv
step,current_chord
0,C
1,Am
2,F
```

## Analysis Compatibility

The minimal CSV format is **fully compatible** with all analysis tools:

✅ `plot_distinct_chords.py` - Works seamlessly
✅ `visualize_tonnetz.py` - Extracts chord sequence
✅ `calculate_distinct_chords_ratio()` - Only needs chord sequence
✅ Custom analysis scripts - Just read `current_chord` column

## When to Use Minimal CSV

**Use minimal CSV when:**
- Running large batches (100+ walks)
- Disk space is limited
- You only need the chord sequence for analysis
- Generating walks for statistical analysis

**Use full CSV when:**
- You need neighbor information for analysis
- You want quantum state amplitudes (quantum walks)
- You need probability distributions (quantum walks)
- Running small experiments (< 50 walks)

## Recommendation for 1000 Walks

For your planned 1000-walk experiment:

```bash
# Classical walks (saves ~1.7 MB)
python classical_walk.py --num-walks 1000 --steps 200 --minimal-csv \
  --seed 42 --output results/classical_batch_1000/walk.csv

# Quantum walks (saves ~285 MB!)
python experiments_qutrit.py --num-runs 1000 --steps 200 \
  --minimal-csv --no-midi --seed 42 \
  --output results/quantum_batch_1000/walk.csv
```

**Total space saved: ~287 MB** for 2000 walks!

## Example Analysis

```python
import pandas as pd
from midi_generator import calculate_distinct_chords_ratio

# Load minimal CSV
df = pd.read_csv('walk_1.csv')
chord_sequence = df['current_chord'].tolist()

# Calculate exploration metric
ratios = calculate_distinct_chords_ratio(chord_sequence, window_size=20)

# Works exactly the same as with full CSV!
```

## Notes

- Minimal CSV files are still human-readable
- MIDI files are independent of CSV format
- You can always regenerate full format if needed (same seed)
- Analysis scripts automatically handle both formats
