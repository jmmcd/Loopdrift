# Parameter Sweep Experiments

The `--parameter-sweep` mode runs systematic experiments to explore the parameter space of the qutrit quantum walk.

## Overview

The parameter sweep runs two sets of experiments:

**Experiment 1:** Varying transformation order (fixed initial state)
- Initial state: C major, spin up (C:up)
- Transformation orders: All 6 permutations (LPR, LRP, PLR, PRL, RLP, RPL)

**Experiment 2:** Varying initial state (fixed transformation order)
- Transformation order: LPR
- Initial states:
  1. Pure up: C:up
  2. Pure down: C:down
  3. Pure right: C:right
  4. Equal superposition: (|C,↑⟩ + |C,→⟩ + |C,↓⟩)/√3

## Usage

```bash
python experiments_qutrit.py --parameter-sweep \
    --steps 1000 \
    --num-runs 20 \
    --sweep-output-dir results \
    --seed 42
```

### Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--parameter-sweep` | Enable parameter sweep mode | (flag) |
| `--steps` | Number of steps per walk | 100 |
| `--num-runs` | Number of independent runs per configuration | 1 |
| `--sweep-output-dir` | Output directory for results | parameter_sweep |
| `--seed` | Base random seed for reproducibility | (none) |

### Output

The sweep creates CSV files with systematic naming:

```
results/
├── exp1_order_LPR_run01.csv
├── exp1_order_LPR_run02.csv
├── ...
├── exp1_order_RPL_run20.csv
├── exp2_state_up_run01.csv
├── exp2_state_down_run01.csv
├── exp2_state_right_run01.csv
├── exp2_state_equal_superposition_run01.csv
└── ...
```

**Filename format:**
- Experiment 1: `exp1_order_{ORDER}_run{N:02d}.csv`
- Experiment 2: `exp2_state_{STATE}_run{N:02d}.csv`

## Total Walks

With default settings (steps=1000, num-runs=20):
- Experiment 1: 6 orders × 20 runs = **120 walks**
- Experiment 2: 4 states × 20 runs = **80 walks**
- **Total: 200 walks**

## Example: Full Parameter Sweep

```bash
# Run the full systematic exploration
python experiments_qutrit.py --parameter-sweep \
    --steps 1000 \
    --num-runs 20 \
    --sweep-output-dir parameter_sweep_results \
    --seed 42
```

This generates 200 CSV files, each containing a 1000-step quantum walk with full state information.

## Seeding Strategy

The function uses a smart seeding strategy to ensure reproducibility while avoiding overlap:

- **Experiment 1 runs:** `seed + run_number` (e.g., 42, 43, 44, ...)
- **Experiment 2 runs:** `seed + 1000 + run_number` (e.g., 1042, 1043, 1044, ...)

This ensures that each walk has a unique seed, while still being reproducible across runs.

## Data Analysis

After running the sweep, you can analyze the results to answer questions like:

### Experiment 1 Questions
- Which transformation ordering produces the highest entropy?
- Do certain orderings favor major vs. minor chords?
- How does the chord progression diversity vary across orderings?
- Are there statistical signatures that distinguish the orderings?

### Experiment 2 Questions
- How does the initial spin state affect the long-term distribution?
- Does the equal superposition explore more of the tonnetz?
- Do pure states converge to different equilibrium distributions?
- How does quantum interference in the superposition affect the walk?

### Example Analysis Script

```python
import pandas as pd
import glob

# Load all Experiment 1 results
exp1_files = glob.glob("results/exp1_order_*_run*.csv")

for order in ['LPR', 'LRP', 'PLR', 'PRL', 'RLP', 'RPL']:
    order_files = [f for f in exp1_files if f"order_{order}" in f]

    # Compute statistics across runs
    entropies = []
    for file in order_files:
        df = pd.read_csv(file)
        # Compute entropy from marginal chord probabilities
        # ... analysis code ...

    print(f"{order}: mean entropy = {np.mean(entropies):.3f}")
```

## File Size Estimates

With 173 columns per row:
- **Small sweep** (steps=100, runs=5): ~10 MB total
- **Medium sweep** (steps=1000, runs=20): ~130 MB total
- **Large sweep** (steps=10000, runs=20): ~1.3 GB total

Consider compressing the output directory after running large sweeps.

## Tips

1. **Start small**: Test with `--steps 10 --num-runs 2` first to verify everything works
2. **Use seeds**: Always use `--seed` for reproducible experiments
3. **Parallel processing**: The runs are independent and could be parallelized (future work)
4. **Incremental analysis**: Analyze results after smaller sweeps before running large ones
5. **Disk space**: Check available space before running large sweeps

## Advanced Usage

You can combine parameter sweep with other analysis tools:

```bash
# Run sweep
python experiments_qutrit.py --parameter-sweep \
    --steps 1000 --num-runs 20 --seed 42

# Analyze each configuration
for file in parameter_sweep/exp1_order_*.csv; do
    python analyze_qutrit_csv.py "$file" --entropy >> entropy_results.txt
done

# Compare results
python compare_configurations.py parameter_sweep/
```

## Notes

- The parameter sweep mode **overrides** individual experiment settings (--initial-state, --transform-order, --output)
- All walks start from C major chord
- Each CSV file contains the complete quantum state evolution (173 columns)
- The naming convention makes it easy to identify and group results programmatically
