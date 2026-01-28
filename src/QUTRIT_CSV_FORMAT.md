# Qutrit Quantum Walk CSV Format

The `experiments_qutrit.py` script outputs CSV files containing complete quantum state information for each step of the walk.

## Column Structure

Each CSV has **173 columns** organized as follows:

### 1. Metadata (5 columns)

| Column | Type | Description |
|--------|------|-------------|
| `step` | int | Step number (0, 1, 2, ...) |
| `current_chord` | string | Current chord at this step (e.g., "C", "Am") |
| `neighbor_L` | string | Leittonwechsel neighbor (empty for step 0) |
| `neighbor_P` | string | Parallel neighbor (empty for step 0) |
| `neighbor_R` | string | Relative neighbor (empty for step 0) |

### 2. Marginal Chord Probabilities (24 columns)

One column for each chord in the tonnetz:

**Major chords (12 columns):** `C`, `C#`, `D`, `D#`, `E`, `F`, `F#`, `G`, `G#`, `A`, `Bb`, `B`

**Minor chords (12 columns):** `Cm`, `C#m`, `Dm`, `D#m`, `Em`, `Fm`, `F#m`, `Gm`, `G#m`, `Am`, `Bbm`, `Bm`

**Values:** Each entry is P(chord) = Σ_spin |amplitude(chord, spin)|²

These probabilities are obtained by marginalizing (summing) over all three spin states.

### 3. Full Quantum State (144 columns)

For each of the 72 basis states (24 chords × 3 spins), we store two columns:

- `{chord}_{spin}_re`: Real part of complex amplitude
- `{chord}_{spin}_im`: Imaginary part of complex amplitude

**Spin names:** `up`, `right`, `down` (corresponding to L, P, R transformations)

**Example columns:**
```
C_up_re, C_up_im, C_right_re, C_right_im, C_down_re, C_down_im,
Cm_up_re, Cm_up_im, Cm_right_re, Cm_right_im, Cm_down_re, Cm_down_im,
...
Bm_up_re, Bm_up_im, Bm_right_re, Bm_right_im, Bm_down_re, Bm_down_im
```

**Complete state vector:** The full quantum state is |ψ⟩ = Σ amplitude(chord, spin) |chord, spin⟩

**Normalization:** ⟨ψ|ψ⟩ = Σ |amplitude|² = 1

## Example Usage

### Reading in Python

```python
import csv
import numpy as np

def read_quantum_state(csv_file, step_num):
    """Read quantum state at specific step"""
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row['step']) == step_num:
                # Get marginal probabilities
                chord_probs = {}
                for chord in ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B',
                             'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'Bbm', 'Bm']:
                    chord_probs[chord] = float(row[chord])

                # Get full quantum state
                state = {}
                for chord in chord_probs.keys():
                    for spin in ['up', 'right', 'down']:
                        re = float(row[f'{chord}_{spin}_re'])
                        im = float(row[f'{chord}_{spin}_im'])
                        state[(chord, spin)] = complex(re, im)

                return chord_probs, state

    return None, None

# Example: analyze step 10
probs, state = read_quantum_state('results.csv', 10)

# Find most likely chord
max_chord = max(probs.items(), key=lambda x: x[1])
print(f"Most likely chord: {max_chord[0]} with P = {max_chord[1]:.4f}")

# Verify normalization
total = sum(abs(amp)**2 for amp in state.values())
print(f"State normalization: {total:.9f}")

# Get amplitude for specific basis state
amp_C_up = state[('C', 'up')]
print(f"|C, ↑⟩ amplitude: {amp_C_up.real:.6f} + {amp_C_up.imag:.6f}i")
print(f"|C, ↑⟩ probability: {abs(amp_C_up)**2:.6f}")
```

### Computing Observables

```python
import numpy as np

# Compute spin expectation values
def compute_spin_distribution(state):
    """Compute marginal spin distribution"""
    spin_probs = {'up': 0.0, 'right': 0.0, 'down': 0.0}
    for (chord, spin), amp in state.items():
        spin_probs[spin] += abs(amp)**2
    return spin_probs

# Compute von Neumann entropy
def compute_entropy(chord_probs):
    """Compute entropy of marginal chord distribution"""
    probs = np.array([p for p in chord_probs.values() if p > 1e-10])
    return -np.sum(probs * np.log2(probs))

# Compute purity
def compute_purity(state):
    """Compute quantum purity Tr(ρ²)"""
    # For pure state, purity = 1
    # For mixed state (after partial trace), purity < 1
    return sum(abs(amp)**4 for amp in state.values())
```

### Using the Analysis Script

A complete analysis script is provided in `analyze_qutrit_csv.py`:

```bash
# Analyze specific step
python analyze_qutrit_csv.py results.csv --step 10

# Compute entropy over time
python analyze_qutrit_csv.py results.csv --entropy

# Analyze all steps
python analyze_qutrit_csv.py results.csv
```

## Data Size

For reference:
- **Small experiment** (100 steps): ~65 KB
- **Medium experiment** (1000 steps): ~650 KB
- **Large experiment** (10000 steps): ~6.5 MB

The file size scales linearly with the number of steps. For very long runs, consider:
1. Using `--num-runs` to create multiple shorter files instead of one long file
2. Compressing output files (CSV compresses very well: ~10x compression typical)
3. Post-processing to extract only needed observables

## Relationship Between Columns

**Key property:** The marginal probabilities are derived from the full state:

```
P(chord) = Σ_spin |amplitude(chord, spin)|²

For example:
P(C) = |C_up|² + |C_right|² + |C_down|²
     = (C_up_re² + C_up_im²) + (C_right_re² + C_right_im²) + (C_down_re² + C_down_im²)
```

This redundancy makes it easy to work with marginal probabilities while preserving full quantum information for deeper analysis.

## Quantum State Properties

At every step, the quantum state satisfies:

1. **Normalization:** Σ |amplitude|² = 1
2. **Complex-valued:** Each amplitude has real and imaginary parts
3. **Coherent superposition:** Phase relationships between amplitudes create quantum interference
4. **No collapse:** The quantum state continues evolving; only the sampled chord is measured

The CSV format preserves all this information, allowing you to:
- Reconstruct the exact quantum state at any step
- Compute any quantum observable
- Analyze quantum interference patterns
- Study decoherence and entanglement (if extended to multiple walkers)
