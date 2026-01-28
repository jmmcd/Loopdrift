# Initial State Specification for Qutrit Quantum Walk

The qutrit quantum walk supports flexible initial state specifications using the `--initial-state` command-line argument. This allows you to explore different quantum superposition states beyond simple pure states.

## Basic Syntax

The general format is:

```
chord_spec:spin_spec
```

- `chord_spec`: Chord name (e.g., `C`, `Am`, `F#`) or chord superposition (e.g., `C+Am+F`)
- `spin_spec`: Spin state specification (e.g., `up`, `down`, `up+down`)

## Spin States

The three qutrit basis states correspond to Neo-Riemannian transformations:
- `up` (|↑⟩): Leittonwechsel (L) - changes the root note
- `right` (|→⟩): Parallel (P) - changes major/minor
- `down` (|↓⟩): Relative (R) - changes the fifth

## Examples

### 1. Pure States

Single chord, single spin:

```bash
# C major in spin up state: |C, ↑⟩
python QutritWalk.py --initial-state "C:up"

# A minor in spin down state: |Am, ↓⟩
python QutritWalk.py --initial-state "Am:down"

# F# major in spin right state: |F#, →⟩
python QutritWalk.py --initial-state "F#:right"
```

### 2. Spin Superpositions

Single chord, multiple spins (equal superposition):

```bash
# Symmetric up/down superposition: (|C,↑⟩ + |C,↓⟩)/√2
python QutritWalk.py --initial-state "C:up+down"

# Three-way equal superposition: (|C,↑⟩ + |C,→⟩ + |C,↓⟩)/√3
python QutritWalk.py --initial-state "C:up+right+down"
```

### 3. Custom Real Amplitudes

Specify custom real-valued coefficients:

```bash
# Weighted superposition with custom amplitudes (will be normalized)
python QutritWalk.py --initial-state "C:0.5*up+0.5*down"

# Different weights
python QutritWalk.py --initial-state "Am:0.8*up+0.2*down"
```

### 4. Complex-Valued Amplitudes

Use `i` to specify imaginary components:

```bash
# Complex superposition: (|C,↑⟩ + i|C,↓⟩)/√2
python QutritWalk.py --initial-state "C:1*up+i*down"

# Multiple complex terms
python QutritWalk.py --initial-state "C:1*up+0.5*right+0.5i*down"

# Just imaginary coefficient
python QutritWalk.py --initial-state "Am:up+i*right"
```

### 5. Chord Superpositions

Multiple chords with same spin state:

```bash
# Equal superposition over three chords: (|C,↑⟩ + |Am,↑⟩ + |F,↑⟩)/√3
python QutritWalk.py --initial-state "C+Am+F:up"

# Major and minor version of same root
python QutritWalk.py --initial-state "C+Cm:down"
```

### 6. Combined Chord and Spin Superpositions

Combine chord and spin superpositions:

```bash
# Each chord in spin superposition: (|C,↑⟩ + |C,↓⟩ + |Am,↑⟩ + |Am,↓⟩)/2
python QutritWalk.py --initial-state "C+Am:up+down"
```

## Usage with Experiments

The same syntax works with the experiments script:

```bash
# Run 100 steps starting from symmetric up/down superposition
python experiments_qutrit.py --steps 100 --initial-state "C:up+down" --output symmetric.csv

# Run 10 independent trials with complex initial state
python experiments_qutrit.py --steps 100 --num-runs 10 --initial-state "C:1*up+i*down" --seed 42 --output complex.csv
```

## CSV Output Format

The experiments script outputs CSV files with complete quantum state information:

**Column structure:**
1. **Metadata** (5 columns):
   - `step`: Step number
   - `current_chord`: Current chord at this step
   - `neighbor_L`, `neighbor_P`, `neighbor_R`: The three neighbor chords (empty for first step)

2. **Marginal chord probabilities** (24 columns):
   - One column per chord: `C`, `C#`, `D`, ..., `B`, `Cm`, `C#m`, ..., `Bm`
   - Values are probabilities P(chord) = sum over all spins of |amplitude|²
   - Useful for quick analysis of chord distribution

3. **Full quantum state** (144 columns = 72 × 2):
   - For each of 72 basis states (24 chords × 3 spins), two columns:
   - `{chord}_{spin}_re`: Real part of complex amplitude
   - `{chord}_{spin}_im`: Imaginary part of complex amplitude
   - Example: `C_up_re`, `C_up_im`, `C_right_re`, `C_right_im`, `C_down_re`, `C_down_im`
   - Complete quantum state information preserving all phase relationships

**Total: 173 columns**

### Analyzing CSV Data

An example analysis script is provided in [analyze_qutrit_csv.py](analyze_qutrit_csv.py):

```bash
# Analyze specific step
python analyze_qutrit_csv.py results.csv --step 10

# Compute entropy over time
python analyze_qutrit_csv.py results.csv --entropy

# Analyze all steps
python analyze_qutrit_csv.py results.csv
```

You can also read the CSV directly in Python:

```python
import csv
import numpy as np

with open('results.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        step = int(row['step'])

        # Get marginal probabilities
        p_C = float(row['C'])

        # Get complex amplitudes
        c_up_re = float(row['C_up_re'])
        c_up_im = float(row['C_up_im'])
        c_up_amplitude = complex(c_up_re, c_up_im)

        # Reconstruct full state vector (72 dimensions)
        state = np.zeros(72, dtype=complex)
        # ... (see analyze_qutrit_csv.py for full example)
```

## Notes

1. **Automatic Normalization**: All states are automatically normalized to unit length, so you don't need to worry about exact normalization when specifying coefficients.

2. **Chord Names**: Use standard note names with optional 'm' for minor:
   - Major: `C`, `C#`, `D`, `D#`, `E`, `F`, `F#`, `G`, `G#`, `A`, `Bb`, `B`
   - Minor: `Cm`, `C#m`, `Dm`, `D#m`, `Em`, `Fm`, `F#m`, `Gm`, `G#m`, `Am`, `Bbm`, `Bm`

3. **Complex Numbers**: Use `i` for the imaginary unit (Python's `j` is also accepted internally but `i` is more natural for mathematical notation).

4. **Separator**: Use `+` to separate multiple chords or multiple spins, not commas.

5. **Whitespace**: Spaces around separators and operators are optional and will be ignored.

## Physical Interpretation

Different initial states lead to different quantum interference patterns during the walk:

- **Pure states** (e.g., `C:up`) start with definite chord and spin
- **Spin superpositions** (e.g., `C:up+down`) create quantum interference between different transformation paths
- **Complex amplitudes** (e.g., `C:up+i*down`) introduce quantum phase differences
- **Chord superpositions** (e.g., `C+Am+F:up`) start the walk simultaneously at multiple locations on the tonnetz

These different initial conditions will result in different probability distributions over chord progressions, allowing you to explore the full richness of the quantum walk dynamics.
