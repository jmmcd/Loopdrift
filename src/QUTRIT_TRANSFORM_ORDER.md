# Transformation Order Configuration

The qutrit quantum walk maps three spin states to three Neo-Riemannian transformations. The mapping between spins and transformations is configurable, allowing you to explore different quantum walk dynamics.

## Overview

The three qutrit basis states are:
- |↑⟩ (up / spin 0)
- |→⟩ (right / spin 1)
- |↓⟩ (down / spin 2)

The three Neo-Riemannian transformations are:
- **L** (Leittonwechsel): Changes the root note
- **P** (Parallel): Changes major ↔ minor
- **R** (Relative): Changes the fifth

## Default Mapping (LPR)

By default, the mapping is:
- |↑⟩ → **L**eittonwechsel
- |→⟩ → **P**arallel
- |↓⟩ → **R**elative

This is specified as `--transform-order LPR`.

## Alternative Orderings

There are **6 possible permutations** of {L, P, R}:

| Order | |↑⟩ maps to | |→⟩ maps to | |↓⟩ maps to |
|-------|-----------|-----------|-----------|
| **LPR** | L | P | R |
| **LRP** | L | R | P |
| **PLR** | P | L | R |
| **PRL** | P | R | L |
| **RLP** | R | L | P |
| **RPL** | R | P | L |

Each ordering produces a different quantum walk with different interference patterns and statistical properties.

## Usage

### MIDI Playback

```bash
# Default LPR ordering
python src/QutritWalk.py --initial-state "C:up+down" --midi-port 0

# Alternative RLP ordering
python src/QutritWalk.py --initial-state "C:up+down" --transform-order RLP --midi-port 0

# Try PRL ordering
python src/QutritWalk.py --initial-state "C:up+down" --transform-order PRL --midi-port 0
```

### Statistical Experiments

```bash
# Compare all 6 orderings with same initial state
python src/experiments_qutrit.py --steps 1000 --initial-state "C:up+down" --transform-order LPR --output lpr.csv --seed 42
python src/experiments_qutrit.py --steps 1000 --initial-state "C:up+down" --transform-order LRP --output lrp.csv --seed 42
python src/experiments_qutrit.py --steps 1000 --initial-state "C:up+down" --transform-order PLR --output plr.csv --seed 42
python src/experiments_qutrit.py --steps 1000 --initial-state "C:up+down" --transform-order PRL --output prl.csv --seed 42
python src/experiments_qutrit.py --steps 1000 --initial-state "C:up+down" --transform-order RLP --output rlp.csv --seed 42
python src/experiments_qutrit.py --steps 1000 --initial-state "C:up+down" --transform-order RPL --output rpl.csv --seed 42
```

## Why Does This Matter?

The transformation order affects:

1. **Quantum Interference Patterns**: Different orderings create different phase relationships during the quantum walk evolution.

2. **Chord Progression Statistics**: The probability distributions over chord transitions will be different for each ordering.

3. **Musical Character**: Different orderings may emphasize different types of harmonic motion (chromatic vs. modal vs. functional).

4. **Symmetry Breaking**: While the tonnetz graph structure is the same, the quantum dynamics break symmetry differently depending on the ordering.

## Example: Comparing LPR vs RLP

Starting from the same initial state `C:up+down` with seed 42:

```
Step  LPR Order  RLP Order
----  ---------  ---------
0     C          C
1     C          C
2     Cm         Em        ← Diverge
3     G#         C
4     G#m        Cm
5     G#         G#        ← Reconverge
6     Cm         Cm
7     D#         D#
8     Gm         Gm
9     D#         D#
```

Even with the same random seed, the walks diverge because the quantum evolution is fundamentally different.

## Physical Interpretation

You can think of the transformation order as defining the "quantum rules" of the walk:

- **LPR (default)**: "Spin up explores chromatically (L), spin right explores mode (P), spin down explores functionally (R)"
- **RLP**: "Spin up explores functionally (R), spin right explores chromatically (L), spin down explores mode (P)"
- etc.

The Grover coin creates quantum superpositions of these three directions, but the *meaning* of each direction changes with the transformation order.

## Validation

The system validates that your transform order is a valid permutation:

```bash
# Valid
python src/QutritWalk.py --transform-order LPR
python src/QutritWalk.py --transform-order plr  # Case insensitive

# Invalid - will error
python src/QutritWalk.py --transform-order LLL  # Duplicate letters
python src/QutritWalk.py --transform-order ABC  # Wrong letters
python src/QutritWalk.py --transform-order LP   # Incomplete
```

Error message:
```
Error: Invalid transform order 'LLL'. Must be a permutation of LPR.
Valid options: LPR, LRP, PLR, PRL, RLP, RPL
```

## Research Questions

This parameter enables exploring questions like:

1. Which transformation ordering produces the most "interesting" chord progressions?
2. Do certain orderings favor major vs. minor chords?
3. How does entropy evolution differ between orderings?
4. Are there statistical signatures that distinguish the orderings?
5. Do certain orderings produce more "tonal" vs "atonal" progressions?

Each of the 6 orderings can be combined with different initial states, creating a rich parameter space for musical exploration!
