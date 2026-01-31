# Transform Order Permutation: Summary of Findings

## Your Observation Was Correct! ✓

You noticed from the Tonnetz visualizations that:
- **LPR ≡ LRP** (identical)
- **RPL ≡ RLP** (identical)
- **PLR ≡ PRL** (identical)

This has been empirically confirmed and mathematically explained.

## Key Findings

### 1. Three Equivalence Classes, Not Six

The 6 permutations of LPR fall into **3 distinct equivalence classes**:

| Class | Permutations | First Transformation | Status |
|-------|--------------|---------------------|---------|
| 1 | LPR, LRP | L (Leading-tone) | Distinct |
| 2 | PLR, PRL | P (Parallel) | Distinct |
| 3 | RLP, RPL | R (Relative) | Distinct |

**Within each class**, swapping positions 1↔2 produces identical results.

**Between classes**, the walks diverge immediately (by step 2).

### 2. Why Position 0 Is Special

The initial state is always `|C major, ↑⟩`, which means:
- Only spin_idx=0 has amplitude initially
- After the first Grover coin operation:
  ```
  [1, 0, 0]  →  [-1/3, 2/3, 2/3]
  ```

This creates an asymmetry:
- **Position 0**: Gets negative amplitude (-1/3)
- **Positions 1 & 2**: Get positive amplitude (2/3 each) - **SYMMETRIC!**

The transformation mapped to position 0 determines the initial trajectory.

### 3. Mathematical Explanation

The Grover coin is permutation-symmetric:
```
      |↑⟩  |→⟩  |↓⟩
|↑⟩ [ -1   2    2  ]
|→⟩ [  2  -1    2  ] × (1/3)
|↓⟩ [  2   2   -1  ]
```

But the **initial state breaks this symmetry** by localizing in spin_idx=0.

When we swap positions 1↔2 (e.g., LPR → LRP):
- Position 0 transformation stays the same (L)
- Positions 1 and 2 swap (P ↔ R)
- **But both have equal amplitude (2/3)**, so swapping them doesn't change the total amplitude distribution
- Result: Identical walks

When we change position 0 (e.g., LPR → PLR):
- Position 0 transformation changes (L → P)
- This position has **special negative amplitude** (-1/3)
- Result: Different walks

## Practical Implications

### For Your Parameter Sweep

You can **reduce computational cost by 50%**:

**Before:** 6 permutations × 4 runs = 24 walks
**After:** 3 permutations × 4 runs = 12 walks

**Recommended canonical representatives:**
- **LPR** (L at position 0)
- **PLR** (P at position 0)
- **RLP** (R at position 0)

### Visualization Recommendations

Since you have visualizations for all 6 permutations:
- Compare **LPR vs PLR vs RLP** to see the 3 distinct behaviors
- Archive or delete the redundant pairs (LRP, PRL, RPL)

## Verification Scripts

Three scripts are available for verification:

1. **[compare_permutations.py](compare_permutations.py)**: Tests the 3 pairs
   ```bash
   python compare_permutations.py
   ```

2. **[verify_all_permutations.py](verify_all_permutations.py)**: Tests all 6
   ```bash
   python verify_all_permutations.py
   ```

3. **[demonstrate_first_step.py](demonstrate_first_step.py)**: Shows amplitude distribution
   ```bash
   python demonstrate_first_step.py
   ```

## What If You Want Different Behavior?

### To Make All 6 Permutations Identical

Use a symmetric initial state:
```python
# Equal superposition of all three spins
walk.set_superposition_state([
    (Chord(0, True, 0), 0, 1/np.sqrt(3)),  # |C, ↑⟩
    (Chord(0, True, 0), 1, 1/np.sqrt(3)),  # |C, →⟩
    (Chord(0, True, 0), 2, 1/np.sqrt(3))   # |C, ↓⟩
])
```

This removes the initial asymmetry → all permutations would be identical.

### To Make All 6 Permutations Different

Use distinct amplitudes for each spin:
```python
# Different amplitude for each spin state
walk.set_superposition_state([
    (Chord(0, True, 0), 0, 0.5),           # |C, ↑⟩
    (Chord(0, True, 0), 1, 0.7),           # |C, →⟩
    (Chord(0, True, 0), 2, 0.866)          # |C, ↓⟩
])
```

This would break all symmetries → each permutation would produce different walks.

## Theoretical Insight

This finding reveals that:

1. **The Grover coin preserves initial asymmetries**: Even though the coin itself is symmetric, the initial localization in spin_idx=0 creates a lasting "preference"

2. **Partial symmetry is a natural consequence**: Given a localized initial state, we expect some permutations to be equivalent (those that preserve the localized index) and others to differ

3. **The transform_order parameter has structure**: It's not just 6 arbitrary configurations - there's a mathematical reason why they cluster into 3 classes of 2

## Conclusion

Your intuition from examining the visualizations was spot-on: **LPR≡LRP, RPL≡RLP, and PLR≡PRL**.

The reason is mathematically elegant: the initial state localizes in spin_idx=0, the Grover coin creates a symmetric distribution in positions 1 and 2 (both get amplitude 2/3), and swapping symmetric positions doesn't change the outcome.

This reduces your parameter space from 6 to 3 distinct configurations, saving computation time and clarifying the structure of your quantum walk experiments.
