# Transform Order Permutation Analysis

## Empirical Results

Testing with seed=42, steps=200:

- **LPR ≡ LRP**: ✓ IDENTICAL
- **RPL ≡ RLP**: ✓ IDENTICAL
- **PLR ≡ PRL**: ✓ IDENTICAL

## Why These Permutations Are Identical

### The Key Insight

The reason certain permutations produce identical walks lies in the **symmetry of the Grover coin operator**.

The Grover coin is:
```
      |↑⟩  |→⟩  |↓⟩
|↑⟩ [ -1   2    2  ]
|→⟩ [  2  -1    2  ] × (1/3)
|↓⟩ [  2   2   -1  ]
```

This matrix is **permutation-symmetric**: all diagonal elements are -1, all off-diagonal elements are 2.

### How the Walk Step Works

For each chord at each time step:

1. **Apply Grover coin** to the 3-component spin state:
   ```python
   new_spin_state = grover_coin @ spin_state
   ```

2. **Apply transformations** based on the mapping:
   ```python
   for spin_idx in range(3):
       transform_letter = transform_map[spin_idx]  # L, P, or R
       transform_func = transform_funcs[transform_letter]
       new_chord = transform_func(chord)
       new_idx = get_index(new_chord, spin_idx)  # Same spin_idx!
       new_state[new_idx] += new_spin_state[spin_idx]
   ```

### Critical Observation: Spin Index Preservation

The crucial line is:
```python
new_idx = self._get_index(new_chord, spin_idx)
```

The amplitude `new_spin_state[spin_idx]` always goes to the same `spin_idx` in the target chord, regardless of which transformation was applied.

### Mathematical Explanation

Consider two permutations that swap positions 1 and 2 (e.g., LPR vs LRP):

**LPR mapping:**
- spin_idx=0 (|↑⟩) → L transformation → amplitude goes to spin_idx=0 at L(chord)
- spin_idx=1 (|→⟩) → P transformation → amplitude goes to spin_idx=1 at P(chord)
- spin_idx=2 (|↓⟩) → R transformation → amplitude goes to spin_idx=2 at R(chord)

**LRP mapping:**
- spin_idx=0 (|↑⟩) → L transformation → amplitude goes to spin_idx=0 at L(chord)
- spin_idx=1 (|→⟩) → R transformation → amplitude goes to spin_idx=1 at R(chord)
- spin_idx=2 (|↓⟩) → P transformation → amplitude goes to spin_idx=2 at P(chord)

The Grover coin mixes the spin states symmetrically. After the coin:
```
new_spin_state[0] = (-1/3)*spin_state[0] + (2/3)*spin_state[1] + (2/3)*spin_state[2]
new_spin_state[1] = (2/3)*spin_state[0] + (-1/3)*spin_state[1] + (2/3)*spin_state[2]
new_spin_state[2] = (2/3)*spin_state[0] + (2/3)*spin_state[1] + (-1/3)*spin_state[2]
```

Due to the symmetry (rows 1 and 2 have the same pattern, just rotated), swapping which transformation corresponds to spin_idx=1 vs spin_idx=2 produces the same total amplitude distribution.

### Why Adjacent Transpositions Are Equivalent

A permutation like (1 2) that swaps positions 1 and 2 leaves the amplitude distribution invariant because:

1. The Grover coin treats all three spin states symmetrically (up to rotation)
2. The amplitude at each target chord is the sum over all transformations
3. Swapping two transformations just reorders the terms in this sum

**Example at one chord:**

For chord C, we have three components of amplitude that get distributed:

**LPR:**
- Component 0 → Em (via L), stays at spin_idx=0
- Component 1 → Cm (via P), stays at spin_idx=1
- Component 2 → Am (via R), stays at spin_idx=2

**LRP:**
- Component 0 → Em (via L), stays at spin_idx=0
- Component 1 → Am (via R), stays at spin_idx=1
- Component 2 → Cm (via P), stays at spin_idx=2

The **total amplitude** at each target chord is the same, just distributed across different spin indices. But since the Grover coin mixes them symmetrically in the next step, the overall dynamics are identical.

## Which Permutations Are Equivalent? (CORRECTED)

**Empirical verification reveals:**

- **Class 1:** {LPR, LRP} - First position is 'L'
- **Class 2:** {PLR, PRL} - First position is 'P'
- **Class 3:** {RLP, RPL} - First position is 'R'

**Pattern:** Permutations are equivalent if and only if:
1. They have the same transformation at position 0 (spin_idx=0, |↑⟩)
2. Positions 1 and 2 can be swapped (spin_idx=1↔2)

So there are **3 distinct classes**, not 1!

## Why Position 0 (|↑⟩) Is Special

The key is the **initial state**: The walk always starts at `|C major, ↑⟩`, i.e., C major chord with spin state |↑⟩ (spin_idx=0).

On the **first step**, only the amplitude at spin_idx=0 is non-zero. After applying the Grover coin:

```python
# Initial: spin_state = [1, 0, 0]
# After Grover coin:
new_spin_state = grover_coin @ [1, 0, 0]
               = [(-1/3)*1 + (2/3)*0 + (2/3)*0,
                  (2/3)*1 + (-1/3)*0 + (2/3)*0,
                  (2/3)*1 + (2/3)*0 + (-1/3)*0]
               = [-1/3, 2/3, 2/3]
```

Then the transformation step:
- spin_idx=0: amplitude -1/3 goes via transform_order[0]
- spin_idx=1: amplitude 2/3 goes via transform_order[1]
- spin_idx=2: amplitude 2/3 goes via transform_order[2]

**Positions 1 and 2 are symmetric** (both get amplitude 2/3), but **position 0 is different** (gets amplitude -1/3).

This breaks the permutation symmetry! The first transformation (at position 0) determines the initial trajectory.

## Implications for Experiments

Since transform_order permutations fall into 3 equivalence classes:

1. **You only need to run 3 permutations**, not all 6:
   - One from each class: LPR (or LRP), PLR (or PRL), RLP (or RPL)

2. **The first transformation matters most** for the initial trajectory because:
   - The initial state is localized in spin_idx=0
   - After the first Grover coin, spin_idx=0 gets amplitude -1/3 (negative!)
   - Spin indices 1 and 2 get amplitude 2/3 (positive and symmetric)

3. **The parameter space is reduced by factor of 2**:
   - 6 permutations → 3 distinct behaviors
   - Save computational resources by running canonical representatives only

## Should Positions 1 and 2 Be Symmetric?

**Yes, they should be symmetric** given:

1. The Grover coin is permutation-symmetric
2. After the first step, the state spreads across all chords and spins
3. For any state where spin indices 1 and 2 have equal amplitude, the Grover coin treats them identically

**Why does this symmetry persist throughout the walk?**

Consider a state vector at any later time. The evolution depends on:
- The Grover coin (symmetric in all three indices)
- The graph structure (determined by L, P, R transformations)
- Which transformation maps to which spin_idx

If we start with |↑⟩, the initial asymmetry (position 0 vs positions 1,2) is preserved because:
- The Grover coin output for input [1, 0, 0] is [-1/3, 2/3, 2/3]
- This creates a lasting "preference" for the transformation at position 0
- Swapping positions 1↔2 permutes two equal amplitudes → no observable difference

## Should They Be Identical?

**Partial symmetry is correct** given the structure:

1. ✓ The Grover coin is completely symmetric
2. ✓ Spin index preservation during transformation
3. ✗ The initial state breaks symmetry by localizing in spin_idx=0

If you wanted **all 6 permutations** to produce identical results, you would need:
- A superposition initial state like `|C, ↑⟩ + |C, →⟩ + |C, ↓⟩` (equal amplitude in all spins)
- OR use a different coin operator that doesn't preserve the initial asymmetry

If you wanted **all 6 to be different**, you would need:
- An initial state with distinct amplitudes in all three spins: α|↑⟩ + β|→⟩ + γ|↓⟩ where α ≠ β ≠ γ

## Verification with Different Initial States

To confirm this theory, try different initial spins:

- Initial spin = 0 (|↑⟩): Position 0 should be special
- Initial spin = 1 (|→⟩): Position 1 should be special → different equivalence classes
- Initial spin = 2 (|↓⟩): Position 2 should be special → different equivalence classes
- Superposition (equal): All 6 should be identical

## Conclusion

The transform_order parameter produces **3 distinct behaviors** (not 1, not 6):

**Equivalence classes:**
- {LPR, LRP}: L at position 0 (|↑⟩)
- {PLR, PRL}: P at position 0 (|↑⟩)
- {RLP, RPL}: R at position 0 (|↑⟩)

**Why:**
1. Initial state is |C major, ↑⟩ (localized in spin_idx=0)
2. Grover coin breaks symmetry: output [-1/3, 2/3, 2/3] for input [1, 0, 0]
3. Position 0 determines the first transformation applied (with negative amplitude!)
4. Positions 1 and 2 remain symmetric throughout (equal positive amplitude)

**Your observation is correct:** The pairs LPR≡LRP, RPL≡RLP, and PLR≡PRL are identical due to the symmetry between spin indices 1 and 2.

## Deep Dive: Why Different Quantum States Produce Identical Walks

### The Quantum States Are Genuinely Different

After the first step, LPR and LRP create **different quantum states**:

**LPR quantum state:**
```
|ψ⟩_LPR = -1/3|Em, ↑⟩ + 2/3|Cm, →⟩ + 2/3|Am, ↓⟩
```

**LRP quantum state:**
```
|ψ⟩_LRP = -1/3|Em, ↑⟩ + 2/3|Am, →⟩ + 2/3|Cm, ↓⟩
```

These states differ in which chords occupy which spin states:
- In LPR: Cm is in spin |→⟩, Am is in spin |↓⟩
- In LRP: Am is in spin |→⟩, Cm is in spin |↓⟩

The spin indices have different semantic meanings - they map to different transformations in subsequent steps!

### Why Measurement Sees Them As Identical

The key is that **measurement marginalizes over spin states**. The probability of measuring a chord is:

```
P(chord) = Σ_spin |⟨chord, spin|ψ⟩|²
```

**LPR probabilities:**
- P(Em) = |−1/3|² + |0|² + |0|² = 1/9 ≈ 0.111
- P(Cm) = |0|² + |2/3|² + |0|² = 4/9 ≈ 0.444
- P(Am) = |0|² + |0|² + |2/3|² = 4/9 ≈ 0.444

**LRP probabilities:**
- P(Em) = |−1/3|² + |0|² + |0|² = 1/9 ≈ 0.111
- P(Cm) = |0|² + |0|² + |2/3|² = 4/9 ≈ 0.444
- P(Am) = |0|² + |2/3|² + |0|² = 4/9 ≈ 0.444

**The probabilities are identical!** Even though Cm and Am occupy different spin states, they both have the same total probability.

### The Double Swap Cancellation

The deeper reason they produce identical walks is a **double swap that cancels out**:

1. **Swap in quantum state**: What's in spin |→⟩ in LPR is in spin |↓⟩ in LRP
2. **Swap in transformation mapping**: What |→⟩ maps to in LPR (P), |↓⟩ maps to in LRP (P)

When these states evolve in the next step:

**LPR evolution from step 1:**
- Cm at spin |→⟩ → spin |→⟩ maps to position 1 → transformation P
- Am at spin |↓⟩ → spin |↓⟩ maps to position 2 → transformation R

**LRP evolution from step 1:**
- Am at spin |→⟩ → spin |→⟩ maps to position 1 → transformation R
- Cm at spin |↓⟩ → spin |↓⟩ maps to position 2 → transformation P

**Same transformations applied to same chords with same amplitudes!**

The swap in which chord is in which spin is exactly canceled by the swap in which transformation each spin maps to.

### Mathematical Formulation

Define a permutation operator σ_{12} that swaps spin indices 1 ↔ 2.

LRP is equivalent to applying σ_{12} to both:
1. The quantum state (swap which chords are in spins 1 vs 2)
2. The transformation map (swap which transformations spins 1 vs 2 apply)

These two swaps are **inverse operations** on the combined system, leaving the dynamics invariant.

### Why This Doesn't Work for Position 0

Position 0 gets amplitude -1/3, while positions 1 and 2 get +2/3 (different magnitudes).

If we swap position 0 with position 1 (e.g., LPR → PLR):
- Different chords get different amplitudes (-1/3 vs +2/3)
- The swap doesn't cancel because the amplitudes aren't equal
- Result: Different walk trajectory

### Summary

The equivalence LPR ≡ LRP arises because:

1. ✓ Positions 1 and 2 get **equal amplitude** (2/3)
2. ✓ The quantum states are **different** (different spin distributions)
3. ✓ Measurement probabilities are **identical** (marginalizing over spins)
4. ✓ Evolution is **identical** (double swap cancellation)

It's a beautiful example of how symmetries in quantum mechanics can preserve observable behavior even when the underlying quantum states differ!
