"""
Pattern Detection - Repeated Subsequences: "A qutrit walk on the triadic tonnetz"

Detects:
1. Oscillation patterns (A↔B back-and-forth)
2. Repeated subsequences (length 4-10)
3. Return times (how quickly walk returns to a chord)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import os
from pathlib import Path

# ============================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "QutritWalkResults"
OUTPUT_PATH = BASE_DIR / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)

# ============================================
# LOAD DATA
# ============================================
def load_all_runs(data_path, pattern='experiment_'):
    """Load all CSV files matching pattern"""
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv') and pattern in f]
    print(f"Found {len(csv_files)} CSV files matching '{pattern}'")
    
    all_sequences = []
    for f in csv_files:
        df = pd.read_csv(os.path.join(data_path, f))
        sequence = df['current_chord'].tolist()
        all_sequences.append(sequence)
    
    return all_sequences

# ============================================
# PATTERN DETECTION FUNCTIONS
# ============================================

def find_oscillations(seq):
    """
    Find A→B→A patterns and count consecutive length
    Returns list of ((chord_a, chord_b), length)
    """
    results = []
    i = 0
    while i < len(seq) - 2:
        if seq[i] == seq[i+2] and seq[i] != seq[i+1]:
            a, b = seq[i], seq[i+1]
            length = 3
            j = i + 3
            while j < len(seq):
                expected = a if (j - i) % 2 == 0 else b
                if seq[j] == expected:
                    length += 1
                    j += 1
                else:
                    break
            results.append(((a, b), length))
            i = j
        else:
            i += 1
    return results


def find_repeated(seq, min_len=4, max_len=10):
    """
    Find repeated subsequences of given length range
    Returns dict of {pattern_tuple: count}
    """
    counts = Counter()
    for length in range(min_len, max_len + 1):
        for i in range(len(seq) - length + 1):
            pattern = tuple(seq[i:i+length])
            counts[pattern] += 1
    return {k: v for k, v in counts.items() if v >= 2}


def get_return_times(seq, chord):
    """
    Get list of return times (steps between consecutive visits) for a chord
    """
    indices = [i for i, c in enumerate(seq) if c == chord]
    if len(indices) < 2:
        return []
    return [indices[i+1] - indices[i] for i in range(len(indices)-1)]


# ============================================
# MAIN ANALYSIS
# ============================================

def analyze_patterns(sequences, output_path, title_suffix=''):
    """Run complete pattern analysis"""
    
    print(f"\n{'='*60}")
    print(f"PATTERN ANALYSIS{title_suffix}")
    print(f"{'='*60}")
    print(f"Sequences: {len(sequences)}")
    print(f"Steps per sequence: {len(sequences[0])}")
    
    # Collect all patterns
    all_osc = []
    all_repeated = Counter()
    return_times_C = []
    return_times_Cm = []
    
    for seq in sequences:
        # Oscillations
        all_osc.extend(find_oscillations(seq))
        
        # Repeated subsequences
        for pattern, count in find_repeated(seq).items():
            all_repeated[pattern] += count
        
        # Return times
        return_times_C.extend(get_return_times(seq, 'C'))
        return_times_Cm.extend(get_return_times(seq, 'Cm'))
    
    # ===== OSCILLATIONS =====
    osc_counts = Counter([o[0] for o in all_osc])
    osc_max_len = defaultdict(int)
    for (pair, length) in all_osc:
        osc_max_len[pair] = max(osc_max_len[pair], length)
    
    print(f'\n--- Oscillations (A↔B) ---')
    print(f'Total oscillation events: {len(all_osc)}')
    print('Top 10:')
    for pair, count in osc_counts.most_common(10):
        max_len = osc_max_len[pair]
        print(f'  {pair[0]} ↔ {pair[1]}: {count} times (max length: {max_len})')
    
    # ===== REPEATED SUBSEQUENCES =====
    print(f'\n--- Repeated Subsequences (length 4-10) ---')
    print('Top 15:')
    for pattern, count in all_repeated.most_common(15):
        print(f'  {" → ".join(pattern)}: {count}')
    
    # ===== RETURN TIMES =====
    print(f'\n--- Return Times ---')
    if return_times_C:
        print(f'C major: mean={np.mean(return_times_C):.2f}, median={np.median(return_times_C):.0f}, min={min(return_times_C)}, max={max(return_times_C)}')
    if return_times_Cm:
        print(f'C minor: mean={np.mean(return_times_Cm):.2f}, median={np.median(return_times_Cm):.0f}, min={min(return_times_Cm)}, max={max(return_times_Cm)}')
    
    # ===== VISUALIZATIONS =====
    
    # 1. Oscillation bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    top_osc = osc_counts.most_common(15)
    labels = [f'{a}↔{b}' for (a, b), _ in top_osc]
    values = [c for _, c in top_osc]
    colors = ['darkred' if osc_max_len[p] >= 6 else 'teal' for p, _ in top_osc]
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    ax.set_xlabel('Frequency', fontsize=14)
    ax.set_title(f'Oscillation Patterns (A↔B){title_suffix}\nRed = max length ≥ 6 steps', fontsize=14)
    ax.tick_params(axis='x', labelsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'oscillations{title_suffix.replace(" ", "_")}.png'), dpi=150)
    plt.savefig(os.path.join(output_path, f'oscillations{title_suffix.replace(" ", "_")}.pdf'))
    plt.close()
    
    # 2. Return time histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if return_times_C:
        axes[0].hist(return_times_C, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(np.mean(return_times_C), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(return_times_C):.1f}')
        axes[0].set_xlabel('Steps', fontsize=13)
        axes[0].set_ylabel('Frequency', fontsize=13)
        axes[0].set_title('Return Time to C major', fontsize=14)
        axes[0].tick_params(labelsize=11)
        axes[0].legend(fontsize=11)
    
    if return_times_Cm:
        axes[1].hist(return_times_Cm, bins=30, color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(np.mean(return_times_Cm), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(return_times_Cm):.1f}')
        axes[1].set_xlabel('Steps', fontsize=13)
        axes[1].set_ylabel('Frequency', fontsize=13)
        axes[1].set_title('Return Time to C minor', fontsize=14)
        axes[1].tick_params(labelsize=11)
        axes[1].legend(fontsize=11)
    
    plt.suptitle(f'Return Time Analysis{title_suffix}', fontweight='bold', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'return_times{title_suffix.replace(" ", "_")}.png'), dpi=150)
    plt.savefig(os.path.join(output_path, f'return_times{title_suffix.replace(" ", "_")}.pdf'))
    plt.close()
    
    # 3. Repeated patterns
    fig, ax = plt.subplots(figsize=(12, 8))
    top_rep = all_repeated.most_common(20)
    labels = [' → '.join(p) for p, _ in top_rep]
    values = [c for _, c in top_rep]
    ax.barh(range(len(labels)), values, color='purple')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=14)
    ax.invert_yaxis()
    ax.set_xlabel('Occurrences', fontsize=18)
    ax.set_title(f'Top 20 Repeated Patterns (length 4-10){title_suffix}', fontsize=20)
    ax.tick_params(axis='x', labelsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'repeated_patterns{title_suffix.replace(" ", "_")}.png'), dpi=150)
    plt.savefig(os.path.join(output_path, f'repeated_patterns{title_suffix.replace(" ", "_")}.pdf'))
    plt.close()
    
    print(f'\nFigures saved to {output_path}')
    
    return {
        'oscillations': osc_counts,
        'osc_max_length': osc_max_len,
        'repeated': all_repeated,
        'return_times_C': return_times_C,
        'return_times_Cm': return_times_Cm
    }


if __name__ == "__main__":
    sequences = load_all_runs(DATA_PATH, pattern='experiment_')
    
    if sequences:
        results = analyze_patterns(sequences, OUTPUT_PATH)
        
        print("\n" + "="*60)
        print("TASK 2 COMPLETE")
        print("="*60)
    else:
        print("No data files found!")
