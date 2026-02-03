"""
Initial Quantum State Effects: "A qutrit walk on the triadic tonnetz"

Compares how different initial spin states affect:
1. Chord distribution (which chords are favored)
2. Transition patterns (which moves are favored)
3. Return times (localization behavior)

Initial states to compare:
- C up (|↑⟩) - L transformation biased
- C down (|↓⟩) - R transformation biased
- C up+down - symmetric L/R
- C up+i*down - complex phase
- C up+right+down - equal superposition (|↑⟩ + |→⟩ + |↓⟩)/√3
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import os
from pathlib import Path

# ============================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "QutritWalkResults"
OUTPUT_PATH = BASE_DIR / "outputs"
OUTPUT_PATH.mkdir(exist_ok=True)


# Define initial state patterns to search for in filenames
# Order matters - check longer patterns first to avoid partial matches
INITIAL_STATES = {
    'C_up_plus_right_plus_down': 'experiment_C_up_plus_right_plus_down_run',
    'C_up_plus_i_down': 'experiment_C_up_plus_i_down_run',
    'C_up_plus_down': 'experiment_C_up_plus_down_run',
    'C_down': 'experiment_C_down_run',
    'C_up': 'experiment_C_up_run'
}

MAJOR_CHORDS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
MINOR_CHORDS = ['Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'Bbm', 'Bm']
ALL_CHORDS = MAJOR_CHORDS + MINOR_CHORDS

# ============================================
# LOAD DATA
# ============================================
def load_sequences(data_path, pattern):
    """Load sequences matching filename pattern exactly"""
    sequences = []
    files = []
    for f in os.listdir(data_path):
        if f.endswith('.csv') and f.startswith(pattern):
            files.append(f)
    for f in files:
        df = pd.read_csv(os.path.join(data_path, f))
        sequences.append(df['current_chord'].tolist())
    return sequences


def load_all_initial_states(data_path):
    """Load data for all initial states"""
    data = {}
    for name, pattern in INITIAL_STATES.items():
        seqs = load_sequences(data_path, pattern)
        if seqs:
            data[name] = seqs
            print(f"  {name}: {len(seqs)} sequences")
    return data


# ============================================
# ANALYSIS FUNCTIONS
# ============================================
def get_distribution(sequences):
    """Get chord frequency distribution (percentages)"""
    all_chords = [c for seq in sequences for c in seq]
    total = len(all_chords)
    counts = Counter(all_chords)
    return {c: counts.get(c, 0) / total * 100 for c in ALL_CHORDS}


def get_top_transitions(sequences, top_k=10):
    """Get top transitions as percentages"""
    bigrams = []
    for seq in sequences:
        for i in range(len(seq)-1):
            bigrams.append((seq[i], seq[i+1]))
    total = len(bigrams)
    counts = Counter(bigrams)
    return {k: v/total*100 for k, v in counts.most_common(top_k)}


def get_return_times(sequences, chord):
    """Get return times for a specific chord"""
    times = []
    for seq in sequences:
        indices = [i for i, c in enumerate(seq) if c == chord]
        if len(indices) >= 2:
            times.extend([indices[i+1] - indices[i] for i in range(len(indices)-1)])
    return times


def get_favored_minor(distribution):
    """Find the most visited minor chord"""
    minor_dist = {c: distribution[c] for c in MINOR_CHORDS}
    return max(minor_dist, key=minor_dist.get)


# ============================================
# COMPARISON AND VISUALIZATION
# ============================================
def compare_initial_states(data, output_path):
    """Run full comparison between initial states"""
    
    print("\n" + "="*60)
    print("INITIAL STATE COMPARISON")
    print("="*60)
    
    # Get distributions for all states
    distributions = {name: get_distribution(seqs) for name, seqs in data.items()}
    
    # Summary table
    print("\n--- Favored Chords by Initial State ---")
    print(f"{'State':<30} {'Top Major':<10} {'Top Minor':<10} {'Minor %':<10}")
    print("-" * 60)
    
    summary = {}
    for name, dist in distributions.items():
        top_major = max(MAJOR_CHORDS, key=lambda c: dist[c])
        top_minor = max(MINOR_CHORDS, key=lambda c: dist[c])
        minor_pct = dist[top_minor]
        print(f"{name:<30} {top_major:<10} {top_minor:<10} {minor_pct:.1f}%")
        summary[name] = {'top_major': top_major, 'top_minor': top_minor, 'minor_pct': minor_pct}
    
    # Top transitions comparison
    print("\n--- Top Transition by Initial State ---")
    for name, seqs in data.items():
        trans = get_top_transitions(seqs, 3)
        top = list(trans.items())[0]
        print(f"  {name}: {top[0][0]} → {top[0][1]} ({top[1]:.1f}%)")
    
    # Return time comparison
    print("\n--- Return Time to C Major ---")
    for name, seqs in data.items():
        rt = get_return_times(seqs, 'C')
        if rt:
            print(f"  {name}: mean={np.mean(rt):.2f}, median={np.median(rt):.0f}")
    
    # ===== VISUALIZATIONS =====
    
    # 1. Multi-state distribution comparison
    if len(distributions) >= 2:
        fig, ax = plt.subplots(figsize=(16, 8))
        
        n_states = len(distributions)
        width = 0.8 / n_states
        x = np.arange(24)
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_states))
        
        for i, (name, dist) in enumerate(distributions.items()):
            offset = (i - n_states/2 + 0.5) * width
            values = [dist[c] for c in ALL_CHORDS]
            ax.bar(x + offset, values, width, label=name, color=colors[i])
        
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_CHORDS, rotation=45, ha='right', fontsize=16)
        ax.set_ylabel('Frequency (%)', fontsize=18)
        ax.set_xlabel('Chord', fontsize=18)
        ax.set_title('Chord Distribution by Initial Quantum State', fontsize=20)
        ax.legend(loc='upper right', fontsize=15)
        ax.tick_params(axis='y', labelsize=15)
        ax.axvline(x=11.5, color='black', linestyle='--', alpha=0.3)  # Major/Minor separator
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, 'distribution_all_states.png'), dpi=150)
        plt.savefig(os.path.join(output_path, 'distribution_all_states.pdf'))
        plt.close()
    
    # 2. Favored minor chord comparison
    fig, ax = plt.subplots(figsize=(12, 7))
    
    minor_data = {}
    for name, dist in distributions.items():
        minor_data[name] = {c: dist[c] for c in MINOR_CHORDS}
    
    n_states = len(minor_data)
    width = 0.8 / n_states
    x = np.arange(12)
    colors = plt.cm.tab10(np.linspace(0, 1, n_states))
    
    for i, (name, md) in enumerate(minor_data.items()):
        offset = (i - n_states/2 + 0.5) * width
        values = [md[c] for c in MINOR_CHORDS]
        ax.bar(x + offset, values, width, label=name, color=colors[i])
    
    ax.set_xticks(x)
    ax.set_xticklabels(MINOR_CHORDS, rotation=45, ha='right', fontsize=16)
    ax.set_ylabel('Frequency (%)', fontsize=18)
    ax.set_title('Minor Chord Distribution by Initial State', fontsize=20)
    ax.tick_params(axis='y', labelsize=16)
    ax.legend(fontsize=15)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'minor_comparison.png'), dpi=150)
    plt.savefig(os.path.join(output_path, 'minor_comparison.pdf'))
    plt.close()
    
    # 3. Return time boxplot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    rt_data = []
    rt_labels = []
    for name, seqs in data.items():
        rt = get_return_times(seqs, 'C')
        if rt:
            rt_data.append(rt)
            rt_labels.append(name.replace('_', '\n'))
    
    if rt_data:
        bp = ax.boxplot(rt_data, tick_labels=rt_labels, patch_artist=True)
        colors_box = plt.cm.tab10(np.linspace(0, 1, len(rt_data)))
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
        ax.set_ylabel('Return Time (steps)', fontsize=18)
        ax.set_title('Return Time to C Major by Initial State', fontsize=20)
        ax.tick_params(axis='both', labelsize=15)
        ax.set_ylim(0, 40)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'return_times_all.png'), dpi=150)
    plt.savefig(os.path.join(output_path, 'return_times_all.pdf'))
    plt.close()
    
    print(f"\nFigures saved to {output_path}")
    
    return distributions, summary


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    print("Loading data from all initial states...")
    data = load_all_initial_states(DATA_PATH)
    
    if len(data) >= 2:
        distributions, summary = compare_initial_states(data, OUTPUT_PATH)
        
        print("\n" + "="*60)
        print("KEY FINDING")
        print("="*60)
        print("The initial spin state determines which minor chord is favored:")
        for name, s in summary.items():
            print(f"  {name}: favors {s['top_minor']} ({s['minor_pct']:.1f}%)")
        
        print("\n" + "="*60)
        print("TASK 3 COMPLETE")
        print("="*60)
    else:
        print(f"\nOnly found {len(data)} initial states. Need at least 2 for comparison.")
        print("Available patterns to search for:")
        for name, pattern in INITIAL_STATES.items():
            print(f"  {name}: files matching '{pattern}*.csv'")
