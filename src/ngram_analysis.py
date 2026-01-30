"""
Statistical Analysis - N-gram chord distributions: "A qutrit walk on the triadic tonnetz"

Generates:
- n=1: Histogram of 24 chords
- n=2: 24x24 heatmap of chord transitions
- n=3: Top trigrams list + grouped visualization
- n=4: Top 4-grams list
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

# Chord ordering (matches CSV columns)
MAJOR_CHORDS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
MINOR_CHORDS = ['Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'Bbm', 'Bm']
ALL_CHORDS = MAJOR_CHORDS + MINOR_CHORDS

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
# N-GRAM EXTRACTION
# ============================================
def extract_ngrams(sequences, n):
    """Extract n-grams from all sequences
    NOTE: Skip first chord (step 0) to avoid artifact from initial position
    """
    ngrams = []
    for seq in sequences:
        # Start from index 1 to skip the initial chord at step 0
        seq_trimmed = seq[1:]
        for i in range(len(seq_trimmed) - n + 1):
            ngram = tuple(seq_trimmed[i:i+n])
            ngrams.append(ngram)
    return ngrams

# ============================================
# ANALYSIS FUNCTIONS
# ============================================
def analyze_unigrams(sequences):
    """n=1: Chord frequency distribution
    NOTE: Skip first chord (step 0) to avoid artifact from initial position
    """
    all_chords_flat = [chord for seq in sequences for chord in seq[1:]]  # Skip first chord
    counts = Counter(all_chords_flat)
    
    # Order by ALL_CHORDS for consistent display
    ordered_counts = {chord: counts.get(chord, 0) for chord in ALL_CHORDS}
    
    return ordered_counts

def analyze_bigrams(sequences):
    """n=2: Transition matrix"""
    bigrams = extract_ngrams(sequences, 2)
    counts = Counter(bigrams)
    
    # Create 24x24 matrix
    matrix = np.zeros((24, 24))
    chord_to_idx = {chord: i for i, chord in enumerate(ALL_CHORDS)}
    
    for (c1, c2), count in counts.items():
        if c1 in chord_to_idx and c2 in chord_to_idx:
            i, j = chord_to_idx[c1], chord_to_idx[c2]
            matrix[i, j] = count
    
    return matrix, counts

def analyze_trigrams(sequences):
    """n=3: Trigram frequencies"""
    trigrams = extract_ngrams(sequences, 3)
    counts = Counter(trigrams)
    return counts

def analyze_fourgrams(sequences):
    """n=4: Four-gram frequencies"""
    fourgrams = extract_ngrams(sequences, 4)
    counts = Counter(fourgrams)
    return counts

# ============================================
# VISUALIZATION
# ============================================
def plot_unigrams(counts, output_path, title_suffix=''):
    """Plot n=1 histogram with log scale"""
    from matplotlib.patches import Patch
    
    chords = list(counts.keys())
    values = list(counts.values())
    
    # Color by major/minor
    colors = ['steelblue' if c in MAJOR_CHORDS else 'coral' for c in chords]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.bar(range(len(chords)), values, color=colors)
    ax.set_xticks(range(len(chords)))
    ax.set_xticklabels(chords, rotation=45, ha='right', fontsize=16)
    ax.set_xlabel('Chord', fontsize=20)
    ax.set_ylabel('Frequency (log scale)', fontsize=20)
    ax.set_yscale('log')
    ax.set_title(f'Chord Distribution{title_suffix}', fontsize=22)
    ax.tick_params(axis='y', labelsize=16)
    
    # Add legend
    legend_elements = [Patch(facecolor='steelblue', label='Major'),
                       Patch(facecolor='coral', label='Minor')]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'ngram_n1_histogram{title_suffix.replace(" ", "_")}.png'), dpi=150)
    plt.savefig(os.path.join(output_path, f'ngram_n1_histogram{title_suffix.replace(" ", "_")}.pdf'))
    print(f"Saved n=1 histogram")
    plt.close()

def plot_bigram_heatmap(matrix, output_path, title_suffix=''):
    """Plot n=2 heatmap (24x24)"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Use log scale for better visibility
    matrix_log = np.log10(matrix + 1)  # +1 to avoid log(0)
    
    sns.heatmap(matrix_log, 
                xticklabels=ALL_CHORDS, 
                yticklabels=ALL_CHORDS,
                cmap='YlOrRd',
                ax=ax,
                cbar_kws={'label': 'log10(count + 1)'})
    
    ax.set_xlabel('To Chord', fontsize=20)
    ax.set_ylabel('From Chord', fontsize=20)
    ax.set_title(f'Chord Transitions (n=2 bigrams){title_suffix}', fontsize=22)
    ax.tick_params(axis='both', labelsize=16)
    
    # Make colorbar label bigger
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('log10(count + 1)', fontsize=16)
    
    # Add grid lines to separate major/minor
    ax.axhline(y=12, color='black', linewidth=2)
    ax.axvline(x=12, color='black', linewidth=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'ngram_n2_heatmap{title_suffix.replace(" ", "_")}.png'), dpi=150)
    plt.savefig(os.path.join(output_path, f'ngram_n2_heatmap{title_suffix.replace(" ", "_")}.pdf'))
    print(f"Saved n=2 heatmap")
    plt.close()

def plot_top_ngrams(counts, n, output_path, top_k=20, title_suffix=''):
    """Plot top-k n-grams as horizontal bar chart"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_ngrams = counts.most_common(top_k)
    labels = [' → '.join(ng) for ng, _ in top_ngrams]
    values = [c for _, c in top_ngrams]
    
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, color='teal')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=16)
    ax.invert_yaxis()
    ax.set_xlabel('Count', fontsize=16)
    ax.set_title(f'Top {top_k} {n}-grams{title_suffix}', fontsize=22)
    ax.tick_params(axis='x', labelsize=16)
    
    # Add count labels
    for i, v in enumerate(values):
        ax.text(v + max(values)*0.01, i, f'{v:,}', va='center', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'ngram_n{n}_top{top_k}{title_suffix.replace(" ", "_")}.png'), dpi=150)
    plt.savefig(os.path.join(output_path, f'ngram_n{n}_top{top_k}{title_suffix.replace(" ", "_")}.pdf'))
    print(f"Saved n={n} top {top_k}")
    plt.close()

# ============================================
# MAIN ANALYSIS
# ============================================
def run_analysis(data_path, output_path, pattern='experiment_', title_suffix=''):
    """Run complete n-gram analysis"""
    print(f"\n{'='*60}")
    print(f"N-GRAM ANALYSIS{title_suffix}")
    print(f"{'='*60}")
    
    # Load data
    sequences = load_all_runs(data_path, pattern)
    if not sequences:
        print(f"No files found with pattern '{pattern}'")
        return
    
    total_chords = sum(len(s) for s in sequences)
    print(f"Total sequences: {len(sequences)}")
    print(f"Total chords: {total_chords:,}")
    
    # n=1: Unigrams
    print(f"\n--- n=1 (Unigrams) ---")
    unigram_counts = analyze_unigrams(sequences)
    print("Top 5 chords:")
    for chord, count in sorted(unigram_counts.items(), key=lambda x: -x[1])[:5]:
        pct = 100 * count / total_chords
        print(f"  {chord}: {count:,} ({pct:.1f}%)")
    plot_unigrams(unigram_counts, output_path, title_suffix)
    
    # n=2: Bigrams
    print(f"\n--- n=2 (Bigrams) ---")
    bigram_matrix, bigram_counts = analyze_bigrams(sequences)
    print("Top 5 transitions:")
    for (c1, c2), count in bigram_counts.most_common(5):
        pct = 100 * count / (total_chords - len(sequences))
        print(f"  {c1} → {c2}: {count:,} ({pct:.1f}%)")
    plot_bigram_heatmap(bigram_matrix, output_path, title_suffix)
    plot_top_ngrams(bigram_counts, 2, output_path, top_k=20, title_suffix=title_suffix)
    
    # n=3: Trigrams
    print(f"\n--- n=3 (Trigrams) ---")
    trigram_counts = analyze_trigrams(sequences)
    print("Top 5 trigrams:")
    for ngram, count in trigram_counts.most_common(5):
        print(f"  {' → '.join(ngram)}: {count:,}")
    plot_top_ngrams(trigram_counts, 3, output_path, top_k=20, title_suffix=title_suffix)
    
    # n=4: Four-grams
    print(f"\n--- n=4 (Four-grams) ---")
    fourgram_counts = analyze_fourgrams(sequences)
    print("Top 5 four-grams:")
    for ngram, count in fourgram_counts.most_common(5):
        print(f"  {' → '.join(ngram)}: {count:,}")
    plot_top_ngrams(fourgram_counts, 4, output_path, top_k=20, title_suffix=title_suffix)
    
    # Summary statistics
    print(f"\n--- Summary ---")
    print(f"Unique unigrams: {len([c for c, v in unigram_counts.items() if v > 0])}/24")
    print(f"Unique bigrams: {len(bigram_counts)}")
    print(f"Unique trigrams: {len(trigram_counts)}")
    print(f"Unique four-grams: {len(fourgram_counts)}")
    
    return {
        'unigrams': unigram_counts,
        'bigrams': bigram_counts,
        'trigrams': trigram_counts,
        'fourgrams': fourgram_counts,
        'bigram_matrix': bigram_matrix
    }


if __name__ == "__main__":
    # Run analysis on available data
    # You can modify the pattern to analyze different initial states
    
    # Example: analyze all runs starting with C major, spin up
    # pattern = 'experiment_C_up_run'
    
    # Example: analyze all runs with up+right+down initial state
    # pattern = 'experiment_C_up_plus_right_plus_down'
    
    # Run on all experiment files
    results = run_analysis(DATA_PATH, OUTPUT_PATH, pattern='experiment_')
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
