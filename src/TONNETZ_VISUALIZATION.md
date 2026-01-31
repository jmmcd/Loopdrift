# Tonnetz Visualization Guide

## Overview

The `visualize_tonnetz.py` script visualizes quantum walk trajectories on a musical Tonnetz (tone network). It automatically generates a triangular lattice where:
- **Nodes** represent musical notes
- **Triangles** represent chords (triads)
- **Arrows** show the quantum walk trajectory over time with color-coded progression (green → orange → red)

**Output format**: Vector PDF files (infinitely zoomable, smaller file size than raster formats)

## Usage

### Single File Processing

Process one CSV file and generate a visualization:

```bash
python visualize_tonnetz.py input.csv
```

This creates `input_viz.pdf` in the same directory.

**With custom output path:**
```bash
python visualize_tonnetz.py input.csv --output my_viz.pdf
```

**With custom title:**
```bash
python visualize_tonnetz.py input.csv --title "My Quantum Walk"
```

### Batch Processing

Process all CSV files in a directory:

```bash
python visualize_tonnetz.py /path/to/csv/directory/
```

This creates `filename_viz.pdf` for each `filename.csv` in the same directory.

**Save to separate output directory:**
```bash
python visualize_tonnetz.py /path/to/csv/directory/ --output /path/to/output/
```

### Examples

```bash
# Process single file
python visualize_tonnetz.py test_walk.csv

# Process all CSVs in a directory
python visualize_tonnetz.py ../experiments/sweep_results/

# Process with custom output directory
python visualize_tonnetz.py ../experiments/sweep_results/ --output ../plots/

# Custom figure size (width, height in inches)
python visualize_tonnetz.py test_walk.csv --figsize 20 15
```

## Input Format

The script expects CSV files with a `current_chord` column containing chord names in the format:
- Major chords: `C`, `C#`, `D`, etc.
- Minor chords: `Cm`, `C#m`, `Dm`, etc.

Example CSV structure:
```csv
step,current_chord,neighbor_L,neighbor_P,neighbor_R,...
0,C,Em,Cm,Am,...
1,C,Em,Cm,Am,...
2,Cm,D#,C,G#m,...
```

## Output

The visualization includes:
- **Triangle grid** showing note positions
- **Color-coded trajectory**: green (start) → orange (middle) → red (end)
- **Small arrowheads** indicating direction of movement
- **Start/End markers**: green and red circles
- **Dynamic grid sizing**: automatically expands to contain the full trajectory

## Features

- **Continuous path tracking**: Ensures all transitions follow adjacent chords
- **Automatic grid sizing**: Expands to fit the trajectory with margins
- **High-resolution output**: 300 DPI for publication-quality figures
- **Batch processing**: Process entire directories efficiently
- **Error handling**: Continues processing if individual files fail

## Python API

You can also use the functions directly in Python:

```python
from visualize_tonnetz import plot_tonnetz_trajectory, plot_multiple_walks
from pathlib import Path

# Single file
plot_tonnetz_trajectory('walk.csv', output_path='walk_viz.pdf', title='My Walk')

# Multiple files
csv_files = list(Path('data/').glob('*.csv'))
plot_multiple_walks(csv_files)  # Saves alongside CSVs as .pdf files
# OR
plot_multiple_walks(csv_files, output_dir='plots/')  # Saves to specific directory
```

## Troubleshooting

**No output generated:**
- Check that the CSV file has a `current_chord` column
- Verify chord names are valid (C, C#, D, etc. with optional 'm' suffix)

**Path goes off-grid:**
- This should not happen with the current implementation
- The script uses dynamic grid sizing with search radius of 15
- If it occurs, it may indicate invalid chord transitions

**Slow processing:**
- Large trajectory files (>500 steps) may take longer
- Consider reducing figure size with `--figsize 12 9`
- Batch processing shows progress with counters

## Technical Details

The visualization uses a triangular lattice based on music theory:
- **Vertical axis**: Perfect fifths (7 semitones)
- **Horizontal axis**: Major thirds (4 semitones)
- Each triangle's three vertices form a musical triad

The algorithm:
1. Parses the chord sequence from CSV
2. Computes all chord positions (centroids of triangles)
3. Determines grid bounds dynamically
4. Renders the Tonnetz with trajectory overlay
5. Saves high-resolution PNG output
