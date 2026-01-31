"""
Visualize qutrit walk trajectories on the Tonnetz

This script reads CSV files from quantum walk experiments and visualizes
the chord sequences as trajectories on a hexagonal Tonnetz layout.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
from pathlib import Path


# Chord names (matching experiments_qutrit.py)
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Enharmonic equivalents (flats to sharps)
ENHARMONIC_MAP = {
    'Db': 'C#',
    'Eb': 'D#',
    'Gb': 'F#',
    'Ab': 'G#',
    'Bb': 'A#'
}

ALL_CHORD_NAMES = []
for root_name in NOTE_NAMES:
    ALL_CHORD_NAMES.append(root_name)  # Major chords
for root_name in NOTE_NAMES:
    ALL_CHORD_NAMES.append(root_name + 'm')  # Minor chords


def parse_chord_name(chord_str):
    """Parse chord string into (root_idx, is_major)"""
    if chord_str.endswith('m'):
        root_name = chord_str[:-1]
        is_major = False
    else:
        root_name = chord_str
        is_major = True

    # Handle enharmonic equivalents
    if root_name in ENHARMONIC_MAP:
        root_name = ENHARMONIC_MAP[root_name]

    root_idx = NOTE_NAMES.index(root_name)
    return root_idx, is_major


def find_note_in_grid(note_name, center_col=5, center_row=5):
    """
    Find the (col, row) position of a note in the Tonnetz grid
    Returns the occurrence closest to the center.

    Args:
        note_name: Note name (e.g., 'C', 'C#', etc.)
        center_col: Column index of center (C is at center)
        center_row: Row index of center

    Returns:
        (col, row) tuple, or None if not found in reasonable range
    """
    # Find all occurrences of this note
    matches = []
    for col in range(center_col - 6, center_col + 7):
        for row in range(center_row - 6, center_row + 7):
            # Calculate note at this position
            col_offset = col - center_col
            row_offset = row - center_row
            half_y = 2 * row + (1 if col % 2 == 1 else 0)
            half_center_y = 2 * center_row + (1 if center_col % 2 == 1 else 0)
            half_y_offset = half_y - half_center_y
            half_fifths = half_y_offset - col_offset
            fifths = half_fifths // 2
            major_thirds = col_offset
            semitones = fifths * 7 + major_thirds * 4
            note = NOTE_NAMES[semitones % 12]

            if note == note_name:
                distance = (col - center_col)**2 + (row - center_row)**2
                matches.append((distance, col, row))

    if not matches:
        return None

    # Return the match closest to center
    matches.sort()
    _, col, row = matches[0]
    return (col, row)


def get_chord_notes(root, is_major):
    """Get the three notes that make up a triad"""
    if is_major:
        return [root, (root + 4) % 12, (root + 7) % 12]
    else:
        return [root, (root + 3) % 12, (root + 7) % 12]


def find_all_chord_triangles(root, is_major, center_col=5, center_row=5, search_radius=6):
    """
    Find all occurrences of a chord as actual triangles on the Tonnetz grid.

    Args:
        root: Root note index (0-11)
        is_major: True for major, False for minor
        center_col: Center column
        center_row: Center row
        search_radius: How far from center to search

    Returns:
        List of (centroid_x, centroid_y, vertices) tuples, sorted by distance from center
    """
    h = np.sqrt(3) / 2
    chord_notes_idx = get_chord_notes(root, is_major)
    chord_notes_names = [NOTE_NAMES[idx] for idx in chord_notes_idx]

    # Center position in pixel coords
    center_x = center_col * h
    center_y = center_row + (0.5 if center_col % 2 == 1 else 0)

    triangles = []

    # Search for all valid triangles
    for col in range(center_col - search_radius, center_col + search_radius + 1):
        for row in range(center_row - search_radius, center_row + search_radius + 1):
            # Check if this (col, row) position could be one vertex of our chord
            # Calculate what note is at this position
            col_offset = col - 5  # Use fixed reference
            half_y = 2 * row + (1 if col % 2 == 1 else 0)
            half_center_y = 2 * 5 + (1 if 5 % 2 == 1 else 0)  # Use fixed center
            half_y_offset = half_y - half_center_y
            half_fifths = half_y_offset - col_offset
            fifths = half_fifths // 2
            major_thirds = col_offset
            semitones = fifths * 7 + major_thirds * 4
            note = NOTE_NAMES[semitones % 12]

            # If this note matches one of our chord notes, check for adjacent triangles
            if note in chord_notes_names:
                # Check six possible triangles around this node
                # (each node participates in up to 6 triangles)
                neighbors = get_triangle_neighbors(col, row)

                for n1, n2 in neighbors:
                    vertices = [(col, row), n1, n2]
                    # Check if these three positions form our desired chord
                    vertex_notes = []
                    valid = True
                    for vc, vr in vertices:
                        vco = vc - 5
                        vhy = 2 * vr + (1 if vc % 2 == 1 else 0)
                        vhyo = vhy - 10
                        vhf = vhyo - vco
                        vf = vhf // 2
                        vmt = vco
                        vsemi = vf * 7 + vmt * 4
                        vn = NOTE_NAMES[vsemi % 12]
                        vertex_notes.append(vn)

                    # Check if this set of notes matches our chord
                    if sorted(vertex_notes) == sorted(chord_notes_names):
                        # Calculate centroid
                        positions = []
                        for vc, vr in vertices:
                            vx = vc * h
                            vy = vr + (0.5 if vc % 2 == 1 else 0)
                            positions.append([vx, vy])
                        centroid = np.mean(positions, axis=0)
                        distance = (centroid[0] - center_x)**2 + (centroid[1] - center_y)**2
                        triangles.append((distance, centroid[0], centroid[1], vertices))

    # Sort by distance and return
    triangles.sort()
    return [(cx, cy, v) for (d, cx, cy, v) in triangles]


def get_triangle_neighbors(col, row):
    """
    Get pairs of neighbors that might form triangles with this node.
    Each pair represents potential edges from this node.
    """
    neighbors = []

    # Vertical neighbor (same column)
    # Up and diagonal neighbors
    if col % 2 == 0:
        # Even column
        neighbors.append(((col + 1, row), (col + 1, row - 1)))  # right triangle pointing up
        neighbors.append(((col + 1, row), (col, row + 1)))  # up-right triangle
        neighbors.append(((col, row + 1), (col - 1, row)))  # up-left triangle
        neighbors.append(((col - 1, row), (col - 1, row - 1)))  # left triangle pointing up
        neighbors.append(((col - 1, row - 1), (col, row - 1)))  # down-left triangle
        neighbors.append(((col, row - 1), (col + 1, row - 1)))  # down-right triangle
    else:
        # Odd column
        neighbors.append(((col + 1, row + 1), (col + 1, row)))  # right triangle pointing down
        neighbors.append(((col + 1, row + 1), (col, row + 1)))  # up-right triangle
        neighbors.append(((col, row + 1), (col - 1, row + 1)))  # up-left triangle
        neighbors.append(((col - 1, row + 1), (col - 1, row)))  # left triangle pointing down
        neighbors.append(((col - 1, row), (col, row - 1)))  # down-left triangle
        neighbors.append(((col, row - 1), (col + 1, row)))  # down-right triangle

    return neighbors


def get_chord_triangle_vertices(root, is_major, center_col=5, center_row=5):
    """
    Get the three (col, row) vertices of a chord triangle on the Tonnetz grid
    Uses the closest occurrence of the chord as an actual triangle.

    Args:
        root: Root note index (0-11)
        is_major: True for major, False for minor
        center_col: Center column
        center_row: Center row

    Returns:
        List of three (col, row) tuples
    """
    # Find all valid triangles for this chord
    triangles = find_all_chord_triangles(root, is_major, center_col, center_row)

    if not triangles:
        # Fallback: use old method
        chord_notes = get_chord_notes(root, is_major)
        vertices = []
        for note_idx in chord_notes:
            note_name = NOTE_NAMES[note_idx]
            grid_pos = find_note_in_grid(note_name, center_col, center_row)
            if grid_pos is None:
                vertices.append((center_col, center_row))
            else:
                vertices.append(grid_pos)
        return vertices

    # Return vertices of closest triangle
    _, _, vertices = triangles[0]
    return vertices


def get_tonnetz_coordinates(root, is_major, center_col=5, center_row=5):
    """
    Get (x, y) pixel coordinates for a chord centroid on the Tonnetz

    Args:
        root: Root note index (0-11)
        is_major: True for major, False for minor
        center_col: Center column reference
        center_row: Center row reference

    Returns:
        (x, y) tuple in pixel coordinates
    """
    h = np.sqrt(3) / 2

    # Get triangle vertices
    vertices = get_chord_triangle_vertices(root, is_major, center_col, center_row)

    # Convert to pixel positions
    positions = []
    for col, row in vertices:
        x = col * h
        y = row + (0.5 if col % 2 == 1 else 0)
        positions.append([x, y])

    # Return the centroid
    centroid = np.mean(positions, axis=0)
    return centroid[0], centroid[1]


def interpolate_offset(step, total_steps, max_offset=0.15):
    """
    Linearly interpolate offset from (+max_offset, +max_offset) to (-max_offset, -max_offset)

    Args:
        step: Current step number (0 to total_steps-1)
        total_steps: Total number of steps
        max_offset: Maximum offset magnitude (as fraction of cell size)

    Returns:
        (offset_x, offset_y) tuple
    """
    # Linear interpolation: t goes from 0 to 1
    t = step / (total_steps - 1) if total_steps > 1 else 0

    # Offset goes from +max_offset to -max_offset
    offset = max_offset * (1 - 2 * t)

    return offset, offset


def get_arrow_color(step, total_steps):
    """
    Get RGB color interpolated from green → orange → red

    Args:
        step: Current step number
        total_steps: Total number of steps

    Returns:
        RGB tuple (r, g, b)
    """
    t = step / (total_steps - 1) if total_steps > 1 else 0

    if t < 0.5:
        # Green (0, 1, 0) → Orange (1, 0.65, 0)
        # Interpolate in first half
        t_local = t * 2  # Map [0, 0.5] → [0, 1]
        r = t_local
        g = 1.0 - 0.35 * t_local
        b = 0
    else:
        # Orange (1, 0.65, 0) → Red (1, 0, 0)
        # Interpolate in second half
        t_local = (t - 0.5) * 2  # Map [0.5, 1] → [0, 1]
        r = 1.0
        g = 0.65 * (1 - t_local)
        b = 0

    return (r, g, b)


def plot_tonnetz_trajectory(csv_path, output_path=None, title=None, figsize=(16, 12)):
    """
    Plot a single walk trajectory on the Tonnetz

    Args:
        csv_path: Path to CSV file containing walk data
        output_path: Optional path to save figure
        title: Optional title for the plot
        figsize: Figure size in inches
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Parse chord sequence
    chords = []
    for chord_str in df['current_chord']:
        root, is_major = parse_chord_name(chord_str)
        chords.append((root, is_major))

    total_steps = len(chords)
    h = np.sqrt(3) / 2

    # First pass: compute all chord positions with large search area
    # to find the bounding box of the trajectory
    initial_center_col = 10
    initial_center_row = 10

    chord_positions = []
    chord_grid_positions = []  # Store (col, row) for grid generation

    # Start at center
    root_first, is_major_first = chords[0]
    triangles_first = find_all_chord_triangles(root_first, is_major_first, initial_center_col, initial_center_row, search_radius=15)
    if triangles_first:
        x_first, y_first, vertices_first = triangles_first[0]
        chord_positions.append((x_first, y_first))
        chord_grid_positions.append(vertices_first)
    else:
        chord_positions.append((initial_center_col * h, initial_center_row))
        chord_grid_positions.append([(initial_center_col, initial_center_row)] * 3)

    # Track through the walk
    for i in range(1, total_steps):
        root, is_major = chords[i]
        prev_x, prev_y = chord_positions[-1]

        # Convert to grid coordinates for reference
        ref_col = int(round(prev_x / h))
        ref_row = int(round(prev_y - (0.5 if ref_col % 2 == 1 else 0)))

        # Find nearest triangle with large search radius
        triangles = find_all_chord_triangles(root, is_major, ref_col, ref_row, search_radius=15)
        if triangles:
            x, y, vertices = triangles[0]
            chord_positions.append((x, y))
            chord_grid_positions.append(vertices)
        else:
            # Fallback
            chord_positions.append((ref_col * h, ref_row))
            chord_grid_positions.append([(ref_col, ref_row)] * 3)

    # Find bounding box of all vertices
    all_cols = []
    all_rows = []
    for vertices in chord_grid_positions:
        for col, row in vertices:
            all_cols.append(col)
            all_rows.append(row)

    min_col = min(all_cols)
    max_col = max(all_cols)
    min_row = min(all_rows)
    max_row = max(all_rows)

    # Add margin
    margin = 3
    grid_min_col = min_col - margin
    grid_max_col = max_col + margin
    grid_min_row = min_row - margin
    grid_max_row = max_row + margin

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Store node positions
    node_positions = {}
    node_notes = {}

    # Use fixed reference point for note calculation (same as before)
    ref_center_col = 5
    ref_center_row = 5

    for col in range(grid_min_col, grid_max_col + 1):
        for row in range(grid_min_row, grid_max_row + 1):
            x = col * h
            y = row + (0.5 if col % 2 == 1 else 0)
            node_positions[(col, row)] = (x, y)

            # Calculate note for this position using fixed reference
            col_offset = col - ref_center_col
            half_y = 2 * row + (1 if col % 2 == 1 else 0)
            half_center_y = 2 * ref_center_row + (1 if ref_center_col % 2 == 1 else 0)
            half_y_offset = half_y - half_center_y
            half_fifths = half_y_offset - col_offset
            fifths = half_fifths // 2
            major_thirds = col_offset
            semitones = fifths * 7 + major_thirds * 4
            note = NOTE_NAMES[semitones % 12]
            node_notes[(col, row)] = note

    # Draw edges
    for col in range(grid_min_col, grid_max_col + 1):
        for row in range(grid_min_row, grid_max_row + 1):
            if (col, row) not in node_positions:
                continue
            x, y = node_positions[(col, row)]

            # Vertical lines (fifths)
            if (col, row + 1) in node_positions:
                x2, y2 = node_positions[(col, row + 1)]
                ax.plot([x, x2], [y, y2], color='lightgray', linewidth=0.5, zorder=1, alpha=0.3)

            # Diagonal lines
            next_col = col + 1
            if col % 2 == 0:
                if (next_col, row) in node_positions:
                    x2, y2 = node_positions[(next_col, row)]
                    ax.plot([x, x2], [y, y2], color='lightgray', linewidth=0.5, zorder=1, alpha=0.3)
                if (next_col, row - 1) in node_positions:
                    x2, y2 = node_positions[(next_col, row - 1)]
                    ax.plot([x, x2], [y, y2], color='lightgray', linewidth=0.5, zorder=1, alpha=0.3)
            else:
                if (next_col, row + 1) in node_positions:
                    x2, y2 = node_positions[(next_col, row + 1)]
                    ax.plot([x, x2], [y, y2], color='lightgray', linewidth=0.5, zorder=1, alpha=0.3)
                if (next_col, row) in node_positions:
                    x2, y2 = node_positions[(next_col, row)]
                    ax.plot([x, x2], [y, y2], color='lightgray', linewidth=0.5, zorder=1, alpha=0.3)

    # Draw note nodes with labels
    for col in range(grid_min_col, grid_max_col + 1):
        for row in range(grid_min_row, grid_max_row + 1):
            if (col, row) not in node_positions:
                continue
            x, y = node_positions[(col, row)]
            note = node_notes[(col, row)]

            # Draw circle
            circle = plt.Circle((x, y), 0.22, fill=True, facecolor='white',
                               edgecolor='gray', linewidth=0.8, zorder=10)
            ax.add_patch(circle)

            # Draw note name
            ax.text(x, y, note, ha='center', va='center', fontsize=8,
                   fontweight='bold', zorder=11)

    # Draw trajectory arrows (using pre-computed positions)
    for i in range(total_steps - 1):
        x_curr, y_curr = chord_positions[i]
        x_next, y_next = chord_positions[i + 1]

        # Apply time-varying offset
        offset_x, offset_y = interpolate_offset(i, total_steps, max_offset=0.05)

        x_start = x_curr + offset_x
        y_start = y_curr + offset_y
        x_end = x_next + offset_x
        y_end = y_next + offset_y

        # Get color based on time
        color = get_arrow_color(i, total_steps)

        # Draw line with small arrowheads
        arrow = FancyArrowPatch(
            (x_start, y_start), (x_end, y_end),
            arrowstyle='->', mutation_scale=10,
            color=color, linewidth=1.5, alpha=0.9, zorder=12
        )
        ax.add_patch(arrow)

    # Mark start and end positions
    x_start, y_start = chord_positions[0]
    x_end, y_end = chord_positions[-1]

    # Small markers for start (green) and end (red)
    ax.plot(x_start, y_start, 'o', color='green', markersize=10,
           markeredgecolor='darkgreen', markeredgewidth=2, zorder=20)
    ax.plot(x_end, y_end, 'o', color='red', markersize=10,
           markeredgecolor='darkred', markeredgewidth=2, zorder=20)

    # Set axis properties based on actual trajectory bounds
    ax.set_aspect('equal')

    # Calculate bounds from node positions
    all_x = [pos[0] for pos in node_positions.values()]
    all_y = [pos[1] for pos in node_positions.values()]

    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)

    # Add small margin
    margin = 0.5
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.axis('off')

    # No title or legend - allows flexible cropping in LaTeX
    plt.tight_layout()

    # Save or show
    if output_path:
        # Use PDF format for vector graphics (infinitely zoomable, smaller file size)
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        print(f"Saved plot to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_multiple_walks(csv_paths, output_dir=None, title_prefix=""):
    """
    Plot multiple walks and save visualizations

    Args:
        csv_paths: List of paths to CSV files
        output_dir: Directory to save output plots (default: same dir as CSV with _viz.png suffix)
        title_prefix: Prefix for plot titles
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for i, csv_path in enumerate(csv_paths, 1):
        csv_path = Path(csv_path)

        # Generate title from filename
        # e.g., "exp1_order_LPR_run01.csv" → "LPR (run 1)"
        stem = csv_path.stem
        if stem.startswith('exp1_order_'):
            parts = stem.split('_')
            order = parts[2]
            run = parts[3].replace('run', '')
            title = f"{title_prefix}Transform Order: {order} (run {run})"
        elif stem.startswith('exp2_state_'):
            parts = stem.split('_')
            state = '_'.join(parts[2:-1])  # Handle multi-word states
            run = parts[-1].replace('run', '')
            title = f"{title_prefix}Initial State: {state} (run {run})"
        else:
            title = f"{title_prefix}{stem}"

        # Output filename: replace .csv with _viz.pdf
        if output_dir is not None:
            output_path = output_dir / f"{stem}_viz.pdf"
        else:
            # Save in same directory as CSV
            output_path = csv_path.with_name(f"{stem}_viz.pdf")

        print(f"[{i}/{len(csv_paths)}] Processing {csv_path.name}...", end=' ')
        try:
            plot_tonnetz_trajectory(csv_path, output_path=output_path, title=title)
            print(f"✓")
        except Exception as e:
            print(f"✗ Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize qutrit walk trajectories on Tonnetz"
    )
    parser.add_argument('input', type=str,
                       help='CSV file or directory containing CSV files')
    parser.add_argument('--output', type=str,
                       help='Output file or directory (default: show plot)')
    parser.add_argument('--title', type=str,
                       help='Plot title (for single file)')
    parser.add_argument('--figsize', type=float, nargs=2, default=[16, 12],
                       help='Figure size in inches (width height)')

    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        # Single file mode
        if args.output:
            output_path = args.output
        else:
            # Replace .csv with _viz.pdf
            output_path = input_path.with_name(f"{input_path.stem}_viz.pdf")

        plot_tonnetz_trajectory(
            input_path,
            output_path=output_path,
            title=args.title,
            figsize=tuple(args.figsize)
        )
    elif input_path.is_dir():
        # Directory/batch mode
        csv_files = sorted(input_path.glob('*.csv'))

        if not csv_files:
            print(f"No CSV files found in {input_path}")
            return

        print(f"Found {len(csv_files)} CSV files")

        if args.output:
            # Save to specified output directory
            print(f"Saving plots to {args.output}/")
            plot_multiple_walks(csv_files, output_dir=args.output)
        else:
            # Save in same directory as CSVs with _viz.pdf suffix
            print(f"Saving plots alongside CSV files with _viz.pdf suffix")
            plot_multiple_walks(csv_files, output_dir=None)
    else:
        print(f"Error: {input_path} not found")


if __name__ == "__main__":
    main()
