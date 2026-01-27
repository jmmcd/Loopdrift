"""
Qutrit Quantum Walk on the Triadic Tonnetz

Implements a quantum walk using a 3-state quantum system (qutrit) on the
triadic tonnetz graph. The graph has 24 nodes (12 major + 12 minor chords),
each of degree 3, forming a bipartite structure.

The three qutrit states |↑⟩, |→⟩, |↓⟩ correspond to Neo-Riemannian transformations:
- |↑⟩: Leittonwechsel (L) - changes the root note
- |→⟩: Parallel (P) - changes the middle note
- |↓⟩: Relative (R) - changes the last note

The quantum walk alternates between applying the Grover coin operator (3×3 unitary)
and shifting according to the P/L/R moves on the graph.
"""

import numpy as np
from typing import Tuple, List, Optional
import random
from midi_generator import MIDIGenerator


# Chord representations: (root, is_major, inversion)
# root: 0-11 (C, C#, D, D#, E, F, F#, G, G#, A, Bb, B)
# is_major: True for major, False for minor
# inversion: 0=root position, 1=first inversion, 2=second inversion

class Chord:
    """Represents a triad with root, quality, and inversion"""

    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']

    def __init__(self, root: int, is_major: bool, inversion: int = 0):
        self.root = root % 12
        self.is_major = is_major
        self.inversion = inversion % 3

    def to_midi_notes(self, base_octave: int = 4) -> List[int]:
        """
        Convert chord to MIDI note numbers

        Returns list of 3 MIDI notes all within the same octave (close voicing)
        """
        base = 12 * (base_octave + 1)  # MIDI note for C in the base octave

        if self.is_major:
            # Major triad: root, major third, perfect fifth
            intervals = [0, 4, 7]
        else:
            # Minor triad: root, minor third, perfect fifth
            intervals = [0, 3, 7]

        # Create notes and wrap them all into the same octave
        # Start from the root note in the base octave
        notes = []
        for interval in intervals:
            # Add interval to root, but keep within same octave
            pitch_class = (self.root + interval) % 12
            midi_note = base + pitch_class
            notes.append(midi_note)

        return notes

    def __str__(self) -> str:
        quality = "" if self.is_major else "m"
        inv_str = "" if self.inversion == 0 else f" ({self.inversion})"
        return f"{self.NOTE_NAMES[self.root]}{quality}{inv_str}"

    def __eq__(self, other) -> bool:
        return (self.root == other.root and
                self.is_major == other.is_major and
                self.inversion == other.inversion)

    def __hash__(self) -> int:
        return hash((self.root, self.is_major, self.inversion))


def apply_P(chord: Chord) -> Chord:
    """
    Apply Parallel transformation: changes middle note (major ↔ minor)
    e.g., C major [C,E,G] → C minor [C,Eb,G]
    Inversion stays the same
    """
    return Chord(chord.root, not chord.is_major, chord.inversion)


def apply_L(chord: Chord) -> Chord:
    """
    Apply Leittonwechsel transformation (Leading-tone exchange)

    Exchanges a triad with its leading-tone relative, sharing two common tones.
    - C major (C, E, G) ↔ E minor (E, G, B): share E and G
    - A minor (A, C, E) ↔ C major (C, E, G): share C and E

    Major chord (r, r+4, r+7) → Minor with root r+4
    Minor chord (r, r+3, r+7) → Major with root r+3
    """
    if chord.is_major:
        # Major → minor: new root is major 3rd above
        # e.g., C (root=0) → Em (root=4)
        new_root = (chord.root + 4) % 12
        return Chord(new_root, False, 0)
    else:
        # Minor → major: new root is minor 3rd above
        # e.g., Am (root=9) → C (root=0), since 9+3=12%12=0
        new_root = (chord.root + 3) % 12
        return Chord(new_root, True, 0)


def apply_R(chord: Chord) -> Chord:
    """
    Apply Relative transformation

    Exchanges a triad with its relative, sharing two common tones.
    - C major (C, E, G) ↔ A minor (A, C, E): share C and E
    - E minor (E, G, B) ↔ G major (G, B, D): share G and B

    Major chord (r, r+4, r+7) → Minor with root r-3
    Minor chord (r, r+3, r+7) → Major with root r+3
    """
    if chord.is_major:
        # Major → minor: new root is minor 3rd below
        # e.g., C (root=0) → Am (root=9)
        new_root = (chord.root - 3) % 12
        return Chord(new_root, False, 0)
    else:
        # Minor → major: new root is perfect 4th below (or perfect 5th above)
        # e.g., Am (root=9) → F (root=5)
        # Am (A, C, E) = (9, 0, 4) shares A and C with F major (F, A, C) = (5, 9, 0)
        new_root = (chord.root - 4) % 12
        return Chord(new_root, True, 0)


class QutritWalker:
    """Qutrit quantum walk on the triadic tonnetz"""

    def __init__(self):
        # State space: 24 chords × 3 qutrit states = 72 dimensions
        self.num_chords = 24
        self.num_spins = 3  # |↑⟩, |→⟩, |↓⟩
        self.state_size = self.num_chords * self.num_spins

        # Quantum state vector (72 dimensions)
        self.state = np.zeros(self.state_size, dtype=complex)

        # Grover coin operator (3×3 unitary matrix)
        self.grover_coin = (1/3) * np.array([
            [-1, 2, 2],
            [2, -1, 2],
            [2, 2, -1]
        ], dtype=complex)

        # Chord index mapping: (root, is_major) → index (0-23)
        self.chord_to_index = {}
        self.index_to_chord = {}

        idx = 0
        for root in range(12):
            # Major chords: 0-11
            self.chord_to_index[(root, True)] = idx
            self.index_to_chord[idx] = Chord(root, True, 0)
            idx += 1

        for root in range(12):
            # Minor chords: 12-23
            self.chord_to_index[(root, False)] = idx
            self.index_to_chord[idx] = Chord(root, False, 0)
            idx += 1

    def _get_index(self, chord: Chord, spin: int) -> int:
        """Convert (chord, spin) to linear index"""
        chord_idx = self.chord_to_index[(chord.root, chord.is_major)]
        return chord_idx * 3 + spin

    def _get_state_params(self, index: int) -> Tuple[Chord, int]:
        """Convert linear index back to (chord, spin)"""
        chord_idx = index // 3
        spin = index % 3
        chord = self.index_to_chord[chord_idx]
        return chord, spin

    def set_initial_state(self, chord: Chord, spin: int = 0):
        """
        Set initial state |chord, spin⟩

        Args:
            chord: Initial chord
            spin: 0=|↑⟩, 1=|→⟩, 2=|↓⟩ (default: 0)
        """
        self.state.fill(0)
        index = self._get_index(chord, spin)
        self.state[index] = 1.0

    def step(self):
        """
        Perform one quantum walk step:
        1. Apply Grover coin to spin state
        2. Shift according to P/L/R transformations
        """
        new_state = np.zeros_like(self.state)

        # For each chord, apply coin operation then shift
        for chord_idx in range(self.num_chords):
            chord = self.index_to_chord[chord_idx]

            # Get the 3-component spin state for this chord
            spin_state = np.array([
                self.state[self._get_index(chord, 0)],  # |↑⟩
                self.state[self._get_index(chord, 1)],  # |→⟩
                self.state[self._get_index(chord, 2)]   # |↓⟩
            ])

            # Skip if this chord has no amplitude
            if np.linalg.norm(spin_state) < 1e-12:
                continue

            # Apply Grover coin
            new_spin_state = self.grover_coin @ spin_state

            # Shift according to spin state:
            # |↑⟩ → L transformation (Leittonwechsel)
            # |→⟩ → P transformation (Parallel)
            # |↓⟩ → R transformation (Relative)

            # |↑⟩ component goes to L(chord)
            if abs(new_spin_state[0]) > 1e-12:
                new_chord = apply_L(chord)
                new_idx = self._get_index(new_chord, 0)
                new_state[new_idx] += new_spin_state[0]

            # |→⟩ component goes to P(chord)
            if abs(new_spin_state[1]) > 1e-12:
                new_chord = apply_P(chord)
                new_idx = self._get_index(new_chord, 1)
                new_state[new_idx] += new_spin_state[1]

            # |↓⟩ component goes to R(chord)
            if abs(new_spin_state[2]) > 1e-12:
                new_chord = apply_R(chord)
                new_idx = self._get_index(new_chord, 2)
                new_state[new_idx] += new_spin_state[2]

        self.state = new_state

    def measure_constrained(self, allowed_chords: List[Chord]) -> Chord:
        """
        Measure the quantum state, but only consider allowed chords
        Does NOT collapse the state - state continues evolving

        Args:
            allowed_chords: List of chords to restrict measurement to

        Returns:
            One of the allowed chords, sampled according to probabilities
        """
        # Get probabilities for all chords
        all_probs = self.get_chord_probabilities()

        # Filter to only allowed chords
        allowed_probs = {}
        for chord in allowed_chords:
            key = (chord.root, chord.is_major)
            if key in all_probs:
                allowed_probs[key] = all_probs[key]
            else:
                allowed_probs[key] = 0.0

        # Normalize
        total = sum(allowed_probs.values())
        if total < 1e-12:
            # No amplitude in allowed chords, pick randomly
            return allowed_chords[np.random.randint(len(allowed_chords))]

        allowed_probs = {k: v/total for k, v in allowed_probs.items()}

        # Sample
        chord_keys = list(allowed_probs.keys())
        probs = np.array([allowed_probs[k] for k in chord_keys])

        chosen_idx = np.random.choice(len(chord_keys), p=probs)
        chosen_root, chosen_is_major = chord_keys[chosen_idx]

        return Chord(chosen_root, chosen_is_major, 0)

    def get_chord_probabilities(self) -> dict:
        """Get probability distribution over chords (marginalizing over spin)"""
        probs = {}
        for i in range(self.state_size):
            if abs(self.state[i]) > 1e-12:
                chord, spin = self._get_state_params(i)
                key = (chord.root, chord.is_major)
                if key not in probs:
                    probs[key] = 0
                probs[key] += abs(self.state[i]) ** 2
        return probs


class QutritWalkMusic(MIDIGenerator):
    """Qutrit quantum walk music generator"""

    def __init__(self):
        super().__init__(description="Qutrit Walk - Quantum music generation on triadic tonnetz")
        self.walker = QutritWalker()
        self.chord_duration = 1.0  # seconds
        self.chord_velocity = 70
        self.initial_chord = Chord(0, True, 0)  # C major
        self.initial_spin = 0  # |↑⟩
        self.current_chord: Chord = self.initial_chord  # Track the last played chord
        self.first_step = True

    def setup_args(self, parser=None):
        """Add qutrit walk specific arguments"""
        parser = super().setup_args(parser)
        parser.add_argument('--duration', type=float, default=1.0,
                          help='Chord duration in seconds (default: 1.0)')
        parser.add_argument('--velocity', type=int, default=70,
                          help='MIDI velocity (default: 70)')
        parser.add_argument('--root', type=str, default='C',
                          help='Starting chord root note (default: C)')
        parser.add_argument('--minor', action='store_true',
                          help='Start with minor chord (default: major)')
        return parser

    def setup(self):
        """Initialize qutrit walk state"""
        self.walker.set_initial_state(self.initial_chord, self.initial_spin)
        self.current_chord = self.initial_chord
        self.first_step = True  # Flag for first chord

        print(f"Starting qutrit quantum walk on triadic tonnetz")
        print(f"Chord duration: {self.chord_duration}s")
        print(f"Initial chord: {self.initial_chord}")
        print()

    def generate_step(self):
        """Measure quantum state and play chord"""
        import mido
        import threading
        import time

        # For the very first step, play the initial chord before stepping
        if self.first_step:
            self.first_step = False
            chord_to_play = self.current_chord
            print(f"{str(chord_to_play):5s} | Initial")
        else:
            # Evolve the quantum state (no collapse)
            self.walker.step()

            # Get the 3 neighbors of the current chord
            neighbors = [
                apply_L(self.current_chord),
                apply_P(self.current_chord),
                apply_R(self.current_chord)
            ]

            # Get probability distribution for neighbors only
            chord_probs = self.walker.get_chord_probabilities()
            neighbor_probs = []
            for chord in neighbors:
                key = (chord.root, chord.is_major)
                prob = chord_probs.get(key, 0.0)
                neighbor_probs.append((chord, prob))

            # Sort by probability and format for display
            neighbor_probs.sort(key=lambda x: x[1], reverse=True)
            prob_str = ", ".join([f"{str(chord):5s}: {prob:.3f}" for chord, prob in neighbor_probs])

            # Sample from neighbors only
            chord_to_play = self.walker.measure_constrained(neighbors)

            # Print: current chord | neighbor probabilities → chosen chord
            print(f"{str(self.current_chord):5s} | [{prob_str}] → {str(chord_to_play):5s}")

            # Update current chord for next iteration
            self.current_chord = chord_to_play

        # Get MIDI notes for the chord
        midi_notes = chord_to_play.to_midi_notes(self.base_octave)

        # Play all three notes of the chord simultaneously
        duration_ms = int(self.chord_duration * 1000)
        for note in midi_notes:
            if 0 <= note <= 127:  # Validate MIDI range
                msg_on = mido.Message('note_on', note=note, velocity=self.chord_velocity)
                self.outport.send(msg_on)

        # Schedule note-offs
        def send_note_offs():
            time.sleep(self.chord_duration)
            for note in midi_notes:
                if 0 <= note <= 127:
                    msg_off = mido.Message('note_off', note=note, velocity=0)
                    self.outport.send(msg_off)

        threading.Thread(target=send_note_offs, daemon=True).start()

        # Note: The quantum state continues evolving without collapse!

    def get_step_duration(self):
        """Return chord duration (time between measurements)"""
        return self.chord_duration


if __name__ == "__main__":
    qw = QutritWalkMusic()

    # Parse arguments
    args = qw.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}\n")

    # Parse root note
    try:
        root_idx = Chord.NOTE_NAMES.index(args.root)
    except ValueError:
        print(f"Error: Invalid root note '{args.root}'. Valid notes: {', '.join(Chord.NOTE_NAMES)}")
        exit(1)

    # Set parameters from args
    qw.base_octave = args.base_octave
    qw.chord_duration = args.duration
    qw.chord_velocity = args.velocity
    qw.initial_chord = Chord(root_idx, not args.minor, 0)

    # Select and open MIDI port
    port_name = qw.select_midi_port(args.midi_port)
    qw.initialize_midi(port_name)

    # Initialize and run
    qw.setup()
    qw.run()
