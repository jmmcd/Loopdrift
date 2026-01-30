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


def parse_initial_state(state_spec: str) -> List[Tuple[Chord, int, complex]]:
    """
    Parse initial state specification string into list of (chord, spin, amplitude) tuples

    Supported formats:
    - "C:up" → |C major, ↑⟩
    - "Am:down" → |A minor, ↓⟩
    - "C:up+down" → (|C,↑⟩ + |C,↓⟩)/√2 (equal superposition)
    - "C:up+right+down" → (|C,↑⟩ + |C,→⟩ + |C,↓⟩)/√3 (all three spins)
    - "C:0.5*up+0.5*down" → Custom real amplitudes
    - "C:1*up+i*down" → Complex amplitudes with imaginary unit i
    - "C+Am+F:up" → (|C,↑⟩ + |Am,↑⟩ + |F,↑⟩)/√3 (chord superposition)

    Args:
        state_spec: State specification string

    Returns:
        List of (chord, spin, amplitude) tuples (unnormalized)
    """
    spin_names = {'up': 0, 'right': 1, 'down': 2}
    components = []

    # Find the colon to split chord(s) and spin specification
    if ':' not in state_spec:
        raise ValueError(f"Invalid format: {state_spec}. Expected 'chord:spin_spec' or 'chord1+chord2:spin_spec'")

    # Split at the LAST colon to handle multiple chords
    last_colon = state_spec.rfind(':')
    chords_str = state_spec[:last_colon].strip()
    spin_spec = state_spec[last_colon+1:].strip()

    # Parse chord(s) - can be single chord or chord+chord+chord
    chord_parts = [c.strip() for c in chords_str.split('+')]

    chords = []
    for chord_str in chord_parts:
        # Parse chord
        is_major = not chord_str.endswith('m')
        root_str = chord_str[:-1] if chord_str.endswith('m') else chord_str

        try:
            root_idx = Chord.NOTE_NAMES.index(root_str)
        except ValueError:
            raise ValueError(f"Invalid chord root: {root_str}")

        chords.append(Chord(root_idx, is_major, 0))

    # Parse spin specification
    if '+' in spin_spec:
        # Superposition of spins
        spin_terms = [s.strip() for s in spin_spec.split('+')]

        for term in spin_terms:
            # Parse coefficient*spin or just spin
            if '*' in term:
                coeff_str, spin_name = term.split('*', 1)
                coeff_str = coeff_str.strip()
                spin_name = spin_name.strip()

                # Parse coefficient (can be complex with 'i')
                if 'i' in coeff_str:
                    # Complex coefficient
                    if coeff_str == 'i':
                        coeff = 1j
                    elif coeff_str == '-i':
                        coeff = -1j
                    else:
                        # Parse forms like "0.5i" or "2.3i"
                        coeff_str = coeff_str.replace('i', 'j')  # Python uses j
                        coeff = complex(coeff_str)
                else:
                    coeff = float(coeff_str)
            else:
                # No coefficient specified, default to 1
                spin_name = term
                coeff = 1.0

            if spin_name not in spin_names:
                raise ValueError(f"Invalid spin name: {spin_name}. Use 'up', 'right', or 'down'")

            spin = spin_names[spin_name]

            # Add this spin state for all chords
            for chord in chords:
                components.append((chord, spin, coeff))
    else:
        # Single spin state
        if spin_spec not in spin_names:
            raise ValueError(f"Invalid spin name: {spin_spec}. Use 'up', 'right', or 'down'")

        spin = spin_names[spin_spec]

        # Add this spin state for all chords
        for chord in chords:
            components.append((chord, spin, 1.0))

    return components


class QutritWalker:
    """Qutrit quantum walk on the triadic tonnetz"""

    def __init__(self, transform_order: str = "LPR"):
        """
        Initialize qutrit walker

        Args:
            transform_order: Order of transformations mapped to spin states.
                           Must be a permutation of "LPR" (default: "LPR")
                           Example: "LPR" means ↑→L, →→P, ↓→R
                                   "RLP" means ↑→R, →→L, ↓→P
        """
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

        # Validate and store transformation order
        self.transform_order = transform_order.upper()
        if sorted(self.transform_order) != ['L', 'P', 'R']:
            raise ValueError(f"Invalid transform_order: {transform_order}. Must be a permutation of 'LPR'")

        # Map spin states to transformations
        self.transform_map = {
            0: self.transform_order[0],  # |↑⟩
            1: self.transform_order[1],  # |→⟩
            2: self.transform_order[2]   # |↓⟩
        }

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

    def set_superposition_state(self, state_components: List[Tuple[Chord, int, complex]]):
        """
        Set arbitrary superposition state with complex amplitudes

        Args:
            state_components: List of (chord, spin, amplitude) tuples
                             where amplitude can be complex

        Example:
            # Create (|C,↑⟩ + |C,↓⟩)/√2
            walker.set_superposition_state([
                (Chord(0, True, 0), 0, 1/np.sqrt(2)),
                (Chord(0, True, 0), 2, 1/np.sqrt(2))
            ])
        """
        self.state.fill(0)
        for chord, spin, amplitude in state_components:
            idx = self._get_index(chord, spin)
            self.state[idx] = amplitude

        # Normalize
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state /= norm

    def step(self):
        """
        Perform one quantum walk step:
        1. Apply Grover coin to spin state
        2. Shift according to configured transformation order
        """
        new_state = np.zeros_like(self.state)

        # Map transformation letters to functions
        transform_funcs = {
            'L': apply_L,
            'P': apply_P,
            'R': apply_R
        }

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

            # Shift according to configured transformation order
            # For each spin state, apply its mapped transformation
            for spin_idx in range(3):
                if abs(new_spin_state[spin_idx]) > 1e-12:
                    # Get the transformation for this spin
                    transform_letter = self.transform_map[spin_idx]
                    transform_func = transform_funcs[transform_letter]

                    # Apply transformation and update state
                    new_chord = transform_func(chord)
                    new_idx = self._get_index(new_chord, spin_idx)
                    new_state[new_idx] += new_spin_state[spin_idx]

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

    def get_full_state(self) -> dict:
        """
        Get full quantum state as dictionary of complex amplitudes

        Returns:
            Dictionary mapping (chord_root, is_major, spin) -> complex amplitude
        """
        state_dict = {}
        for i in range(self.state_size):
            chord, spin = self._get_state_params(i)
            key = (chord.root, chord.is_major, spin)
            state_dict[key] = self.state[i]
        return state_dict


class QutritWalkSimulator:
    """Core qutrit walk logic without any I/O (no MIDI, no printing)"""

    def __init__(self, initial_chord: Chord, initial_spin: int = 0,
                 initial_superposition: Optional[List[Tuple[Chord, int, complex]]] = None,
                 transform_order: str = "LPR"):
        """
        Initialize the simulator

        Args:
            initial_chord: Starting chord (used if initial_superposition is None)
            initial_spin: Initial spin state (0=|↑⟩, 1=|→⟩, 2=|↓⟩)
            initial_superposition: Optional list of (chord, spin, amplitude) tuples
                                   for custom superposition states
            transform_order: Order of transformations (default: "LPR")
        """
        self.walker = QutritWalker(transform_order=transform_order)
        self.current_chord = initial_chord

        if initial_superposition is not None:
            self.walker.set_superposition_state(initial_superposition)
        else:
            self.walker.set_initial_state(initial_chord, initial_spin)

        self.is_first_step = True

    def step_and_sample(self) -> dict:
        """
        Perform one step: evolve quantum state and sample next chord

        Returns:
            dict with keys:
                - 'current': Current chord (before transition)
                - 'neighbors': List of 3 neighbor chords [L, P, R]
                - 'next': Next chord (sampled from neighbors)
                - 'all_probs': Dict of all 24 chord probabilities {(root, is_major): prob}
                - 'full_state': Dict of all 72 complex amplitudes {(root, is_major, spin): amplitude}
                - 'is_first': Boolean indicating if this is the first step
        """
        result = {
            'current': self.current_chord,
            'is_first': self.is_first_step
        }

        if self.is_first_step:
            self.is_first_step = False
            # For first step, just return current chord info
            result['neighbors'] = []
            result['next'] = self.current_chord
            result['all_probs'] = self.walker.get_chord_probabilities()
            result['full_state'] = self.walker.get_full_state()
            return result

        # Evolve the quantum state
        self.walker.step()

        # Get the 3 neighbors
        neighbors = [
            apply_L(self.current_chord),
            apply_P(self.current_chord),
            apply_R(self.current_chord)
        ]

        # Get full probability distribution
        all_probs = self.walker.get_chord_probabilities()

        # Get full quantum state
        full_state = self.walker.get_full_state()

        # Sample next chord from neighbors
        next_chord = self.walker.measure_constrained(neighbors)

        result['neighbors'] = neighbors
        result['next'] = next_chord
        result['all_probs'] = all_probs
        result['full_state'] = full_state

        # Update current chord for next iteration
        self.current_chord = next_chord

        return result


class QutritWalkMusic(MIDIGenerator):
    """Qutrit quantum walk music generator"""

    def __init__(self):
        super().__init__(description="Qutrit Walk - Quantum music generation on triadic tonnetz")
        self.chord_duration = 1.0  # seconds
        self.chord_velocity = 70
        self.initial_chord = Chord(0, True, 0)  # C major
        self.initial_spin = 0  # |↑⟩
        self.initial_superposition: Optional[List[Tuple[Chord, int, complex]]] = None
        self.transform_order = "LPR"  # Default transformation order
        self.simulator: QutritWalkSimulator = None  # type: ignore - Will be initialized in setup()

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
        parser.add_argument('--initial-state', type=str,
                          help='Initial quantum state specification (e.g., "C:up+down" for symmetric superposition)')
        parser.add_argument('--transform-order', type=str, default='LPR',
                          help='Transformation order for spin states: permutation of LPR (default: LPR). '
                               'Example: "RLP" means ↑→R, →→L, ↓→P')
        return parser

    def setup(self):
        """Initialize qutrit walk state"""
        self.simulator = QutritWalkSimulator(
            self.initial_chord,
            self.initial_spin,
            self.initial_superposition,
            self.transform_order
        )

        print(f"Starting qutrit quantum walk on triadic tonnetz")
        print(f"Chord duration: {self.chord_duration}s")
        print(f"Transform order: {self.transform_order} (↑→{self.transform_order[0]}, →→{self.transform_order[1]}, ↓→{self.transform_order[2]})")
        if self.initial_superposition:
            print(f"Initial state: Custom superposition")
        else:
            spin_names = ['↑', '→', '↓']
            print(f"Initial state: |{self.initial_chord}, {spin_names[self.initial_spin]}⟩")
        print()

    def generate_step(self):
        """Measure quantum state and play chord"""
        import mido
        import threading
        import time

        # Get next step from simulator
        result = self.simulator.step_and_sample()

        # Print output
        if result['is_first']:
            print(f"{str(result['current']):5s} | Initial")
            chord_to_play = result['current']
        else:
            # Format neighbor probabilities for display
            neighbor_probs = []
            for chord in result['neighbors']:
                key = (chord.root, chord.is_major)
                prob = result['all_probs'].get(key, 0.0)
                neighbor_probs.append((chord, prob))

            # Sort by probability and format for display
            neighbor_probs.sort(key=lambda x: x[1], reverse=True)
            prob_str = ", ".join([f"{str(chord):5s}: {prob:.3f}" for chord, prob in neighbor_probs])

            # Print: current chord | neighbor probabilities → chosen chord
            print(f"{str(result['current']):5s} | [{prob_str}] → {str(result['next']):5s}")
            chord_to_play = result['next']

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

    # Handle initial state specification
    if args.initial_state:
        # Parse custom initial state
        try:
            qw.initial_superposition = parse_initial_state(args.initial_state)
            # Extract first chord as current chord for display
            qw.initial_chord = qw.initial_superposition[0][0]
        except ValueError as e:
            print(f"Error parsing initial state: {e}")
            exit(1)
    else:
        # Use simple initial state from --root and --minor
        try:
            root_idx = Chord.NOTE_NAMES.index(args.root)
        except ValueError:
            print(f"Error: Invalid root note '{args.root}'. Valid notes: {', '.join(Chord.NOTE_NAMES)}")
            exit(1)
        qw.initial_chord = Chord(root_idx, not args.minor, 0)

    # Validate transform order
    transform_order = args.transform_order.upper()
    if sorted(transform_order) != ['L', 'P', 'R']:
        print(f"Error: Invalid transform order '{args.transform_order}'. Must be a permutation of LPR.")
        print(f"Valid options: LPR, LRP, PLR, PRL, RLP, RPL")
        exit(1)

    # Set parameters from args
    qw.base_octave = args.base_octave
    qw.chord_duration = args.duration
    qw.chord_velocity = args.velocity
    qw.transform_order = transform_order

    # Select and open MIDI port
    port_name = qw.select_midi_port(args.midi_port)
    qw.initialize_midi(port_name)

    # Initialize and run
    qw.setup()
    qw.run()
