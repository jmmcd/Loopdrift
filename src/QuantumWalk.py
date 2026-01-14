import numpy as np
from typing import Tuple, List
import random
from midi_generator import MIDIGenerator


def note_to_name(note: int) -> str:
    """Convert scale degree to name"""
    names = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    return names[note % 7]


class QuantumWalker:
    """Quantum walk state and evolution"""

    def __init__(self):
        self.num_pitches = 7   # 0-6 representing scale degrees
        self.num_spins = 2     # up/down
        self.state_size = self.num_pitches * self.num_pitches * self.num_spins * self.num_spins

        # Initialize quantum state vector (196 dimensions)
        self.state = np.zeros(self.state_size, dtype=complex)

    def _get_index(self, x: int, y: int, a: int, b: int) -> int:
        """Convert (x, y, a, b) to linear index"""
        return x * (7 * 2 * 2) + y * (2 * 2) + a * 2 + b

    def _get_state_params(self, index: int) -> Tuple[int, int, int, int]:
        """Convert linear index back to (x, y, a, b)"""
        x = index // (7 * 2 * 2)
        remainder = index % (7 * 2 * 2)
        y = remainder // (2 * 2)
        remainder = remainder % (2 * 2)
        a = remainder // 2
        b = remainder % 2
        return x, y, a, b

    def set_initial_state(self, x: int, y: int, a: int = 0, b: int = 0):
        """Set initial state |x, y, ab⟩ where spin 0=↑, 1=↓"""
        self.state.fill(0)
        index = self._get_index(x % 7, y % 7, a, b)
        self.state[index] = 1.0

    def set_entangled_state(self, states: List[Tuple[int, int, int, int]], amplitudes: List[complex]):
        """Set superposition state with given amplitudes"""
        self.state.fill(0)
        for (x, y, a, b), amp in zip(states, amplitudes):
            index = self._get_index(x % 7, y % 7, a, b)
            self.state[index] = amp

        # Normalize
        norm = np.linalg.norm(self.state)
        if norm > 0:
            self.state /= norm

    def step(self):
        """Perform one quantum walk step: Hadamard + conditional shift"""
        new_state = np.zeros_like(self.state)

        for i in range(self.state_size):
            if abs(self.state[i]) < 1e-12:
                continue

            x, y, a, b = self._get_state_params(i)

            # Hadamard operation on both spins
            # H|↑⟩ = (1/√2)(|↑⟩ + |↓⟩), H|↓⟩ = (1/√2)(|↑⟩ - |↓⟩)
            # Combined operation: H⊗H applied to both spins
            coeff = 0.5  # (1/√2) × (1/√2) = 1/2

            hadamard_coeffs = []
            if a == 0 and b == 0:  # ↑↑
                hadamard_coeffs = [(0, 0, coeff), (0, 1, coeff), (1, 0, coeff), (1, 1, coeff)]
            elif a == 0 and b == 1:  # ↑↓
                hadamard_coeffs = [(0, 0, coeff), (0, 1, -coeff), (1, 0, coeff), (1, 1, -coeff)]
            elif a == 1 and b == 0:  # ↓↑
                hadamard_coeffs = [(0, 0, coeff), (0, 1, coeff), (1, 0, -coeff), (1, 1, -coeff)]
            else:  # ↓↓
                hadamard_coeffs = [(0, 0, coeff), (0, 1, -coeff), (1, 0, -coeff), (1, 1, coeff)]

            # Apply conditional shift for each Hadamard outcome
            for new_a, new_b, coeff in hadamard_coeffs:
                # Conditional shift based on spin (diatonic steps)
                if new_a == 0:  # spin up: up scale
                    new_x = (x + 1) % 7
                else:  # spin down: down scale
                    new_x = (x - 1) % 7

                if new_b == 0:  # spin up: up scale
                    new_y = (y + 1) % 7
                else:  # spin down: down scale
                    new_y = (y - 1) % 7

                new_index = self._get_index(new_x, new_y, new_a, new_b)
                new_state[new_index] += self.state[i] * coeff

        self.state = new_state

    def measure(self) -> Tuple[int, int, int, int]:
        """Measure the quantum state and return (x, y, a, b)"""
        probabilities = np.abs(self.state) ** 2

        # Sample from probability distribution
        index = np.random.choice(self.state_size, p=probabilities)
        x, y, a, b = self._get_state_params(index)

        return x, y, a, b

    def get_probabilities(self) -> dict:
        """Get probability distribution over note pairs"""
        probs = {}
        for i in range(self.state_size):
            if abs(self.state[i]) > 1e-12:
                x, y, a, b = self._get_state_params(i)
                key = (x, y)
                if key not in probs:
                    probs[key] = 0
                probs[key] += abs(self.state[i]) ** 2
        return probs


class QuantumWalkMusic(MIDIGenerator):
    """Quantum Walk music generator"""

    def __init__(self):
        super().__init__(description="Quantum Walk - Quantum music generation")
        self.walker = QuantumWalker()
        self.note_duration = 0.8  # seconds
        self.initial_notes = (0, 7)  # C and G
        self.note_velocity = 80

    def setup_args(self, parser=None):
        """Add quantum walk specific arguments"""
        parser = super().setup_args(parser)
        parser.add_argument('--duration', type=float, default=0.8,
                          help='Note duration in seconds (default: 0.8)')
        parser.add_argument('--velocity', type=int, default=80,
                          help='MIDI velocity (default: 80)')
        return parser

    def setup(self):
        """Initialize quantum walk state"""
        # Initialize with entangled state for more interesting evolution
        self.walker.set_entangled_state(
            [(self.initial_notes[0], self.initial_notes[1], 0, 0),
             ((self.initial_notes[0] + 1) % 7, (self.initial_notes[1] + 1) % 7, 0, 0)],
            [1/np.sqrt(2), 1/np.sqrt(2)]
        )
        print(f"Starting quantum music generation")
        print(f"Note duration: {self.note_duration}s")
        print(f"Initial state: {note_to_name(self.initial_notes[0])}-{note_to_name(self.initial_notes[1])}")
        print()

    def generate_step(self):
        """Measure quantum state and play notes"""
        x, y, a, b = self.walker.measure()

        print(f"Measured: {note_to_name(x)}-{note_to_name(y)} (scale degrees {x}, {y}, spins {a}, {b})")

        # Play both notes of the pair using fire-and-forget
        duration_ms = int(self.note_duration * 1000)
        self.play_note(x, self.note_velocity, duration_ms, channel=0)

        if x != y:  # Only play second note if different
            self.play_note(y, self.note_velocity, duration_ms, channel=0)

        # Evolve quantum state for next step
        self.walker.step()

    def get_step_duration(self):
        """Return note duration (time between measurements)"""
        return self.note_duration


if __name__ == "__main__":
    qw = QuantumWalkMusic()

    # Parse arguments
    args = qw.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}\n")

    # Set parameters from args
    qw.scale_type = args.scale
    qw.base_octave = args.base_octave
    qw.note_duration = args.duration
    qw.note_velocity = args.velocity

    # Select and open MIDI port
    port_name = qw.select_midi_port(args.midi_port)
    qw.initialize_midi(port_name)

    # Initialize and run
    qw.setup()
    qw.run()
