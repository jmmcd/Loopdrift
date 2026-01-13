import numpy as np
from typing import Tuple, List
import random
import time

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    print("mido not available. Install with: pip install mido python-rtmidi")

def note_to_name(note: int) -> str:
    """Convert scale degree to name"""
    names = ['C', 'D', 'E', 'F', 'G', 'A', 'B']
    return names[note % 7]

class QuantumWalker:
    def __init__(self):
        self.num_pitches = 7   # 0-6 representing major scale degrees
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
            sqrt2_inv = 1.0 / np.sqrt(2)
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
    
    def measure(self) -> Tuple[int, int]:
        """Measure the quantum state and return (x, y) note pair"""
        probabilities = np.abs(self.state) ** 2
        
        # Sample from probability distribution
        index = np.random.choice(self.state_size, p=probabilities)
        x, y, a, b = self._get_state_params(index)
        
        # Collapse to measured state
        # self.state.fill(0)
        # self.state[index] = 1.0
        
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

class MidiPlayer:
    def __init__(self, port_name: str = None, base_octave: int = 4, init_delay: float = 2.5):
        self.base_octave = base_octave
        self.current_notes = []

        if not MIDO_AVAILABLE:
            print("MIDI output disabled - mido not available")
            self.midi_out = None
            return

        # List available ports
        available_ports = mido.get_output_names()
        print(f"Available MIDI ports: {available_ports}")

        if port_name:
            # Try to find specified port
            for port in available_ports:
                if port_name.lower() in port.lower():
                    self.midi_out = mido.open_output(port)
                    print(f"Connected to MIDI port: {port}")
                    time.sleep(init_delay)
                    return

        # If no port found or no port_name specified, try GarageBand Virtual In
        if 'GarageBand Virtual In' in available_ports:
            self.midi_out = mido.open_output('GarageBand Virtual In')
            print("Connected to MIDI port: GarageBand Virtual In")
        elif available_ports:
            # Use first available port
            self.midi_out = mido.open_output(available_ports[0])
            print(f"Connected to first available MIDI port: {available_ports[0]}")
        else:
            print("No MIDI ports available")
            self.midi_out = None
            return

        # Wait for MIDI port to be ready
        #print(f"Waiting {init_delay}s for MIDI port to initialize...")
        #time.sleep(init_delay)
            
    def note_to_midi(self, note: int) -> int:
        """Convert scale degree (0-6) to MIDI note number in C major"""
        # C major scale intervals: 0, 2, 4, 5, 7, 9, 11 semitones
        major_scale = [0, 2, 4, 5, 7, 9, 11]
        octave = note // 7
        degree = note % 7
        return 12 + (self.base_octave * 12) + (octave * 12) + major_scale[degree]
    
    def play_note_pair(self, x: int, y: int, duration: float = 0.5, velocity: int = 80):
        """Play a pair of notes simultaneously"""
        if not self.midi_out:
            print(f"MIDI: {note_to_name(x)}-{note_to_name(y)} ({x}, {y})")
            return

        if self.current_notes:
            # Stop previous notes
            print("stopping")
            self.stop_current_notes()

        # Convert to MIDI note numbers
        midi_x = self.note_to_midi(x)
        midi_y = self.note_to_midi(y)

        # Send note on messages
        msg_x = mido.Message('note_on', note=midi_x, velocity=velocity)
        self.midi_out.send(msg_x)

        if midi_x != midi_y:  # Avoid duplicate if same note
            print("non-duplicates")
            msg_y = mido.Message('note_on', note=midi_y, velocity=velocity)
            self.midi_out.send(msg_y)
        else:
            print("duplicates")

        # Small delay to ensure MIDI messages are processed
        time.sleep(0.01)

        self.current_notes = [midi_x, midi_y] if midi_x != midi_y else [midi_x]

        print(f"MIDI: {note_to_name(x)}-{note_to_name(y)} (MIDI {midi_x}, {midi_y})")
    
    def stop_current_notes(self):
        """Stop currently playing notes"""
        if not self.midi_out:
            return

        for note in self.current_notes:
            msg = mido.Message('note_off', note=note, velocity=0)
            self.midi_out.send(msg)
        self.current_notes = []

    def close(self):
        """Clean up MIDI connection"""
        self.stop_current_notes()
        if self.midi_out:
            self.midi_out.close()

def play_quantum_music_live(steps: int = 20, note_duration: float = 1.0, 
                           initial_notes: Tuple[int, int] = (0, 7),
                           midi_port: str = None):
    """Generate and play quantum music in real-time"""
    player = MidiPlayer(midi_port)
    walker = QuantumWalker()
    
    try:
        # Initialize with entangled state for more interesting evolution
        walker.set_entangled_state(
            [(initial_notes[0], initial_notes[1], 0, 0), 
             ((initial_notes[0] + 1) % 7, (initial_notes[1] + 1) % 7, 0, 0)],
            [1/np.sqrt(2), 1/np.sqrt(2)]
        )
        
        print(f"Starting quantum music generation for {steps} steps...")
        print(f"Note duration: {note_duration}s")
        
        # Small delay to ensure MIDI is ready
        time.sleep(0.1)
        
        for step in range(steps):
            x, y, a, b = walker.measure()
            
            print(f"Step {step+1}: {x} {y} {a} {b}")
            player.play_note_pair(x, y, note_duration)
            
            # Always wait after playing, even for first note
            time.sleep(note_duration)
            
            if step < steps - 1:  # Don't evolve after last step
                walker.step()
        
        # Let final notes ring
        time.sleep(note_duration)
        
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        player.close()

def generate_quantum_music(steps: int = 20, initial_notes: Tuple[int, int] = (0, 4)) -> List[Tuple[int, int]]:
    """Generate a sequence of note pairs using quantum walk"""
    walker = QuantumWalker()
    
    # Start with C-G (perfect fifth in scale degrees 0-4) or create entangled state
    if random.random() < 0.5:
        # Simple initial state
        walker.set_initial_state(initial_notes[0], initial_notes[1])
    else:
        # Entangled state: superposition of two consonant intervals
        walker.set_entangled_state(
            [(0, 4, 0, 0), (1, 5, 0, 0)],  # {C,G} and {D,A}
            [1/np.sqrt(2), 1/np.sqrt(2)]
        )
    
    sequence = []
    for _ in range(steps):
        walker.step()
        notes = walker.measure()
        sequence.append(notes)
    
    return sequence

# Example usage
if False:
    # Generate quantum music sequence
    music_sequence = generate_quantum_music(20)
    
    print("Quantum Music Sequence:")
    for i, (x, y) in enumerate(music_sequence):
        print(f"Step {i+1}: {note_to_name(x)}-{note_to_name(y)} ({x}, {y})")
    
    # Demonstrate entangled state evolution
    print("\nEntangled State Example:")
    walker = QuantumWalker()
    walker.set_entangled_state(
        [(0, 4, 0, 0), (1, 5, 0, 0)],
        [1/np.sqrt(2), 1/np.sqrt(2)]
    )
    
    for step in range(3):
        probs = walker.get_probabilities()
        print(f"\nStep {step}:")
        for (x, y), prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            if prob > 0.00001:  # Only show significant probabilities
                print(f"  {note_to_name(x)}-{note_to_name(y)}: {prob:.3f}")
        
        if step < 2:
            walker.step()
    
if __name__ == "__main__":
    # Play live quantum music (uncomment to try)
    print("\nPress Ctrl+C to stop live playback...")
    play_quantum_music_live(steps=4, note_duration=0.8)