from __future__ import division, print_function

import random
import numpy as np
from midi_generator import MIDIGenerator


class Loopdrift(MIDIGenerator):
    """Loopdrift - Evolving matrix-based generative music system"""

    def __init__(self):
        super().__init__(description="Loopdrift - Generative MIDI music system")

        # Matrix parameters
        self.steps = 24
        self.note_range = 23
        self.seq = None
        self.t = 0
        self.s = 0
        self.offset = 0
        self.L = self.steps

        # Evolution parameters
        self.pmut = 0.007
        self.p_scale = 0.2

        # Density thresholds per channel
        density = 0.0005 * self.note_range
        self.density_thresh = {
            0: 1 - 20 * density,
            1: 1 - 8 * density,
            2: 1 - 3 * density
        }

        # Timing parameters
        self.bpm: int = 120
        self.time_signature: tuple = (4, 4)  # (beats_per_measure, beat_unit)
        self.steps_per_beat: int = 2  # How many grid steps per beat
        self.sleep_time = self._calculate_sleep_time()

    def setup_args(self, parser=None):
        """Add Loopdrift-specific arguments"""
        parser = super().setup_args(parser)
        parser.add_argument('--bpm', type=int, default=120,
                          help='Tempo in beats per minute (default: 120)')
        parser.add_argument('--time-signature', type=str, default='4/4',
                          help='Time signature as numerator/denominator (default: 4/4)')
        parser.add_argument('--steps-per-beat', type=int, default=2,
                          help='Number of grid steps per beat (default: 2)')
        return parser

    def _calculate_sleep_time(self):
        """Calculate sleep time based on BPM, time signature, and steps per beat"""
        #beat_unit tells us what note value gets one beat (4 = quarter note, 8 = eighth note)
        beats_per_minute = self.bpm
        beats_per_measure, beat_unit = self.time_signature

        # Calculate duration of one beat in seconds
        # For beat_unit=4 (quarter note): one beat = 60/bpm seconds
        # For beat_unit=8 (eighth note): one beat = 30/bpm seconds
        seconds_per_beat = 60.0 / beats_per_minute * (4.0 / beat_unit)

        # Calculate duration of one step (grid column)
        seconds_per_step = seconds_per_beat / self.steps_per_beat

        return seconds_per_step

    def setup(self):
        """Initialize the sequence matrix"""
        self.seq = np.random.random((self.steps, self.note_range))
        print(f"Initialized {self.steps}x{self.note_range} sequence matrix\n")

    def print_all(self, tracks):
        """Print the current state of the sequence matrix"""
        for i in range(self.seq.shape[1] - 1, -1, -1):
            print("  ", " ".join(map(lambda x:
                                     str(int(x * 100)).zfill(2), self.seq.T[i])),
                  "  *" if i in tracks else "   ")

    def generate_step(self):
        """Generate and play notes for one time step"""
        tracks_on_this_step = []
        chans_on_this_step = set()

        # Check each track to see if it should play
        for track in range(self.note_range):
            # Assign channel based on track range
            if track < self.note_range / 3:
                chan = 0
            elif track < 2 * self.note_range / 3:
                chan = 1
            else:
                chan = 2

            # Check if this track exceeds the density threshold
            if self.seq[self.t, track] > self.density_thresh[chan]:
                # Map track to scale degree (track 0 = scale degree 0, etc.)
                scale_degree = track + self.offset
                velocity = int(127.0 *
                              (self.seq[self.t, track] - self.density_thresh[chan]) /
                              (1 - self.density_thresh[chan]))
                duration = int(3000 / (chan + 1))

                self.play_note(scale_degree, velocity, duration, channel=chan)
                tracks_on_this_step.append(track)
                chans_on_this_step.add(chan)

        # Mutate the sequence
        for i in range(self.steps):
            for j in range(self.note_range):
                if random.random() < self.pmut:
                    self.seq[i, j] = random.random()

        # Print current state
        self.print_all(tracks_on_this_step)
        print("   " + " ".join(" *" if i == self.t else "  " for i in range(self.L)))
        print("")
        print("channels", chans_on_this_step)
        print("")
        print(self.L, self.scale_type, self.offset)
        print("")

        # Update offset at end of loop
        if self.t == self.L - 1:
            self.s += 1
            self.s %= self.steps
            if self.s == 0:
                self.offset += random.randrange(-2, 3)
                while self.offset > 7:
                    self.offset -= 3
                while self.offset < -7:
                    self.offset += 3

        # Advance time step
        self.t += 1
        self.t %= self.L

    def get_step_duration(self):
        """Return sleep time between steps"""
        return self.sleep_time


if __name__ == "__main__":
    from midi_generator import SCALES

    loopdrift = Loopdrift()

    # Parse arguments
    args = loopdrift.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        print(f"Using random seed: {args.seed}\n")

    # Set scale and base octave from args
    loopdrift.scale_type = args.scale
    loopdrift.scale = SCALES[args.scale]
    loopdrift.base_octave = args.base_octave

    # Set BPM and time signature
    loopdrift.bpm = args.bpm
    time_sig_parts = args.time_signature.split('/')
    if len(time_sig_parts) != 2:
        print(f"Error: Invalid time signature format '{args.time_signature}'. Use format like '4/4' or '6/8'")
        import sys
        sys.exit(1)
    try:
        loopdrift.time_signature = (int(time_sig_parts[0]), int(time_sig_parts[1]))
    except ValueError:
        print(f"Error: Time signature must contain integers, got '{args.time_signature}'")
        import sys
        sys.exit(1)
    loopdrift.steps_per_beat = args.steps_per_beat
    loopdrift.sleep_time = loopdrift._calculate_sleep_time()

    print(f"BPM: {loopdrift.bpm}, Time signature: {loopdrift.time_signature[0]}/{loopdrift.time_signature[1]}, Steps per beat: {loopdrift.steps_per_beat}")
    print(f"Step duration: {loopdrift.sleep_time:.3f} seconds\n")

    # Select and open MIDI port
    port_name = loopdrift.select_midi_port(args.midi_port)
    loopdrift.initialize_midi(port_name)

    # Initialize and run
    loopdrift.setup()
    loopdrift.run()
