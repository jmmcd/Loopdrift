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
        self.p_timesig = 1.0
        self.p_scale = 0.2

        # Density thresholds per channel
        density = 0.0005 * self.note_range
        self.density_thresh = {
            0: 1 - 20 * density,
            1: 1 - 8 * density,
            2: 1 - 3 * density
        }

        # Timing
        self.sleep_time = 1.0 / 2

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

    # Select and open MIDI port
    port_name = loopdrift.select_midi_port(args.midi_port)
    loopdrift.initialize_midi(port_name)

    # Initialize and run
    loopdrift.setup()
    loopdrift.run()
