"""
Base class for generative MIDI systems
Provides common infrastructure for port selection, note sending, and event loop
"""

from abc import ABC, abstractmethod
import time
import sys
import threading
import argparse
import mido


SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 9, 10],
    "harmonic_minor": [0, 2, 3, 5, 7, 8, 11],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
}


class MIDIGenerator(ABC):
    """Base class for generative MIDI music systems"""

    def __init__(self, description="Generative MIDI music system"):
        """
        Initialize the MIDI generator

        Args:
            description: Description for argument parser
        """
        self.description = description
        self.outport = None
        self.base_octave = 4
        self.scale_type = "minor"
        self.scale = SCALES[self.scale_type]

    def setup_args(self, parser=None):
        """
        Setup command-line arguments. Can be extended by subclasses.

        Args:
            parser: Optional existing ArgumentParser to add to

        Returns:
            ArgumentParser instance
        """
        if parser is None:
            parser = argparse.ArgumentParser(description=self.description)

        parser.add_argument('--seed', type=int, help='Random seed for reproducible sequences')
        parser.add_argument('--midi-port', type=str, help='MIDI output port name or number')
        parser.add_argument('--scale', type=str, choices=list(SCALES.keys()),
                          default='minor', help='Scale to use (default: minor)')
        parser.add_argument('--base-octave', type=int, default=4,
                          help='Base octave for note mapping (default: 4)')

        return parser

    def parse_args(self, parser=None):
        """Parse command-line arguments"""
        parser = self.setup_args(parser)
        return parser.parse_args()

    def select_midi_port(self, port_arg=None):
        """
        Select and open a MIDI output port

        Args:
            port_arg: Optional port name or index from command line

        Returns:
            Selected port name
        """
        available_ports = mido.get_output_names()
        print("Available MIDI outputs:")
        for i, name in enumerate(available_ports):
            print(f"  {i}: {name}")
        print()

        # Select MIDI port
        if port_arg is not None:
            # Check if argument is a number (index) or a name
            try:
                port_index = int(port_arg)
                if 0 <= port_index < len(available_ports):
                    selected_port = available_ports[port_index]
                else:
                    print(f"Error: Port index {port_index} out of range (0-{len(available_ports)-1})")
                    sys.exit(1)
            except ValueError:
                # Argument is a port name
                selected_port = port_arg
                if selected_port not in available_ports:
                    print(f"Error: MIDI port '{selected_port}' not found")
                    sys.exit(1)
        elif len(available_ports) == 0:
            print("Error: No MIDI outputs available")
            sys.exit(1)
        elif len(available_ports) == 1:
            # Only one port, use it automatically
            selected_port = available_ports[0]
            print(f"Automatically selecting the only available port: {selected_port}\n")
        else:
            # Multiple ports, show menu
            while True:
                try:
                    choice = input(f"Select MIDI output (0-{len(available_ports)-1}): ")
                    port_index = int(choice)
                    if 0 <= port_index < len(available_ports):
                        selected_port = available_ports[port_index]
                        break
                    else:
                        print(f"Please enter a number between 0 and {len(available_ports)-1}")
                except ValueError:
                    print("Please enter a valid number")
                except (KeyboardInterrupt, EOFError):
                    print("\nCancelled")
                    sys.exit(0)
            print()

        return selected_port

    def initialize_midi(self, port_name):
        """Open the MIDI output port"""
        self.outport = mido.open_output(port_name)
        print(f"Opened: {self.outport.name}\n")

    def diatonic_to_midi(self, scale_degree):
        """
        Convert a scale degree to a MIDI note number

        Args:
            scale_degree: Scale degree (can be negative or > 6 for different octaves)

        Returns:
            MIDI note number (0-127)
        """
        octave = scale_degree // 7
        chroma = scale_degree % 7
        midi_note = 12 + (self.base_octave * 12) + (octave * 12) + self.scale[chroma]
        return midi_note

    def play_note(self, scale_degree, velocity, duration, channel=0):
        """
        Play a note using fire-and-forget (schedules note-off automatically)

        Args:
            scale_degree: Scale degree to play
            velocity: MIDI velocity (0-127)
            duration: Duration in milliseconds
            channel: MIDI channel (0-15)
        """
        if not self.outport:
            return

        # Convert scale degree to MIDI note
        pitch = self.diatonic_to_midi(scale_degree)

        # Validate MIDI note range
        if pitch < 0 or pitch > 127:
            return

        # Send note on immediately
        msg_on = mido.Message('note_on', note=pitch, velocity=velocity, channel=channel)
        self.outport.send(msg_on)

        # Schedule note off in background thread
        def send_note_off():
            time.sleep(duration / 1000.0)
            msg_off = mido.Message('note_off', note=pitch, velocity=0, channel=channel)
            self.outport.send(msg_off)

        threading.Thread(target=send_note_off, daemon=True).start()

    def cleanup(self):
        """Clean up MIDI connection"""
        if self.outport:
            print("\nStopping...")

            # Send note-off for all notes on all channels to prevent stuck notes
            print("Sending all notes off...")
            for channel in range(16):
                for note in range(128):
                    msg_off = mido.Message('note_off', note=note, velocity=0, channel=channel)
                    self.outport.send(msg_off)

            self.outport.close()
            print("MIDI port closed")

    @abstractmethod
    def setup(self):
        """
        Initialize the generator-specific state
        Called once before the main loop starts
        """
        pass

    @abstractmethod
    def generate_step(self):
        """
        Generate and play notes for one time step
        Should call self.play_note() as needed
        """
        pass

    @abstractmethod
    def get_step_duration(self) -> float:
        """
        Return the duration to sleep between steps (in seconds)

        Returns:
            Sleep duration in seconds
        """
        pass

    def run(self):
        """
        Main event loop - runs indefinitely until interrupted
        """
        try:
            while True:
                self.generate_step()
                time.sleep(self.get_step_duration())
        except KeyboardInterrupt:
            self.cleanup()
