from __future__ import division, print_function

import time
import random
import numpy as np
import sys
import threading
import mido
import argparse

scales = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "harmonic_minor": [0, 1, 4, 5, 7, 8, 11],
    "minor": [0, 2, 3, 5, 7, 9, 10],
}

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Loopdrift - Generative MIDI music system')
parser.add_argument('--seed', type=int, help='Random seed for reproducible sequences')
parser.add_argument('--midi-port', type=str, help='MIDI output port name or number')
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)
    random.seed(args.seed)
    print(f"Using random seed: {args.seed}\n")

# List available MIDI outputs
available_ports = mido.get_output_names()
print("Available MIDI outputs:")
for i, name in enumerate(available_ports):
    print(f"  {i}: {name}")
print()

# Select MIDI port
if args.midi_port is not None:
    # Check if argument is a number (index) or a name
    try:
        port_index = int(args.midi_port)
        if 0 <= port_index < len(available_ports):
            selected_port = available_ports[port_index]
        else:
            print(f"Error: Port index {port_index} out of range (0-{len(available_ports)-1})")
            sys.exit(1)
    except ValueError:
        # Argument is a port name
        selected_port = args.midi_port
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

outport = mido.open_output(selected_port)
print(f"Opened: {outport.name}\n")

def play_note(pitch, velocity, duration, channel=0):
    """
    Send a MIDI note with automatic note-off after duration

    Args:
        pitch: MIDI note number (0-127)
        velocity: note velocity (0-127)
        duration: duration in milliseconds
        channel: MIDI channel (0-15)
    """
    # Send note on immediately
    msg_on = mido.Message('note_on', note=pitch, velocity=velocity, channel=channel)
    outport.send(msg_on)

    # Schedule note off in background thread
    def send_note_off():
        time.sleep(duration / 1000.0)
        msg_off = mido.Message('note_off', note=pitch, velocity=0, channel=channel)
        outport.send(msg_off)

    threading.Thread(target=send_note_off, daemon=True).start()

steps = 24
note_range = 23
seq = np.random.random((steps, note_range))
t = 0
s = 0
offset = 0
pmut = 0.007
p_timesig = 1.0
p_scale = 0.2
L = int(steps)
density = 0.0001 * note_range
density_thresh = {0: 1 - 20 *density,
                  1: 1 - 2  *density,
                  2: 1 - 8  *density}

scale_key = "minor"
scale = scales[scale_key]
sleep_time = 1.0/2

def diatonic_map(n):
    global scale
    octave = n // 7
    chroma = n % 7
    retval = octave * 12 + scale[chroma]
    return retval


def print_all(seq, tracks):
    for i in range(seq.shape[1] - 1, -1, -1):
        print("  ", " ".join(map(lambda x:
                                 str(int(x*100)).zfill(2), seq.T[i])),
              "  *" if i in tracks else "   ")

def f(dt):
    global t, s, offset, L, scale_key
    tracks_on_this_step = []
    chans_on_this_step = set()
    for track in range(note_range):
        if track < note_range / 3:
            chan = 0
        elif track < 2 * note_range / 3:
            chan = 1
        else:
            chan = 2
        if seq[t, track] > density_thresh[chan]:
            pitch = diatonic_map(28+track+offset)
            velocity = int(127.0 *
                          (seq[t, track] - density_thresh[chan]) /
                          (1 - density_thresh[chan]))
            duration = int(3000/(chan+1))
            play_note(pitch, velocity, duration, channel=chan)
            tracks_on_this_step.append(track)
            chans_on_this_step.add(chan)
    for i in range(steps):
        for j in range(note_range):
            if random.random() < pmut:
                seq[i, j] = random.random()
    print_all(seq, tracks_on_this_step)
    print("   " + " ".join(" *" if i == t else "  " for i in range(L)))
    print("")
    print("channels", chans_on_this_step)
    print("")
    print(L, scale_key, offset)
    print("")

    if t == L - 1:
        # controller = 35
        # val = random.randrange(128)
        # b.controlChange(controller, val)
        s += 1
        s %= steps
        if s == 0:
            offset += random.randrange(-2, 3)
            while offset > 7:
                offset -= 3
            while offset < -7:
                offset += 3
            # if random.random() < p_timesig:
            #     L = random.randrange(int(steps/2), steps+1, 2)
            # if random.random() < p_scale:
            #     scale_key = random.choice(list(scales.keys()))
            #     scale = scales[scale_key]
        
    t += 1
    t %= L

try:
    while True:
        f(0)
        time.sleep(sleep_time)
except KeyboardInterrupt:
    print("\nStopping...")
    outport.close()
    print("MIDI port closed")


