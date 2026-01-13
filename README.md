# Loopdrift and Quantum Walk

Two experiments in simple generative music.

# Installation

`pip install mido python-rtmidi`

Run GarageBand, Logic Pro, or your favourite MIDI synth.


# Loopdrift

This is about the simplest generative music system I can make, while I still enjoy listening to the output.

We have a grid of numbers like this. Time is from left to right, pitch is from bottom to top. Each number x at time t and pitch p means: "if x is above some threshold, then at time t, play pitch p with velocity x". 

```
   19 82 64 10 97 34 77 67 32 23 15 88 53 49 84 77 26 13 53 69 96 96 85 81
   26 52 58 52 96 21 49 19 35 75 58 71 81 17 45 12 20 58 61 23 29 51 26 30
   16 15 10 09 87 47 74 10 07 47 04 41 49 74 53 75 22 50 78 90 00 70 68 13
   08 80 68 61 99 83 60 17 02 92 70 69 05 94 22 35 47 91 96 33 78 01 54 93
   76 49 51 25 52 63 96 64 13 17 99 86 14 12 54 30 08 56 03 35 69 98 02 88
   63 87 45 93 86 68 37 02 33 77 66 91 40 06 23 35 40 00 65 85 35 95 18 19
   08 82 46 61 62 54 80 21 98 08 81 54 36 12 60 04 74 55 78 23 36 80 62 84
   97 05 26 11 06 06 07 48 00 09 36 23 88 36 07 31 43 64 82 87 52 68 29 22
   08 46 45 22 51 29 11 39 56 82 95 06 97 55 61 10 97 92 15 15 54 64 75 29
   88 98 56 37 68 45 73 62 85 18 52 95 67 23 65 50 94 56 47 95 20 49 28 39
   05 44 69 57 47 54 22 76 05 28 70 85 01 11 47 10 39 01 45 14 94 55 85 02
   22 20 69 08 54 50 41 25 62 19 69 91 54 78 88 27 08 69 60 73 31 32 94 93
   53 56 21 32 46 83 33 39 79 86 81 92 65 38 89 70 65 75 70 24 33 10 59 45
   57 00 97 72 13 37 75 14 72 54 55 22 17 25 43 21 51 40 73 50 75 72 31 66
   00 69 14 99 17 92 33 13 07 05 43 56 43 37 40 30 55 84 68 58 14 14 21 23
   12 68 49 70 69 41 27 50 40 49 09 01 17 15 28 98 12 09 89 77 44 66 97 40   *
   17 82 66 32 63 63 23 05 27 43 55 79 32 42 86 77 19 81 18 73 57 18 18 84
   83 63 18 67 96 70 66 95 96 36 57 18 52 59 62 33 46 56 01 75 17 84 10 64
   25 85 89 37 92 36 54 52 92 57 17 03 22 42 14 90 42 28 17 01 88 89 08 66
   69 24 54 96 47 31 97 87 55 58 24 89 81 85 21 76 97 72 86 00 85 01 39 03
   43 96 23 68 53 18 53 40 05 20 72 10 16 22 75 66 67 79 61 46 86 97 48 81
   48 81 32 62 85 02 04 96 61 30 36 57 77 37 67 13 55 65 65 16 00 33 96 31
   27 25 37 47 81 16 76 63 98 93 74 13 45 87 65 62 28 13 56 78 69 71 00 97
                                                 *
```

So at each time-step we get 0, 1, or more notes, of different velocity values. The pitches are 2-3 octaves of a diatonic scale, so they don't clash too much. After completing all time-steps, we loop, so we get patterns. But after every time-step, we take a step in a random walk, ie mutate some of the values in the grid. So, the loop drifts.

There are some other tricks, including pitch offsets which change rarely, three different channels representing different pitch ranges, mapped to different instruments (requires multi-channel setup with track record buttons enabled in Logic Pro or another serious DAW, not GarageBand), different scales, and possibility for the grid to loop early (so, shorter loops).

Good instruments: piano, acoustic guitar, bass, simple piano-like synths. 

## Running

`python Loopdrift.py`

You might have to select a MIDI output. There are also command-line arguments for the MIDI output and the random seed.


## TODO

* Make it into a simple class
* One object represents a pitch range and an instrument, so much easier to control multiple instruments



# Quantum Walk

This is a very early piece of work with the Galway Music, Maths, Computing, etc, group. The code is by Claude. This is in flux so not much documentation yet.



# A Quantum Singularity 

This is a tune in The Session!

https://thesession.org/tunes/8612

