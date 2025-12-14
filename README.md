# Bork Boy 9000 (Destructive / Chaotic Filtering)

Turns good-sounding signals into noisy, aliased, chaotic messes.  
A Python3 simulation of a guitar effect pedal, with potential C implementations for microcontrollers in physical pedals.

## Features

- Bit depth reduction (bit-crush)
- Downsampling with optional interpolation
- Random amplitude modulation (SHRED)
- Sample dropouts (GLITCH)
- Chaotic low-pass filtering (CHAOS)

## Requirements

- Python 3.x
- numpy
- scipy
- matplotlib
- soundfile

Install dependencies via pip:

```bash
pip install numpy scipy matplotlib soundfile
```

## Running the Script

Run the script using:

```bash
python3 bork_boy.py
```

Setting the input and output file in main():

```python
def main(input_file="audio_input.wav", output_file="audio_output.wav"):
  ...
```

Effect parameters can be adjusted at the top of the script:

```python
# ======================================
# Effect Parameters
# ======================================
BITS = 4                     # Bit depth (higher = cleaner)
DOWNSAMPLE_FACTOR = 2 ** 4   # Downsampling factor (2**0 = original rate)
SHRED = 0.5                  # Random amplitude modulation depth
GLITCH = 0.5                 # Probability of sample dropouts
CHAOS = 0.5                  # Random cutoff modulation amount

# Interpolation mode:
# False = Zero-order hold
# True  = Linear interpolation
SMOOTH = False
```

## Accepted Values

- input_file:
   - WAVE file (with recommended samplerate of 48 kHz)
- BITS:
   - MAX = input file bit depth (16 is safe assumption)
   - MIN = 4 (everything below turns signal into pure noise)
- DOWNSAMPLE_FACTOR:
   - MAX = 2**0 (original sample rate)
   - MIN = 2**7 (128 times lower than original)
- SHRED:
   - MAX = 1 (high amplitude modulation)
   - MIN = 0 (no amplitude modulation)
- GLITCH:
   - MAX = 1 (all samples are dropped)
   - MIN = 0 (no samples are dropped)
- CHAOS:
   - MAX = 1 (high variability in filtering)
   - MIN = 0 (zero variability in filtering)
- SMOOTH:
   - True (no distortion effect)
   - False (zero-order hold introduces distortion)
