#!/usr/bin/env python3

"""generate shared little-endian f32 data and initial weights; no dependencies."""

from pathlib import Path
import math
import random
import struct


rng = random.Random(1234)
NTRAIN, NTEST, INPUTS, HIDDEN, OUTPUTS = 1024, 512, 2, 16, 2

def samples(n):
    x = [rng.uniform(-1, 1) for _ in range(n * 2)]
    y = [int(x[2*i] * x[2*i+1] > 0) for i in range(n)]
    return x, y

def weights(nout, nin):
    scale = math.sqrt(6 / nin) # He uniform, suitable for ReLU.
    return [rng.uniform(-scale, scale) for _ in range(nout * nin)]

train_x, train_y = samples(NTRAIN)
test_x, test_y = samples(NTEST)
arrays = [('f', train_x), ('I', train_y), ('f', test_x), ('I', test_y),
          ('f', weights(HIDDEN, INPUTS)), ('f', [0]*HIDDEN),
          ('f', weights(OUTPUTS, HIDDEN)), ('f', [0]*OUTPUTS)]
          
path = Path(__file__).with_name('model.bin')
with path.open('wb') as f:
    f.write(b'NN2CLS01')
    f.write(struct.pack('<5I', NTRAIN, NTEST, INPUTS, HIDDEN, OUTPUTS))
    for code, values in arrays:
        f.write(struct.pack(f'<{len(values)}{code}', *values))

print(path)
