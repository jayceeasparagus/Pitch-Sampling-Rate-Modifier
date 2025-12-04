import numpy as np
from scipy.io import wavfile

def iir(x, b, a):
    """Function IIR takes in coefficient lists b and a along with input signal x
    and returns output y
    
    output reflects the iir filter, or infinite impulse response filter"""

    # define
    x_len = len(x)
    y = np.zeros(x_len)

    # normalize coefficients
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]

    # replicate summation
    for n in range(x_len):
        # look at inputs
        for k in range(len(b)):
            if n - k >= 0:
                y[n] += b[k] * x[n-k]

        # look at outputs
        for k in range(1, (len(a))):
            if n - k >= 0:
                y[n] -= a[k] * y[n-k]

    return y