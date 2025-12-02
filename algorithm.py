import numpy as np
from scipy.io import wavfile

def iir(x, b, a):
    print("iir")
    x_len = len(x)
    y = np.zeros(x_len)
    
    # normalize
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]

    for n in range(x_len):
        for k in range(len(b)):
            if n - k >= 0:
                y[n] += b[k] * x[n-k]
                # print(y[n])

        for k in range(1, (len(a))):
            if n - k >= 0:
                y[n] -= a[k] * y[n-k]
                # print(y[n])

    return y