import numpy as np
from scipy.io import wavfile
from scipy.signal import butter
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq
import algorithm

# change variable for cutoff
CUTOFF = 500

# get audio
fs, audio = wavfile.read("test_audio.wav")

# monotone
if audio.ndim == 2:
    audio = audio.mean(axis=1)

# normalize audio
audio = audio / np.max(np.abs(audio))

# get Butterworth coefficients
b, a = butter(4, CUTOFF, btype="low", fs=fs)

# create modified audio
audio2 = algorithm.iir(audio, b, a)

# plot original waveform
t = np.arange(len(audio)) / fs
plt.figure(figsize=(12, 4))
plt.plot(t, audio)
plt.title("Original Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# plot modified waveform
t = np.arange(len(audio2)) / fs
plt.figure(figsize=(12, 4))
plt.plot(t, audio2)
plt.title("Modified Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# plot original frequency graph
N = len(audio)
Y = fft(audio)
freqs = fftfreq(N, 1/fs)
plt.figure(figsize=(12,4))
plt.plot(freqs[:N//2], np.abs(Y[:N//2]))
plt.title("Original Frequency Graph")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()

# plot modified frequency graph
N = len(audio2)
Y = fft(audio2)
freqs = fftfreq(N, 1/fs)
plt.figure(figsize=(12,4))
plt.plot(freqs[:N//2], np.abs(Y[:N//2]))
plt.title("Modified Frequency Graph")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()