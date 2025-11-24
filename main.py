import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq

fs, audio = wavfile.read("test_audio.wav")

# describe audio file
print("Sample rate:", fs)
print("Shape:", audio.shape)

# make it monotone
if audio.ndim == 2:
    audio = audio.mean(axis=1)

# normalize audio
audio = audio / np.max(np.abs(audio))

# plot waveform
t = np.arange(len(audio)) / fs
plt.figure(figsize=(12, 4))
plt.plot(t, audio)
plt.title("Waveform (Time Domain)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.show()

# plot frequency graph
N = len(audio)
Y = fft(audio)
freqs = fftfreq(N, 1/fs)
plt.figure(figsize=(12,4))
plt.plot(freqs[:N//2], np.abs(Y[:N//2]))  # only positive frequencies
plt.title("Frequency Spectrum (Magnitude)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.show()