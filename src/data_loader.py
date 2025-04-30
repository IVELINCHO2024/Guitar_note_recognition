import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from glob import glob
import librosa
import librosa.display
import sounddevice as sd
import soundfile as sf
from itertools import cycle

def load_audio(file_path, sample_rate=22050):
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio, sr

y, sr = librosa.load('data/mixkit-instrument-echo-swell-2673.wav', sr=22050)


def waveform(y,sr):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(y, sr=sr, color='blue') 
    plt.title('Audio Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D),ref=np.max)
S_db.shape

fig,ax = plt.subplots(figsize=(10,5))
img = librosa.display.specshow(S_db, x_axis='time', y_axis ='log',ax=ax)

plt.colorbar(img, ax=ax)
plt.title('Spectrogram')
plt.show()