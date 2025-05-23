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
import torch

def mel_spectrogram(filename, note):
    y, sr = librosa.load(filename, sr=22050)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_db = librosa.power_to_db(S, ref=np.max)

    mel_tensor = torch.tensor(S_db).unsqueeze(0)

    torch.save(mel_tensor, f'mel_tensors/{note}.pt')

