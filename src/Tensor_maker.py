import numpy as np
import librosa
import torch
import os

def mel_spectrogram(filename):
    y, sr = librosa.load(filename, sr=22050)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_db = librosa.power_to_db(S, ref=np.max)
    return torch.tensor(S_db).unsqueeze(0)

for root, dirs, files in os.walk(input_dir):


    



