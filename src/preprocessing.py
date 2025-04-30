import librosa
import numpy as np

import librosa
import numpy as np

def audio_to_spectrogram(audio, sr, n_mels=128):
    S = librosa.feature.spectrogram(y=audio, sr=sr, n_mels=n_mels)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB