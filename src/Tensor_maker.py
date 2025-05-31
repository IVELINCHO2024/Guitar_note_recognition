import numpy as np
import librosa
import torch
import os

input_root = 'guitar_wav'
output_root = 'mel_tensors'


def mel_spectrogram_tensor(input_root):
    y, sr = librosa.load(input_root, sr=22050)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_db = librosa.power_to_db(S, ref=np.max)
    return torch.tensor(S_db).unsqueeze(0)



for root, dirs, files in os.walk(input_root):
    print(f"Looking in: {root}")
    for file in files:
        print("Found file:", file)
        if file.lower().strip().endswith(".wav"):
            input_path = os.path.join(root, file)
            
            note = os.path.basename(root)
            
            note_folder = os.path.join(output_root, note)
            os.makedirs(note_folder, exist_ok=True)
            
            try:
                mel_tensor = mel_spectrogram_tensor(input_path)
                save_path = os.path.join(note_folder, os.path.splitext(file)[0] + '.pt')
                torch.save(mel_tensor, save_path)
                print(f"Saved: {save_path}")
            except Exception as e:
                print(f"Error processing {input_path}: {e}")



    



