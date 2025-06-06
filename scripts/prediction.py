import torch
import torch.nn.functional as F
import librosa
import numpy as np
from .CNN_model import NoteCNN
import os


def wav_to_mel_tensor_fixed(wav_path, target_time_frames=86):
    y, sr = librosa.load(wav_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_db = librosa.power_to_db(S, ref=np.max)

    if S_db.shape[1] > target_time_frames:
        S_db = S_db[:, :target_time_frames]
    elif S_db.shape[1] < target_time_frames:
        pad_width = target_time_frames - S_db.shape[1]
        S_db = np.pad(S_db, ((0, 0), (0, pad_width)), mode='constant')

    mel_tensor = torch.tensor(S_db).unsqueeze(0).unsqueeze(0) 
    return mel_tensor


def predict_note(model_path, label_mapping_path, wav_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label2idx = torch.load(label_mapping_path)
    idx2label = {v: k for k, v in label2idx.items()}

    num_classes = len(label2idx)
    model = NoteCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mel_tensor = wav_to_mel_tensor_fixed(wav_path).to(device).float()

    with torch.no_grad():
        output = model(mel_tensor)
        pred_idx = output.argmax(dim=1).item()
        predicted_note = idx2label[pred_idx]

    return predicted_note


if __name__ == "__main__":
    model_path = os.path.join('saved_models', 'note_model.pth')
    label_mapping_path = os.path.join('saved_models', 'label_mapping.pth')
    wav_path = input('Put your .wav file path here: ')

    note = predict_note(model_path, label_mapping_path, wav_path)
    print(f"Predicted note: {note}")
