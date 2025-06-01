import torch
import torch.nn.functional as F
from CNN_model import NoteCNN
import librosa
import numpy as np
import sys

def wav_to_mel_tensor(wav_path):
    y, sr = librosa.load(wav_path, sr=22050)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=256)
    S_db = librosa.power_to_db(S, ref=np.max)
    mel_tensor = torch.tensor(S_db).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, time)
    return mel_tensor.float()

def predict_note(model_path, label_mapping_path, wav_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    label2idx = torch.load(label_mapping_path)
    idx2label = {v: k for k, v in label2idx.items()}

    num_classes = len(label2idx)
    model = NoteCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mel_tensor = wav_to_mel_tensor(wav_path).to(device)

    with torch.no_grad():
        output = model(mel_tensor)
        pred_idx = output.argmax(dim=1).item()
        predicted_note = idx2label[pred_idx]

    return predicted_note

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_note.py path/to/audio.wav")
        sys.exit(1)

    wav_path = sys.argv[1]
    model_path = 'note_model.pth'
    label_mapping_path = 'label_mapping.pth'

    note = predict_note(model_path, label_mapping_path, wav_path)
    print(f"Predicted note: {note}")
