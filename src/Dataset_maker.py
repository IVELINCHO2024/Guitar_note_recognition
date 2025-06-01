import os
import torch
from torch.utils.data import Dataset

class MelspectrogramDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filepaths = []
        self.labels = []

        # Recursively walk through all folders under root_dir
        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.pt'):
                    full_path = os.path.join(dirpath, file)
                    self.filepaths.append(full_path)
                    # label = parent folder name
                    label = os.path.basename(dirpath)
                    self.labels.append(label)

        self.labels = sorted(list(set(self.labels)))
        self.label2idx = {label: i for i, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        mel_tensor = torch.load(filepath)

        label_str = os.path.basename(os.path.dirname(filepath))
        label = self.label2idx[label_str]

        return mel_tensor, label

    def get_list(self):
        return self.filepaths


