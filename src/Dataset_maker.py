import os
import torch
from torch.utils.data import Dataset

class MelspectrogramDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filepaths = []
        self.labels = []

        for dirpath, _, filenames in os.walk(root_dir):
            for file in filenames:
                if file.endswith('.pt'):
                    full_path = os.path.join(dirpath, file)
                    self.filepaths.append(full_path)
                    label = os.path.basename(dirpath)
                    self.labels.append(label)

        self.labels = sorted(list(set(self.labels)))
        self.label2idx = {label: i for i, label in enumerate(self.labels)}