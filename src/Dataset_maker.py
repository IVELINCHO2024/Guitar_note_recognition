import os
import torch
from torch.utils.data import Dataset

class MelspectrogramDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.filepaths = []

        for file in os.listdir(root_dir):
            if file.endswith('.pt'):
                self.filepaths.append(os.path.join(root_dir, file))
        
        self.labels = sorted(list(set(f.split('_')[0] for f in os.listdir(root_dir))))
        self.label2idx = {label: i for i, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        mel_tensor = torch.load(filepath)

        filename = os.path.basename(filepath)
        label_str = filename.split('_')[0]
        label = self.label2idx[label_str]

        return mel_tensor, label
    
    def get_list(self):
        return self.filepaths
    
dataset = MelspectrogramDataset('mel_tensors')
print(dataset.get_list())