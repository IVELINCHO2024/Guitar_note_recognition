from Dataset_maker import MelspectrogramDataset
from torch.utils.data import DataLoader


dataset = MelspectrogramDataset('mel_tensors')

train_loader = DataLoader(dataset, batch_size= 16, shuffle = True)
