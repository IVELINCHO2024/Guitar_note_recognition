import torch.nn as nn
import torch.nn.functional as F

class NoteCNN(nn.Module):
    def __init__(self, num_classes):
        super(NoteCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)  # output: [16, 254, 85]
        self.pool = nn.MaxPool2d(2, 2)                 # output: [16, 127, 42]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3) # output: [32, 125, 40]
        # another pool                                # output: [32, 62, 20]
        self.dropout = nn.Dropout(0.3)

        # Flattened size calculation: 32 channels * 62 height * 20 width
        self.fc1 = nn.Linear(32 * 62 * 20, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # conv1 + pool
        x = self.pool(F.relu(self.conv2(x)))  # conv2 + pool
        x = self.dropout(x)
        x = x.view(x.size(0), -1)             # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
