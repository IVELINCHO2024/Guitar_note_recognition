import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset_maker import MelspectrogramDataset
from CNN_model import NoteCNN

def main():
    dataset = MelspectrogramDataset('mel_tensors')
    print(f"Dataset size: {len(dataset)}")
    print(f"Labels: {dataset.labels}")

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    num_classes = len(dataset.labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NoteCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0

        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X.float())
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (output.argmax(dim=1) == y).sum().item()

        accuracy = correct / len(dataset)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")

    torch.save(model.state_dict(), 'note_model.pth')
    torch.save(dataset.label2idx, 'label_mapping.pth')
    print("Training finished and model saved.")

if __name__ == "__main__":
    main()
