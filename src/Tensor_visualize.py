import torch
import matplotlib.pyplot as plt

tensor = torch.load('mel_tensors/D1_01.pt')

mel_spectrogram = tensor.squeeze(0).numpy()

plt.figure(figsize = (10,4))
plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='magma')
plt.title('Tensor testing')
plt.xlabel('Time frames')
plt.ylabel('Mel bands')

plt.tight_layout()
plt.show()