import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
BATCH_SIZE = 64
EPOCHS = 50
# Size of the mask applied to input images during preprocessing
MASK_SIZE = 2

transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)

def apply_mask(images, mask_size=MASK_SIZE):
    images = images.clone()
    _, _, h, w = images.shape
    start = h // 2 - mask_size // 2
    end = start + mask_size
    images[:, :, start:end, start:end] = 0
    return images

class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))
        self.v_bias = nn.Parameter(torch.zeros(n_visible))

    def sample_h(self, v):
        prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        return prob, torch.bernoulli(prob)

    def sample_v(self, h):
        prob = torch.sigmoid(torch.matmul(h, self.W) + self.v_bias)
        return prob, torch.bernoulli(prob)

    def contrastive_divergence(self, v, k=1):
        v = v.view(v.size(0), -1)
        v0 = v
        for _ in range(k):
            ph, h = self.sample_h(v)
            pv, v = self.sample_v(h)
        return v0, pv

    def forward(self, v):
        ph, h = self.sample_h(v.view(v.size(0), -1))
        return h

    def reconstruct(self, v, k=1):
        v = v.view(v.size(0), -1)
        for _ in range(k):
            prob_h, h = self.sample_h(v)
            prob_v, v = self.sample_v(h)
        return prob_v.view(-1, 1, 28, 28)


def train_rbm(rbm, data_loader, epochs):
    optimizer = torch.optim.SGD(rbm.parameters(), lr=0.1)
    for epoch in range(epochs):
        total_loss = 0
        for images, _ in tqdm(data_loader):
            images = apply_mask(images).to(device)
            images = images.view(images.size(0), -1)
            v0, v1 = rbm.contrastive_divergence(images)
            loss = torch.mean((v0 - v1)**2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader):.4f}")


rbm = RBM(n_visible=28*28, n_hidden=512).to(device)
train_rbm(rbm, trainloader, EPOCHS)

dataiter = iter(trainloader)
images, labels = next(dataiter)
masked_images = apply_mask(images)

original = images.to(device)
masked = masked_images.to(device)
reconstructed = rbm.reconstruct(masked, k=1)

def show_images(original, masked, reconstructed, n=6):
    plt.figure(figsize=(12, 6))
    for i in range(n):
        # Original
        plt.subplot(3, n, i + 1)
        plt.imshow(original[i].cpu().squeeze(), cmap='gray')
        plt.axis('off')
        # Masked
        plt.subplot(3, n, i + 1 + n)
        plt.imshow(masked[i].cpu().squeeze(), cmap='gray')
        plt.axis('off')
        # Reconstructed
        plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(reconstructed[i].detach().cpu().squeeze(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

show_images(original, masked, reconstructed)
