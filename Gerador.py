import torch
import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.autograd.variable import Variable
import pickle
from IPython import display

noise_dim = 100
imgsize = 56
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Generator(nn.Module):
  def __init__(self, n_in, n_out):
    super().__init__()

    self.layers = nn.Sequential(
        nn.Linear(n_in, 256),
        nn.LeakyReLU(0.2),
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 1024),
        nn.LeakyReLU(0.2),
        nn.Linear(1024, n_out),
        nn.Tanh()

    )

  def forward(self, z):
    return self.layers(z)

gerador = Generator(noise_dim, 3* imgsize * imgsize).to(device)
gerador.load_state_dict(torch.load("gerador_medio.pth"))
def images_to_vectors(images):
    return images.view(images.size(0), -1)

def vectors_to_images(vectors, nc=3):
    return vectors.view(vectors.size(0), nc, imgsize, imgsize)

def noise(size, dim=noise_dim):
    return torch.randn(size, dim).to(device)

def log_images(test_images, savepath=None):
    figure = plt.figure(figsize=(8, 8))
    figure.subplots_adjust(wspace=-0.08, hspace=0.01)
    rows, cols = len(test_images) // 4, 4
    for i, img in enumerate(test_images):
        figure.add_subplot(rows, cols, i + 1)
        plt.axis("off")
        # Normalizar os valores das imagens para o intervalo [0, 1]
        img = (img + 1) / 2
        # Transpor os eixos de [channels, height, width] para [height, width, channels]
        plt.imshow(img.transpose(1, 2, 0))
    
    if savepath is not None:
        figure.savefig(savepath)
    plt.show()
    
test_noise = noise(16,noise_dim)
test_images = vectors_to_images(gerador(test_noise)).cpu().detach().numpy() 
log_images(test_images)