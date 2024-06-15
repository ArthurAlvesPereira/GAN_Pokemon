import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
imgsize = 56
noise_dim = 100
batch_size = 16
DATA_DIR = 'Datasets\Elements'  # Substitua pelo caminho correto do seu dataset

# Transformação dos dados
fashiontransform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Carregamento do dataset
fashiondata = datasets.ImageFolder(root=DATA_DIR, transform=fashiontransform)
dataloader = DataLoader(fashiondata, batch_size=batch_size, shuffle=True)

# Funções de conversão
def images_to_vectors(images):
    print(images.size())
    return images.view(images.size(0), 3 * imgsize * imgsize)

def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 3, imgsize, imgsize)

def noise(size, dim=noise_dim):
    return torch.randn(size, dim).to(device)

def log_images(test_images, savepath=None):
    figure = plt.figure(figsize=(8, 8))
    figure.subplots_adjust(wspace=-0.08, hspace=0.01)
    rows, cols = len(test_images) // 4, 4
    for i, img in enumerate(test_images):
        figure.add_subplot(rows, cols, i + 1)
        plt.axis("off")
        plt.imshow(img.permute(1, 2, 0).cpu().numpy())  # Ajuste para imagens RGB

    if savepath is not None:
        figure.savefig(savepath)
    plt.show()

# Definição do Discriminador
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 1)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = x.view(x.size(0), -1)  # Achatar o tensor
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = torch.sigmoid(self.fc2(x))
        return x

# Definição do Gerador
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 512 * 4 * 4)
        self.conv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.conv2 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.conv3 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.conv4 = nn.ConvTranspose2d(64, 3, 4, 2, 1)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(x.size(0), 512, 4, 4)  # Redimensionar para 4x4x512
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        return x

# Inicializando o modelo e otimizadores
discriminator = Discriminator().to(device)
generator = Generator().to(device)
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)
loss = nn.BCELoss()

# Funções de treino
def train_discriminator(optimizer, real_data, fake_data):
    optimizer.zero_grad()
    
    pred_real = discriminator(real_data)
    error_real = loss(pred_real, torch.ones_like(pred_real))
    
    pred_fake = discriminator(fake_data)
    error_fake = loss(pred_fake, torch.zeros_like(pred_fake))
    
    d_error = error_real + error_fake
    d_error.backward()
    optimizer.step()
    
    return d_error, pred_real, pred_fake

def train_generator(optimizer, fake_data):
    optimizer.zero_grad()
    
    pred_fake = discriminator(fake_data)
    g_error = loss(pred_fake, torch.ones_like(pred_fake))
    
    g_error.backward()
    optimizer.step()
    
    return g_error

# Exemplo de treino
num_epochs = 100

for epoch in range(num_epochs):
    for batch, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        
        # Treinamento do Discriminador
        fake_data = generator(noise(batch_size)).detach()
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_images, fake_data)
        
        # Treinamento do Gerador
        fake_data = generator(noise(batch_size))
        g_error = train_generator(g_optimizer, fake_data)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, D Error: {d_error.item()}, G Error: {g_error.item()}')
        test_images = vectors_to_images(fake_data)
        log_images(test_images)
