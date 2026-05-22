import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyVAE(nn.Module):
    """
    A compact convolutional Variational Autoencoder (VAE) for 28x28 binarized
    MNIST images. The exported decoder has far fewer weights than the previous
    fully connected decoder while keeping decode(z) as a flat 784-pixel output.
    """
    def __init__(self, latent_dim=8):
        super(TinyVAE, self).__init__()
        self.latent_dim = latent_dim

        # Encoder: 1x28x28 -> 8x14x14 -> 16x7x7 -> latent stats
        self.enc_conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(16 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(16 * 7 * 7, latent_dim)

        # Decoder: latent -> 8x7x7 -> 4x14x14 -> 1x28x28
        self.dec_fc = nn.Linear(latent_dim, 8 * 7 * 7)
        self.dec_conv1 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(4, 1, kernel_size=3, padding=1)
        
    def encode(self, x):
        # Accept either flat (batch, 784) inputs or image (batch, 1, 28, 28) inputs.
        h = x.view(-1, 1, 28, 28)
        h = F.relu(self.enc_conv1(h))
        h = F.relu(self.enc_conv2(h))
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        # z shape: (batch_size, latent_dim)
        h = F.relu(self.dec_fc(z))
        h = h.view(-1, 8, 7, 7)
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = F.relu(self.dec_conv1(h))
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        # Use sigmoid output to represent Bernoulli probability distribution per pixel.
        return torch.sigmoid(self.dec_conv2(h)).view(-1, 784)
        
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class TokenEmbeddingModel(nn.Module):
    """
    A separate embedding model that maps integer tokens 1..9 into the latent space.
    """
    def __init__(self, num_tokens=10, latent_dim=8):
        super(TokenEmbeddingModel, self).__init__()
        # 10 indices so we can use 1..9 directly (0 is unused)
        self.embedding = nn.Embedding(num_tokens, latent_dim)
        
    def forward(self, x):
        # x is a tensor of token IDs (e.g. [1, 2, 9, ...])
        return self.embedding(x)
