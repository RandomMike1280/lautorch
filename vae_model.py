import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyVAE(nn.Module):
    """
    A compact convolutional Variational Autoencoder (VAE) for 28x28 binarized
    MNIST images. The exported decoder has far fewer weights than the previous
    fully connected decoder while keeping decode(z) as a flat 784-pixel output.
    """
    def __init__(self, latent_dim=8, lora_rank=0, lora_alpha=1.0):
        super(TinyVAE, self).__init__()
        self.latent_dim = latent_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha

        # Encoder: 1x28x28 -> 8x14x14 -> 16x7x7 -> latent stats
        self.enc_conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
        self.enc_conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.fc_mu = nn.Linear(16 * 7 * 7, latent_dim)
        self.fc_logvar = nn.Linear(16 * 7 * 7, latent_dim)

        # Decoder: latent -> 8x7x7 -> pixel-shuffle blocks -> 1x28x28
        self.dec_fc = nn.Linear(latent_dim, 8 * 7 * 7)
        self.dec_fc_mask_logits = nn.Parameter(torch.full_like(self.dec_fc.weight, 2.0))
        if lora_rank > 0:
            self.dec_fc_lora_a = nn.Parameter(torch.empty(lora_rank, latent_dim))
            self.dec_fc_lora_b = nn.Parameter(torch.zeros(8 * 7 * 7, lora_rank))
            nn.init.kaiming_uniform_(self.dec_fc_lora_a, a=5 ** 0.5)
        else:
            self.register_parameter("dec_fc_lora_a", None)
            self.register_parameter("dec_fc_lora_b", None)
        self.mask_temperature = 1.0
        self.use_hard_mask = False
        self.dec_up1_dw = nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8)
        self.dec_up1_pw = nn.Conv2d(8, 16, kernel_size=1)
        self.dec_up2_dw = nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4)
        self.dec_up2_pw = nn.Conv2d(4, 4, kernel_size=1)

    def dec_fc_lora_delta(self):
        if self.lora_rank <= 0:
            return 0
        return (self.lora_alpha / self.lora_rank) * (self.dec_fc_lora_b @ self.dec_fc_lora_a)

    def effective_dec_fc_weight(self):
        return self.dec_fc.weight + self.dec_fc_lora_delta()

    def fc_mask(self):
        if self.use_hard_mask:
            return (self.dec_fc_mask_logits > 0).float()
        return torch.sigmoid(self.dec_fc_mask_logits / self.mask_temperature)

    def fc_mask_sum(self):
        return self.fc_mask().sum()

    def fc_active_count(self):
        return int((self.dec_fc_mask_logits > 0).sum().item())
        
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
        h = F.relu(F.linear(z, self.effective_dec_fc_weight() * self.fc_mask(), self.dec_fc.bias))
        h = h.view(-1, 8, 7, 7)
        h = F.pixel_shuffle(self.dec_up1_pw(F.relu(self.dec_up1_dw(h))), 2)
        h = F.relu(h)
        h = F.pixel_shuffle(self.dec_up2_pw(F.relu(self.dec_up2_dw(h))), 2)
        # Use sigmoid output to represent Bernoulli probability distribution per pixel.
        return torch.sigmoid(h).view(-1, 784)
        
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class TokenEmbeddingModel(nn.Module):
    """
    A separate embedding model that maps integer tokens 0..9 into the latent space.
    """
    def __init__(self, num_tokens=10, latent_dim=8):
        super(TokenEmbeddingModel, self).__init__()
        # 10 indices for MNIST digits 0..9.
        self.embedding = nn.Embedding(num_tokens, latent_dim)
        
    def forward(self, x):
        # x is a tensor of token IDs (e.g. [1, 2, 9, ...])
        return self.embedding(x)
