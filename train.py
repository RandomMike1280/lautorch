import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from vae_model import TinyVAE, TokenEmbeddingModel

def binarize_mnist(x):
    """ Binarize image values to exactly 0.0 or 1.0 """
    return (x >= 0.5).float()

def loss_function(recon_x, x, mu, logvar):
    """ Standard VAE ELBO Loss (BCE + KLD) """
    # Sum over all pixels (784) and batch
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    
    # KL Divergence: 0.5 * sum(1 + logvar - mu^2 - e^logvar)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return BCE + KLD, BCE, KLD

def train(epochs=10, batch_size=128, lr=1e-3, latent_dim=4):
    # Set seed for reproducibility
    torch.manual_seed(42)
    
    # Load and preprocess MNIST dataset (Binarized)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(binarize_mnist)
    ])
    
    print("Loading binarized MNIST dataset...")
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize convolutional VAE and Token Embedding models
    vae = TinyVAE(latent_dim=latent_dim)
    token_embedding = TokenEmbeddingModel(num_tokens=10, latent_dim=latent_dim)
    
    # Optimizers
    optimizer = optim.Adam(list(vae.parameters()) + list(token_embedding.parameters()), lr=lr)
    
    decoder_params = (
        sum(p.numel() for p in vae.dec_fc.parameters()) +
        sum(p.numel() for p in vae.dec_conv1.parameters()) +
        sum(p.numel() for p in vae.dec_conv2.parameters())
    )
    print(f"Conv TinyVAE Parameters: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"Exported Decoder Parameters: {decoder_params:,}")
    print(f"TokenEmbedding Parameters: {sum(p.numel() for p in token_embedding.parameters()):,}")
    print("Starting joint training on CPU...")
    
    for epoch in range(1, epochs + 1):
        vae.train()
        token_embedding.train()
        train_loss = 0.0
        train_vae_bce = 0.0
        train_vae_kld = 0.0
        train_embed_loss = 0.0
        train_align_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Flatten image data
            data_flat = data.view(-1, 784)
            
            optimizer.zero_grad()
            
            # 1. Forward pass VAE
            recon_batch, mu, logvar = vae(data_flat)
            vae_loss, bce, kld = loss_function(recon_batch, data_flat, mu, logvar)
            
            # 2. Token Embedding Loss for digits 1..9
            # Filter tokens to strictly be between 1 and 9 (inclusive)
            embed_mask = (targets >= 1) & (targets <= 9)
            if embed_mask.any():
                active_targets = targets[embed_mask]
                active_data = data_flat[embed_mask]
                active_mu = mu[embed_mask]
                
                # Fetch latent code from token embedding
                z_embed = token_embedding(active_targets)
                
                # Decode the latent code
                recon_embed = vae.decode(z_embed)
                
                # Embedding reconstruction loss (against corresponding binarized hand-written digit)
                embed_recon = F.binary_cross_entropy(recon_embed, active_data, reduction='sum')
                
                # Alignment loss: Pull embedding close to the mean of VAE encoder's outputs
                # We detach active_mu to align the embedding with VAE latent space without destabilizing VAE training
                embed_align = F.mse_loss(z_embed, active_mu.detach(), reduction='sum')
                
                # Total embedding loss component
                embed_loss = embed_recon + 10.0 * embed_align
            else:
                embed_loss = torch.tensor(0.0)
                embed_recon = torch.tensor(0.0)
                embed_align = torch.tensor(0.0)
                
            # Joint objective
            total_loss = vae_loss + embed_loss
            total_loss.backward()
            
            optimizer.step()
            
            # Accumulate metrics
            train_loss += total_loss.item()
            train_vae_bce += bce.item()
            train_vae_kld += kld.item()
            train_embed_loss += embed_recon.item()
            train_align_loss += embed_align.item()
            
        # Average epoch metrics
        num_samples = len(train_loader.dataset)
        num_embed_samples = sum((train_dataset.targets >= 1) & (train_dataset.targets <= 9)).item()
        
        print(f"Epoch {epoch:02d} | "
              f"VAE BCE: {train_vae_bce/num_samples:.2f} | "
              f"KLD: {train_vae_kld/num_samples:.2f} | "
              f"Embed Recon: {train_embed_loss/num_embed_samples:.2f} | "
              f"Embed Align: {train_align_loss/num_embed_samples:.4f}")
              
    # Save trained model weights
    checkpoint = {
        'vae_state_dict': vae.state_dict(),
        'embed_state_dict': token_embedding.state_dict(),
        'latent_dim': latent_dim,
        'architecture': 'conv_tiny_vae_v1'
    }
    torch.save(checkpoint, 'vae_and_embed.pth')
    print("Training finished successfully. Saved model weights to 'vae_and_embed.pth'.")

    decoder_checkpoint = {
        'decoder_state_dict': {
            'dec_fc.weight': vae.state_dict()['dec_fc.weight'],
            'dec_fc.bias': vae.state_dict()['dec_fc.bias'],
            'dec_conv1.weight': vae.state_dict()['dec_conv1.weight'],
            'dec_conv1.bias': vae.state_dict()['dec_conv1.bias'],
            'dec_conv2.weight': vae.state_dict()['dec_conv2.weight'],
            'dec_conv2.bias': vae.state_dict()['dec_conv2.bias'],
        },
        'embed_state_dict': token_embedding.state_dict(),
        'latent_dim': latent_dim,
        'architecture': 'conv_tiny_vae_v1'
    }
    torch.save(decoder_checkpoint, 'decoder_and_embed.pth')
    print("Saved exported decoder weights to 'decoder_and_embed.pth'.")

if __name__ == '__main__':
    train(epochs=10, latent_dim=4)
