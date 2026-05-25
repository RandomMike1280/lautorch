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

def laplacian_loss(recon_x, x):
    """Match local edge/curvature structure to discourage blurry shared shapes."""
    kernel = torch.tensor(
        [[0.0, 1.0, 0.0],
         [1.0, -4.0, 1.0],
         [0.0, 1.0, 0.0]],
        device=recon_x.device,
    ).view(1, 1, 3, 3)
    recon_edges = F.conv2d(recon_x.view(-1, 1, 28, 28), kernel, padding=1)
    target_edges = F.conv2d(x.view(-1, 1, 28, 28), kernel, padding=1)
    return F.l1_loss(recon_edges, target_edges, reduction='sum')

def sobel_loss(recon_x, x):
    """Match stroke direction gradients as a lightweight perceptual proxy."""
    kernels = torch.tensor(
        [[[-1.0, 0.0, 1.0],
          [-2.0, 0.0, 2.0],
          [-1.0, 0.0, 1.0]],
         [[-1.0, -2.0, -1.0],
          [0.0, 0.0, 0.0],
          [1.0, 2.0, 1.0]]],
        device=recon_x.device,
    ).view(2, 1, 3, 3)
    recon_edges = F.conv2d(recon_x.view(-1, 1, 28, 28), kernels, padding=1)
    target_edges = F.conv2d(x.view(-1, 1, 28, 28), kernels, padding=1)
    return F.l1_loss(recon_edges, target_edges, reduction='sum')

def structure_loss(recon_x, x):
    return laplacian_loss(recon_x, x) + 0.5 * sobel_loss(recon_x, x)

def train(epochs=10, batch_size=128, lr=1e-3, latent_dim=4,
          sparsity_lambda=1.0, temp_start=1.0, temp_end=0.05,
          lora_rank=3, lora_alpha=2.0, laplacian_lambda=0.1,
          sparsity_target=1000.0, embed_align_weight=0.1):
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
    vae = TinyVAE(latent_dim=latent_dim, lora_rank=lora_rank, lora_alpha=lora_alpha)
    token_embedding = TokenEmbeddingModel(num_tokens=10, latent_dim=latent_dim)
    
    # Optimizers
    optimizer = optim.Adam(list(vae.parameters()) + list(token_embedding.parameters()), lr=lr)
    
    decoder_params = (
        sum(p.numel() for p in vae.dec_fc.parameters()) +
        sum(p.numel() for p in vae.dec_up1_dw.parameters()) +
        sum(p.numel() for p in vae.dec_up1_pw.parameters()) +
        sum(p.numel() for p in vae.dec_up2_dw.parameters()) +
        sum(p.numel() for p in vae.dec_up2_pw.parameters())
    )
    lora_params = sum(
        p.numel() for p in (vae.dec_fc_lora_a, vae.dec_fc_lora_b)
        if p is not None
    )
    print(f"Conv TinyVAE Parameters: {sum(p.numel() for p in vae.parameters()):,}")
    print(f"Exported Decoder Parameters: {decoder_params:,}")
    print(f"LoRA Decoder Parameters: {lora_params:,} (rank={lora_rank}, alpha={lora_alpha})")
    print(f"TokenEmbedding Parameters: {sum(p.numel() for p in token_embedding.parameters()):,}")
    print(f"Sparsity lambda: {sparsity_lambda}")
    print(f"Sparsity target: {sparsity_target:.0f}/1568")
    print(f"Structure lambda: {laplacian_lambda}")
    print(f"Embedding align weight: {embed_align_weight}")
    print("Starting joint training on CPU...")
    
    for epoch in range(1, epochs + 1):
        progress = (epoch - 1) / max(1, epochs - 1)
        vae.mask_temperature = temp_start * ((temp_end / temp_start) ** progress)
        vae.train()
        token_embedding.train()
        train_loss = 0.0
        train_vae_bce = 0.0
        train_vae_kld = 0.0
        train_embed_loss = 0.0
        train_align_loss = 0.0
        train_sparse_loss = 0.0
        train_lap_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Flatten image data
            data_flat = data.view(-1, 784)
            
            optimizer.zero_grad()
            
            # 1. Forward pass VAE
            recon_batch, mu, logvar = vae(data_flat)
            vae_loss, bce, kld = loss_function(recon_batch, data_flat, mu, logvar)
            vae_lap = structure_loss(recon_batch, data_flat)
            
            # 2. Token Embedding Loss for digits 0..9
            embed_mask = (targets >= 0) & (targets <= 9)
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
                embed_lap = structure_loss(recon_embed, active_data)
                
                # Alignment loss: Pull embedding close to the mean of VAE encoder's outputs
                # We detach active_mu to align the embedding with VAE latent space without destabilizing VAE training
                embed_align = F.mse_loss(z_embed, active_mu.detach(), reduction='sum')
                
                # Total embedding loss component
                embed_loss = embed_recon + embed_align_weight * embed_align + laplacian_lambda * embed_lap
            else:
                embed_loss = torch.tensor(0.0)
                embed_recon = torch.tensor(0.0)
                embed_align = torch.tensor(0.0)
                embed_lap = torch.tensor(0.0)
                
            sparse_loss = sparsity_lambda * F.relu(vae.fc_mask_sum() - sparsity_target).pow(2) / sparsity_target
            lap_loss = laplacian_lambda * vae_lap
            total_loss = vae_loss + embed_loss + sparse_loss + lap_loss
            total_loss.backward()
            
            optimizer.step()
            
            # Accumulate metrics
            train_loss += total_loss.item()
            train_vae_bce += bce.item()
            train_vae_kld += kld.item()
            train_embed_loss += embed_recon.item()
            train_align_loss += embed_align.item()
            train_sparse_loss += sparse_loss.item()
            train_lap_loss += (lap_loss + laplacian_lambda * embed_lap).item()
            
        # Average epoch metrics
        num_samples = len(train_loader.dataset)
        num_embed_samples = sum((train_dataset.targets >= 0) & (train_dataset.targets <= 9)).item()
        
        print(f"Epoch {epoch:02d} | "
              f"Temp: {vae.mask_temperature:.4f} | "
              f"VAE BCE: {train_vae_bce/num_samples:.2f} | "
              f"KLD: {train_vae_kld/num_samples:.2f} | "
              f"Embed Recon: {train_embed_loss/num_embed_samples:.2f} | "
              f"Embed Align: {train_align_loss/num_embed_samples:.4f} | "
              f"Struct: {train_lap_loss/num_samples:.2f} | "
              f"Mask Sum: {vae.fc_mask_sum().item():.1f} | "
              f"Active: {vae.fc_active_count()}/1568 | "
              f"Sparse Loss: {train_sparse_loss/len(train_loader):.1f}")
              
    # Save trained model weights
    checkpoint = {
        'vae_state_dict': vae.state_dict(),
        'embed_state_dict': token_embedding.state_dict(),
        'latent_dim': latent_dim,
        'architecture': 'conv_tiny_vae_sparse_lora_pixelshuffle_v3',
        'mask_temperature': vae.mask_temperature,
        'sparsity_lambda': sparsity_lambda,
        'sparsity_target': sparsity_target,
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'laplacian_lambda': laplacian_lambda,
        'embed_align_weight': embed_align_weight,
    }
    torch.save(checkpoint, 'vae_and_embed.pth')
    print("Training finished successfully. Saved model weights to 'vae_and_embed.pth'.")

    decoder_checkpoint = {
        'decoder_state_dict': {
            # Fold LoRA into the exported decoder so Lau and minimal inference
            # still consume the same sparse FC architecture.
            'dec_fc.weight': vae.effective_dec_fc_weight().detach(),
            'dec_fc.bias': vae.state_dict()['dec_fc.bias'],
            'dec_fc_mask_logits': vae.state_dict()['dec_fc_mask_logits'],
            'dec_up1_dw.weight': vae.state_dict()['dec_up1_dw.weight'],
            'dec_up1_dw.bias': vae.state_dict()['dec_up1_dw.bias'],
            'dec_up1_pw.weight': vae.state_dict()['dec_up1_pw.weight'],
            'dec_up1_pw.bias': vae.state_dict()['dec_up1_pw.bias'],
            'dec_up2_dw.weight': vae.state_dict()['dec_up2_dw.weight'],
            'dec_up2_dw.bias': vae.state_dict()['dec_up2_dw.bias'],
            'dec_up2_pw.weight': vae.state_dict()['dec_up2_pw.weight'],
            'dec_up2_pw.bias': vae.state_dict()['dec_up2_pw.bias'],
        },
        'embed_state_dict': token_embedding.state_dict(),
        'latent_dim': latent_dim,
        'architecture': 'conv_tiny_vae_sparse_lora_pixelshuffle_v3',
        'mask_temperature': vae.mask_temperature,
        'sparsity_lambda': sparsity_lambda,
        'sparsity_target': sparsity_target,
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'lora_folded': True,
        'laplacian_lambda': laplacian_lambda,
        'embed_align_weight': embed_align_weight,
    }
    torch.save(decoder_checkpoint, 'decoder_and_embed.pth')
    print("Saved exported decoder weights to 'decoder_and_embed.pth'.")

if __name__ == '__main__':
    train(epochs=30, latent_dim=4)
