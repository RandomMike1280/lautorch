import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_NOISE_SCALE = 0.25

class TokenEmbeddingModel(nn.Module):
    """ Maps integer tokens 0..9 into the latent space. """
    def __init__(self, num_tokens=10, latent_dim=8):
        super(TokenEmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_tokens, latent_dim)
        
    def forward(self, x):
        return self.embedding(x)


class TinyDecoder(nn.Module):
    """
    Self-contained convolutional tiny decoder.
    Takes an 8-dimensional latent vector and reconstructs a 28x28 binarized image.
    Has fewer than 4,000 parameters at the default latent size.
    """
    def __init__(self, latent_dim=8):
        super(TinyDecoder, self).__init__()
        self.dec_fc = nn.Linear(latent_dim, 8 * 7 * 7)
        self.dec_fc_mask_logits = nn.Parameter(torch.empty(8 * 7 * 7, latent_dim))
        self.mask_temperature = 1.0
        self.use_hard_mask = True
        self.dec_conv1 = nn.Conv2d(8, 4, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv2d(4, 1, kernel_size=3, padding=1)

    def fc_mask(self):
        if self.use_hard_mask:
            return (self.dec_fc_mask_logits > 0).float()
        return torch.sigmoid(self.dec_fc_mask_logits / self.mask_temperature)
        
    def forward(self, z):
        h = F.relu(F.linear(z, self.dec_fc.weight * self.fc_mask(), self.dec_fc.bias))
        h = h.view(-1, 8, 7, 7)
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        h = F.relu(self.dec_conv1(h))
        h = F.interpolate(h, scale_factor=2, mode='nearest')
        return torch.sigmoid(self.dec_conv2(h)).view(-1, 784)


def print_ascii_digit(image):
    """ Print 28x28 binarized image to the console using ASCII-safe characters. """
    image = image.view(28, 28)
    for r in range(28):
        row_str = ""
        for c in range(28):
            pixel = image[r, c].item()
            row_str += "##" if pixel >= 0.5 else "  "
        print(row_str)


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_minimal.py [token_id] (where token_id is between 0 and 9)")
        sys.exit(1)
        
    try:
        token_id = int(sys.argv[1])
    except ValueError:
        print("Error: token_id must be an integer between 0 and 9.")
        sys.exit(1)
        
    if token_id < 0 or token_id > 9:
        print("Error: token_id must be between 0 and 9.")
        sys.exit(1)

    noise_scale = DEFAULT_NOISE_SCALE
    if len(sys.argv) >= 3:
        try:
            noise_scale = float(sys.argv[2])
        except ValueError:
            print("Error: optional noise_scale must be a number.")
            sys.exit(1)
        
    checkpoint_path = 'decoder_and_embed.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found. Please run creation first.")
        sys.exit(1)
        
    print(f"Loading minimal checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    latent_dim = checkpoint.get('latent_dim', 8)
    decoder = TinyDecoder(latent_dim=latent_dim)
    token_embedding = TokenEmbeddingModel(num_tokens=10, latent_dim=latent_dim)
    
    # Load state dicts directly
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    token_embedding.load_state_dict(checkpoint['embed_state_dict'])
    
    decoder.eval()
    token_embedding.eval()
    
    with torch.no_grad():
        token_tensor = torch.tensor([token_id], dtype=torch.long)
        z = token_embedding(token_tensor)
        if noise_scale > 0:
            z = z + torch.randn_like(z) * noise_scale
        decoded_output = decoder(z)
        binarized_output = (decoded_output >= 0.5).float()
        
    print(f"\nSuccessfully generated hand-written digit for token '{token_id}' using ONLY decoder/embedding (noise={noise_scale}):\n")
    print_ascii_digit(binarized_output)
    print("\n" + "="*56 + "\n")

if __name__ == '__main__':
    main()
