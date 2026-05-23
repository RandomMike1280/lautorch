import os
import sys
import torch
from vae_model import TinyVAE, TokenEmbeddingModel

DEFAULT_NOISE_SCALE = 0.25

def print_ascii_digit(image):
    """
    Print 28x28 binarized image to the console using ASCII characters.
    1s are rendered as '█' and 0s are rendered as ' '.
    """
    image = image.view(28, 28)
    for r in range(28):
        row_str = ""
        for c in range(28):
            pixel = image[r, c].item()
            row_str += "##" if pixel >= 0.5 else "  "
        print(row_str)

def main():
    if len(sys.argv) < 2:
        print("Usage: python generate.py [token_id] (where token_id is between 0 and 9)")
        print("Example: python generate.py 5")
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
        
    checkpoint_path = 'vae_and_embed.pth'
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found. Please run 'train.py' first.")
        sys.exit(1)
        
    # Load model checkpoint
    print(f"Loading checkpoint from '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    latent_dim = checkpoint.get('latent_dim', 8)
    vae = TinyVAE(latent_dim=latent_dim)
    token_embedding = TokenEmbeddingModel(num_tokens=10, latent_dim=latent_dim)
    
    vae.load_state_dict(checkpoint['vae_state_dict'])
    token_embedding.load_state_dict(checkpoint['embed_state_dict'])
    vae.use_hard_mask = True
    
    vae.eval()
    token_embedding.eval()
    
    with torch.no_grad():
        # Get embedding vector for the requested token (digit)
        token_tensor = torch.tensor([token_id], dtype=torch.long)
        z = token_embedding(token_tensor)
        if noise_scale > 0:
            z = z + torch.randn_like(z) * noise_scale
        
        # Decode the embedding vector
        decoded_output = vae.decode(z)
        
        # Quantize decoded output strictly to 1s and 0s (threshold at 0.5)
        binarized_output = (decoded_output >= 0.5).float()
        
    print(f"\nSuccessfully generated hand-written digit for token '{token_id}' (noise={noise_scale}):\n")
    print_ascii_digit(binarized_output)
    print("\n" + "="*56 + "\n")
    
    # Optional: Save image file using PIL if PIL is installed
    try:
        from PIL import Image
        import numpy as np
        
        # Scale to 0..255
        img_np = (binarized_output.view(28, 28).numpy() * 255).astype(np.uint8)
        img = Image.fromarray(img_np)
        
        filename = f"generated_digit_{token_id}.png"
        img.save(filename)
        print(f"Saved generated image to '{filename}'")
    except ImportError:
        print("Note: PIL (Pillow) is not installed. Skipping saving image file. (Only ASCII art displayed)")

if __name__ == '__main__':
    main()
