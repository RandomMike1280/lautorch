import base64
import torch

OUTPUT_FILES = (
    ("weights_data_1.laum", {"embed", "conv1_b", "conv2_b"}, {"SW": [0, 3, 4], "SM": [0, 1], "b_fc": [0, 1]}),
    ("weights_data_2.laum", {"conv1_w"}, {"SW": [1]}),
    ("weights_data_3.laum", {"conv2_w"}, {"SW": [2]}),
)
SPLIT_TENSORS = {"SW": 5, "SM": 2, "b_fc": 2}

def quantize_and_encode(tensor, name):
    """
    Quantizes a PyTorch tensor to 8-bit integers and encodes it as base64.
    Returns a dictionary of metadata and the encoded string.
    """
    flat_tensor = tensor.flatten()
    min_val = flat_tensor.min().item()
    max_val = flat_tensor.max().item()
    
    # Avoid division by zero if min and max are equal
    range_val = max_val - min_val if max_val != min_val else 1.0
    
    # 8-bit scale to 0..255
    quantized = ((flat_tensor - min_val) / range_val * 255.0).round().clamp(0, 255).int()
    byte_data = bytes(quantized.tolist())
    b64_str = base64.b64encode(byte_data).decode("ascii")
    
    # Get rows and cols for matrix consumers. Full dims are kept for conv kernels.
    shape = tensor.shape
    rows = shape[0] if len(shape) > 0 else 1
    cols = shape[1] if len(shape) > 1 else 1
    
    return {
        "name": name,
        "min": min_val,
        "max": max_val,
        "b64": b64_str,
        "rows": rows,
        "cols": cols,
        "dims": list(shape)
    }

def encode_raw_bytes(values, name, rows, cols):
    byte_data = bytes(values)
    return {
        "name": name,
        "min": 0.0,
        "max": 255.0,
        "b64": base64.b64encode(byte_data).decode("ascii"),
        "rows": rows,
        "cols": cols,
        "dims": [rows, cols],
    }

def get_decoder_state(checkpoint):
    if 'decoder_state_dict' in checkpoint:
        return checkpoint['decoder_state_dict']
    if 'vae_state_dict' in checkpoint:
        vae_state = checkpoint['vae_state_dict']
        return {
            'dec_fc.weight': vae_state['dec_fc.weight'],
            'dec_fc.bias': vae_state['dec_fc.bias'],
            'dec_fc_mask_logits': vae_state['dec_fc_mask_logits'],
            'dec_conv1.weight': vae_state['dec_conv1.weight'],
            'dec_conv1.bias': vae_state['dec_conv1.bias'],
            'dec_conv2.weight': vae_state['dec_conv2.weight'],
            'dec_conv2.bias': vae_state['dec_conv2.bias'],
        }
    raise KeyError("Checkpoint must contain 'decoder_state_dict' or 'vae_state_dict'.")

def split_b64(b64_str, parts):
    part_size = (len(b64_str) + parts - 1) // parts
    if part_size % 4:
        part_size += 4 - (part_size % 4)
    return [b64_str[i:i + part_size] for i in range(0, len(b64_str), part_size)]

def write_weight_entry(f, name, meta, b64_override=None, part_names=None):
    f.write(f"modelWeights[\"{name}\"] = {{\n")
    f.write(f"    [\"min\"] = {meta['min']:.8f},\n")
    f.write(f"    [\"max\"] = {meta['max']:.8f},\n")
    f.write(f"    [\"rows\"] = {meta['rows']},\n")
    f.write(f"    [\"cols\"] = {meta['cols']},\n")
    f.write("    [\"dims\"] = {" + ", ".join(str(dim) for dim in meta['dims']) + "}")
    if part_names:
        f.write(",\n")
        f.write("    [\"parts\"] = {" + ", ".join(f"\"{part}\"" for part in part_names) + "}\n")
    else:
        b64_str = meta['b64'] if b64_override is None else b64_override
        f.write(",\n")
        f.write(f"    [\"b64\"] = \"{b64_str}\"\n")
    f.write("}\n\n")

def write_b64_part(f, name, b64_str):
    f.write(f"modelWeights[\"{name}\"] = \"{b64_str}\"\n\n")

def write_weight_file(output_filename, tensors, split_entries, split_parts):
    print(f"Quantizing and writing Lau weight tables to '{output_filename}'...")

    with open(output_filename, "w") as f:
        f.write("-- Quantized weights data for Lau MNIST VAE\n")
        f.write("varol modelWeights = {}\n\n")

        for name, tensor in tensors:
            meta = tensor if isinstance(tensor, dict) else quantize_and_encode(tensor, name)
            write_weight_entry(f, name, meta)

        for name, part_indexes in split_entries.items():
            meta = split_parts[name]["meta"]
            part_names = split_parts[name]["part_names"]
            if isinstance(part_indexes, int):
                part_indexes = [part_indexes]
            for part_index in part_indexes:
                part_name = part_names[part_index]
                if part_index == 0:
                    write_weight_entry(f, name, meta, part_names=part_names)
                write_b64_part(f, part_name, split_parts[name]["b64_parts"][part_index])

        f.write("return modelWeights\n")

def main():
    checkpoint_path = 'decoder_and_embed.pth'
    print(f"Loading '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    decoder_state = get_decoder_state(checkpoint)
    
    embed_weight = checkpoint['embed_state_dict']['embedding.weight']
    dec_fc_weight = decoder_state['dec_fc.weight']
    dec_fc_bias = decoder_state['dec_fc.bias']
    dec_fc_mask_logits = decoder_state.get('dec_fc_mask_logits', torch.ones_like(dec_fc_weight))
    dec_conv1_weight = decoder_state['dec_conv1.weight']
    dec_conv1_bias = decoder_state['dec_conv1.bias']
    dec_conv2_weight = decoder_state['dec_conv2.weight']
    dec_conv2_bias = decoder_state['dec_conv2.bias']
    
    # Transpose weights to match Lau matrix multiplication convention (x * W_transposed)
    # so we do not need to transpose at runtime in Lau.
    # z (1 x latent_dim) * W_fc (latent_dim x 392) -> (1 x 392)
    dec_fc_weight_t = dec_fc_weight.t()
    hard_mask_t = (dec_fc_mask_logits.t() > 0)
    sparse_weights = []
    sparse_masks = []
    for j in range(dec_fc_weight_t.shape[1]):
        mask_byte = 0
        for i, bit in ((3, 8), (2, 4), (1, 2), (0, 1)):
            if hard_mask_t[i, j]:
                mask_byte += bit
                sparse_weights.append(dec_fc_weight_t[i, j].item())
        sparse_masks.append(mask_byte)

    sparse_weight_tensor = torch.tensor(sparse_weights, dtype=dec_fc_weight_t.dtype)
    print(f"Sparse FC active weights: {len(sparse_weights)}/{dec_fc_weight_t.numel()}")

    tensors = [
        ("embed", embed_weight),
        ("SW", sparse_weight_tensor),
        ("SM", encode_raw_bytes(sparse_masks, "SM", 1, len(sparse_masks))),
        ("b_fc", dec_fc_bias.unsqueeze(0)),
        ("conv1_w", dec_conv1_weight),
        ("conv1_b", dec_conv1_bias.unsqueeze(0)),
        ("conv2_w", dec_conv2_weight),
        ("conv2_b", dec_conv2_bias.unsqueeze(0)),
    ]
    
    tensor_map = dict(tensors)
    split_parts = {}
    for name, part_count in SPLIT_TENSORS.items():
        tensor = tensor_map[name]
        meta = tensor if isinstance(tensor, dict) else quantize_and_encode(tensor, name)
        split_parts[name] = {
            "meta": meta,
            "part_names": [f"{name}_part{i}" for i in range(1, part_count + 1)],
            "b64_parts": split_b64(meta["b64"], part_count),
        }

    for output_filename, tensor_names, split_entries in OUTPUT_FILES:
        write_weight_file(
            output_filename,
            [(name, tensor_map[name]) for name in tensor_map if name in tensor_names],
            split_entries,
            split_parts
        )

    print("Done! Weights successfully serialized into split Lau modules.")

if __name__ == '__main__':
    main()
