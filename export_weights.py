import torch

OUTPUT_FILES = (
    ("weights_data_1.laum", {"embed", "b_fc", "conv1_b", "conv2_b"}, {"W_fc": 0}),
    ("weights_data_2.laum", set(), {"W_fc": 1}),
    ("weights_data_3.laum", {"conv1_w", "conv2_w"}, {"W_fc": 2}),
)
SPLIT_TENSORS = {"W_fc": 3}

def quantize_and_encode(tensor, name):
    """
    Quantizes a PyTorch tensor to 8-bit integers and encodes it as a hex string.
    Returns a dictionary of metadata and the hex string.
    """
    flat_tensor = tensor.flatten()
    min_val = flat_tensor.min().item()
    max_val = flat_tensor.max().item()
    
    # Avoid division by zero if min and max are equal
    range_val = max_val - min_val if max_val != min_val else 1.0
    
    # 8-bit scale to 0..255
    quantized = ((flat_tensor - min_val) / range_val * 255.0).round().clamp(0, 255).int()
    hex_str = "".join(f"{val:02x}" for val in quantized.tolist())
    
    # Get rows and cols for matrix consumers. Full dims are kept for conv kernels.
    shape = tensor.shape
    rows = shape[0] if len(shape) > 0 else 1
    cols = shape[1] if len(shape) > 1 else 1
    
    return {
        "name": name,
        "min": min_val,
        "max": max_val,
        "hex": hex_str,
        "rows": rows,
        "cols": cols,
        "dims": list(shape)
    }

def get_decoder_state(checkpoint):
    if 'decoder_state_dict' in checkpoint:
        return checkpoint['decoder_state_dict']
    if 'vae_state_dict' in checkpoint:
        vae_state = checkpoint['vae_state_dict']
        return {
            'dec_fc.weight': vae_state['dec_fc.weight'],
            'dec_fc.bias': vae_state['dec_fc.bias'],
            'dec_conv1.weight': vae_state['dec_conv1.weight'],
            'dec_conv1.bias': vae_state['dec_conv1.bias'],
            'dec_conv2.weight': vae_state['dec_conv2.weight'],
            'dec_conv2.bias': vae_state['dec_conv2.bias'],
        }
    raise KeyError("Checkpoint must contain 'decoder_state_dict' or 'vae_state_dict'.")

def split_hex(hex_str, parts):
    part_size = (len(hex_str) + parts - 1) // parts
    if part_size % 2 == 1:
        part_size += 1
    return [hex_str[i:i + part_size] for i in range(0, len(hex_str), part_size)]

def write_weight_entry(f, name, meta, hex_override=None, part_names=None):
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
        hex_str = meta['hex'] if hex_override is None else hex_override
        f.write(",\n")
        f.write(f"    [\"hex\"] = \"{hex_str}\"\n")
    f.write("}\n\n")

def write_hex_part(f, name, hex_str):
    f.write(f"modelWeights[\"{name}\"] = \"{hex_str}\"\n\n")

def write_weight_file(output_filename, tensors, split_entries, split_parts):
    print(f"Quantizing and writing Lau weight tables to '{output_filename}'...")

    with open(output_filename, "w") as f:
        f.write("-- Quantized weights data for Lau MNIST VAE\n")
        f.write("varol modelWeights = {}\n\n")

        for name, tensor in tensors:
            meta = quantize_and_encode(tensor, name)
            write_weight_entry(f, name, meta)

        for name, part_index in split_entries.items():
            meta = split_parts[name]["meta"]
            part_names = split_parts[name]["part_names"]
            part_name = part_names[part_index]
            if part_index == 0:
                write_weight_entry(f, name, meta, part_names=part_names)
            write_hex_part(f, part_name, split_parts[name]["hex_parts"][part_index])

        f.write("return modelWeights\n")

def main():
    checkpoint_path = 'decoder_and_embed.pth'
    print(f"Loading '{checkpoint_path}'...")
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    decoder_state = get_decoder_state(checkpoint)
    
    embed_weight = checkpoint['embed_state_dict']['embedding.weight']
    dec_fc_weight = decoder_state['dec_fc.weight']
    dec_fc_bias = decoder_state['dec_fc.bias']
    dec_conv1_weight = decoder_state['dec_conv1.weight']
    dec_conv1_bias = decoder_state['dec_conv1.bias']
    dec_conv2_weight = decoder_state['dec_conv2.weight']
    dec_conv2_bias = decoder_state['dec_conv2.bias']
    
    # Transpose weights to match Lau matrix multiplication convention (x * W_transposed)
    # so we do not need to transpose at runtime in Lau.
    # z (1 x 8) * W_fc (8 x 392) -> (1 x 392)
    dec_fc_weight_t = dec_fc_weight.t()
    
    tensors = [
        ("embed", embed_weight),
        ("W_fc", dec_fc_weight_t),
        ("b_fc", dec_fc_bias.unsqueeze(0)),
        ("conv1_w", dec_conv1_weight),
        ("conv1_b", dec_conv1_bias.unsqueeze(0)),
        ("conv2_w", dec_conv2_weight),
        ("conv2_b", dec_conv2_bias.unsqueeze(0)),
    ]
    
    tensor_map = dict(tensors)
    split_parts = {}
    for name, part_count in SPLIT_TENSORS.items():
        meta = quantize_and_encode(tensor_map[name], name)
        split_parts[name] = {
            "meta": meta,
            "part_names": [f"{name}_part{i}" for i in range(1, part_count + 1)],
            "hex_parts": split_hex(meta["hex"], part_count),
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
