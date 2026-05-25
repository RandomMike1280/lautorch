"""
Verification script: Emulates the Lau runtime logic in Python to confirm
that the base64-encoded quantized convolutional weights produce the same
binarized outputs as the original PyTorch model.
"""
import base64
import torch
from vae_model import TinyVAE, TokenEmbeddingModel

def b64_to_bytes(b64_str):
    """Decode base64 string to list of 0-255 ints."""
    return list(base64.b64decode(b64_str.encode("ascii")))

def dequantize(byte_list, min_val, max_val):
    """Dequantize byte list back to floats (mirrors Lau dequantize)."""
    range_val = max_val - min_val
    return [min_val + (b / 255.0) * range_val for b in byte_list]

def reshape(flat, rows, cols):
    """Reshape flat list to 2D list (mirrors Lau reshape)."""
    mat = []
    idx = 0
    for _ in range(rows):
        row = []
        for _ in range(cols):
            row.append(flat[idx])
            idx += 1
        mat.append(row)
    return mat

def reshape_feature(flat, channels, height, width):
    feature = []
    idx = 0
    for _ in range(channels):
        channel = []
        for _ in range(height):
            row = []
            for _ in range(width):
                row.append(flat[idx])
                idx += 1
            channel.append(row)
        feature.append(channel)
    return feature

def matmul(A, B):
    """Matrix multiply two 2D lists (mirrors matmul.laum)."""
    rows_a = len(A)
    cols_a = len(A[0])
    cols_b = len(B[0])
    result = [[0.0]*cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            s = 0.0
            for k in range(cols_a):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result

def add_bias(mat, bias):
    return [[mat[0][j] + bias[0][j] for j in range(len(mat[0]))]]

def relu(mat):
    return [[max(0, x) for x in mat[0]]]

def relu_feature(feature):
    return [
        [[max(0, value) for value in row] for row in channel]
        for channel in feature
    ]

def upsample_nearest_2x(feature):
    channels = len(feature)
    height = len(feature[0])
    width = len(feature[0][0])
    result = []
    for ch in range(channels):
        channel = []
        for y in range(height * 2):
            row = []
            src_y = y // 2
            for x in range(width * 2):
                src_x = x // 2
                row.append(feature[ch][src_y][src_x])
            channel.append(row)
        result.append(channel)
    return result

def conv2d_same(feature, weight, bias, in_channels, out_channels, height, width, kernel_size):
    pad = kernel_size // 2
    result = []
    for out_ch in range(out_channels):
        out_channel = []
        for y in range(height):
            row = []
            for x in range(width):
                value = bias[0][out_ch]
                for in_ch in range(in_channels):
                    for ky in range(kernel_size):
                        in_y = y + ky - pad
                        if 0 <= in_y < height:
                            for kx in range(kernel_size):
                                in_x = x + kx - pad
                                if 0 <= in_x < width:
                                    w_idx = (((out_ch * in_channels) + in_ch) * kernel_size + ky) * kernel_size + kx
                                    value += feature[in_ch][in_y][in_x] * weight[w_idx]
                row.append(value)
            out_channel.append(row)
        result.append(out_channel)
    return result

def depthwise_conv2d_same(feature, weight, bias, channels, height, width):
    result = []
    for ch in range(channels):
        channel = []
        for y in range(height):
            row = []
            for x in range(width):
                value = bias[0][ch]
                for ky in range(3):
                    in_y = y + ky - 1
                    if 0 <= in_y < height:
                        for kx in range(3):
                            in_x = x + kx - 1
                            if 0 <= in_x < width:
                                value += feature[ch][in_y][in_x] * weight[ch * 9 + ky * 3 + kx]
                row.append(value)
            channel.append(row)
        result.append(channel)
    return result

def pointwise_conv2d(feature, weight, bias, in_channels, out_channels, height, width):
    result = []
    for out_ch in range(out_channels):
        channel = []
        for y in range(height):
            row = []
            for x in range(width):
                value = bias[0][out_ch]
                for in_ch in range(in_channels):
                    value += feature[in_ch][y][x] * weight[out_ch * in_channels + in_ch]
                row.append(value)
            channel.append(row)
        result.append(channel)
    return result

def pixel_shuffle_2x(feature, out_channels, height, width):
    result = []
    for out_ch in range(out_channels):
        channel = []
        for y in range(height * 2):
            row = []
            for x in range(width * 2):
                sub = (y % 2) * 2 + (x % 2)
                row.append(feature[out_ch * 4 + sub][y // 2][x // 2])
            channel.append(row)
        result.append(channel)
    return result

def sigmoid_approx_flatten(feature):
    result = [[]]
    for row in feature[0]:
        for x in row:
            result[0].append(0.5 * (1 + x / (1 + abs(x))))
    return result

def binarize(mat):
    return [[1.0 if x >= 0.5 else 0.0 for x in mat[0]]]

def print_ascii(flat):
    for r in range(28):
        row = ""
        for c in range(28):
            row += "##" if flat[r*28 + c] >= 1 else "  "
        print(row)

def parse_weights_laum(filepath):
    """Quick parser to extract weight data from the generated .laum file."""
    import re
    with open(filepath, 'r') as f:
        content = f.read()

    weights = {}
    string_pattern = r'modelWeights\["(\w+)"\]\s*=\s*"([^"]+)"'
    for match in re.finditer(string_pattern, content):
        weights[match.group(1)] = {'b64': match.group(2)}

    pattern = r'modelWeights\["(\w+)"\]\s*=\s*\{(.*?)\n\}'
    for match in re.finditer(pattern, content, re.DOTALL):
        name = match.group(1)
        block = match.group(2)

        min_val = float(re.search(r'\["min"\]\s*=\s*([+-]?\d+\.\d+)', block).group(1))
        max_val = float(re.search(r'\["max"\]\s*=\s*([+-]?\d+\.\d+)', block).group(1))
        rows = int(re.search(r'\["rows"\]\s*=\s*(\d+)', block).group(1))
        cols = int(re.search(r'\["cols"\]\s*=\s*(\d+)', block).group(1))

        b64_match = re.search(r'\["b64"\]\s*=\s*"([^"]+)"', block)
        part_match = re.search(r'\["parts"\]\s*=\s*\{([^}]+)\}', block)
        part_names = re.findall(r'"([^"]+)"', part_match.group(1)) if part_match else []

        weights[name] = {
            'min': min_val, 'max': max_val,
            'rows': rows, 'cols': cols,
            'b64': b64_match.group(1) if b64_match else '',
            'parts': part_names
        }
    return weights

def parse_split_weights_laum(filepaths):
    weights = {}
    for filepath in filepaths:
        weights.update(parse_weights_laum(filepath))
    for entry in weights.values():
        if entry.get('parts'):
            entry['b64'] = ''.join(weights[part]['b64'] for part in entry['parts'])
    return weights

def decode_weight(entry):
    """Decode a weight entry as a matrix (mirrors Lau decodeWeight)."""
    byte_list = b64_to_bytes(entry['b64'])
    floats = dequantize(byte_list, entry['min'], entry['max'])
    return reshape(floats, entry['rows'], entry['cols'])

def decode_flat_weight(entry):
    """Decode a weight entry as a flat list (mirrors Lau decodeFlatWeight)."""
    byte_list = b64_to_bytes(entry['b64'])
    return dequantize(byte_list, entry['min'], entry['max'])

def sparse_relu(z, sparse_weights, sparse_masks, bias):
    out = [[]]
    k = 0
    for j, raw_mask in enumerate(sparse_masks):
        s = bias[0][j]
        mask = int(round(raw_mask))
        if mask >= 8:
            s += z[3] * sparse_weights[k]
            k += 1
            mask -= 8
        if mask >= 4:
            s += z[2] * sparse_weights[k]
            k += 1
            mask -= 4
        if mask >= 2:
            s += z[1] * sparse_weights[k]
            k += 1
            mask -= 2
        if mask >= 1:
            s += z[0] * sparse_weights[k]
            k += 1
        out[0].append(max(0, s))
    return out

def lau_forward(digit, embed, sparse_weights, sparse_masks, b_fc,
                up1_dw, up1_dw_b, up1_pw, up1_pw_b,
                up2_dw, up2_dw_b, up2_pw, up2_pw_b):
    z_lau = [embed[digit]]
    h = sparse_relu(z_lau[0], sparse_weights, sparse_masks, b_fc)

    feature = reshape_feature(h[0], 8, 7, 7)
    feature = relu_feature(feature)
    feature = depthwise_conv2d_same(feature, up1_dw, up1_dw_b, 8, 7, 7)
    feature = relu_feature(feature)
    feature = pointwise_conv2d(feature, up1_pw, up1_pw_b, 8, 16, 7, 7)
    feature = pixel_shuffle_2x(feature, 4, 7, 7)
    feature = relu_feature(feature)
    feature = depthwise_conv2d_same(feature, up2_dw, up2_dw_b, 4, 14, 14)
    feature = relu_feature(feature)
    feature = pointwise_conv2d(feature, up2_pw, up2_pw_b, 4, 4, 14, 14)
    feature = pixel_shuffle_2x(feature, 1, 14, 14)
    out = sigmoid_approx_flatten(feature)
    return binarize(out)[0]

def main():
    print("=" * 60)
    print("VERIFICATION: Comparing Lau-emulated vs PyTorch outputs")
    print("=" * 60)

    weights = parse_split_weights_laum((
        'weights_data_1.laum',
        'weights_data_2.laum',
        'weights_data_3.laum',
    ))
    embed = decode_weight(weights['embed'])
    sparse_weights = decode_flat_weight(weights['SW'])
    sparse_masks = decode_flat_weight(weights['SM'])
    b_fc = decode_weight(weights['b_fc'])
    up1_dw = decode_flat_weight(weights['U1D'])
    up1_dw_b = decode_weight(weights['U1DB'])
    up1_pw = decode_flat_weight(weights['U1P'])
    up1_pw_b = decode_weight(weights['U1PB'])
    up2_dw = decode_flat_weight(weights['U2D'])
    up2_dw_b = decode_weight(weights['U2DB'])
    up2_pw = decode_flat_weight(weights['U2P'])
    up2_pw_b = decode_weight(weights['U2PB'])

    checkpoint = torch.load('vae_and_embed.pth', map_location='cpu')
    vae = TinyVAE(
        latent_dim=checkpoint['latent_dim'],
        lora_rank=checkpoint.get('lora_rank', 0),
        lora_alpha=checkpoint.get('lora_alpha', 1.0),
    )
    token_emb = TokenEmbeddingModel(num_tokens=10, latent_dim=checkpoint['latent_dim'])
    vae.load_state_dict(checkpoint['vae_state_dict'])
    token_emb.load_state_dict(checkpoint['embed_state_dict'])
    vae.use_hard_mask = True
    vae.eval()
    token_emb.eval()

    for digit in range(0, 10):
        lau_binary = lau_forward(digit, embed, sparse_weights, sparse_masks, b_fc,
                                 up1_dw, up1_dw_b, up1_pw, up1_pw_b,
                                 up2_dw, up2_dw_b, up2_pw, up2_pw_b)

        with torch.no_grad():
            t = torch.tensor([digit], dtype=torch.long)
            z_pt = token_emb(t)
            decoded = vae.decode(z_pt)
            pt_binary = (decoded >= 0.5).float().squeeze().tolist()

        match_count = sum(1 for a, b in zip(lau_binary, pt_binary) if a == b)
        total = len(lau_binary)
        pct = match_count / total * 100

        status = "PASS" if pct > 95 else "WARN" if pct > 85 else "FAIL"
        print(f"Digit {digit}: {match_count}/{total} pixels match ({pct:.1f}%) [{status}]")

    print()
    print("Rendering Lau-emulated outputs for digits 0-9:")
    for digit in range(0, 10):
        result = lau_forward(digit, embed, sparse_weights, sparse_masks, b_fc,
                             up1_dw, up1_dw_b, up1_pw, up1_pw_b,
                             up2_dw, up2_dw_b, up2_pw, up2_pw_b)

        print(f"\n--- Digit {digit} ---")
        print_ascii(result)

if __name__ == '__main__':
    main()
