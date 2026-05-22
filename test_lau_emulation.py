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

def lau_forward(digit, embed, W_fc, b_fc, conv1_w, conv1_b, conv2_w, conv2_b):
    z_lau = [embed[digit]]
    h = matmul(z_lau, W_fc)
    h = add_bias(h, b_fc)
    h = relu(h)

    feature = reshape_feature(h[0], 8, 7, 7)
    feature = upsample_nearest_2x(feature)
    feature = conv2d_same(feature, conv1_w, conv1_b, 8, 4, 14, 14, 3)
    feature = relu_feature(feature)
    feature = upsample_nearest_2x(feature)
    feature = conv2d_same(feature, conv2_w, conv2_b, 4, 1, 28, 28, 3)
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
    W_fc = decode_weight(weights['W_fc'])
    b_fc = decode_weight(weights['b_fc'])
    conv1_w = decode_flat_weight(weights['conv1_w'])
    conv1_b = decode_weight(weights['conv1_b'])
    conv2_w = decode_flat_weight(weights['conv2_w'])
    conv2_b = decode_weight(weights['conv2_b'])

    checkpoint = torch.load('vae_and_embed.pth', map_location='cpu')
    vae = TinyVAE(latent_dim=checkpoint['latent_dim'])
    token_emb = TokenEmbeddingModel(num_tokens=10, latent_dim=checkpoint['latent_dim'])
    vae.load_state_dict(checkpoint['vae_state_dict'])
    token_emb.load_state_dict(checkpoint['embed_state_dict'])
    vae.eval()
    token_emb.eval()

    for digit in range(1, 10):
        lau_binary = lau_forward(digit, embed, W_fc, b_fc, conv1_w, conv1_b, conv2_w, conv2_b)

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
    print("Rendering Lau-emulated outputs for digits 1-9:")
    for digit in range(1, 10):
        result = lau_forward(digit, embed, W_fc, b_fc, conv1_w, conv1_b, conv2_w, conv2_b)

        print(f"\n--- Digit {digit} ---")
        print_ascii(result)

if __name__ == '__main__':
    main()
