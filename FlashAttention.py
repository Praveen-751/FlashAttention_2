import torch
import time

def flash_attention_cpu(qkv, block_size=128):
    B, N, _, H, D = qkv.shape
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
    
    output = torch.zeros_like(q)
    scale = 1.0 / (D ** 0.5)

    for start in range(0, N, block_size):
        end = min(start + block_size, N)
        q_block = q[:, start:end]

        block_result = torch.zeros_like(q_block)
        for k_start in range(0, N, block_size):
            k_end = min(k_start + block_size, N)
            k_block = k[:, k_start:k_end]
            v_block = v[:, k_start:k_end]

            scores = torch.einsum("bqhd,bkhd->bqkh", q_block, k_block) * scale
            weights = torch.softmax(scores, dim=2)
            attn_output = torch.einsum("bqkh,bkhd->bqhd", weights, v_block)
            block_result += attn_output

        output[:, start:end] = block_result

    return output

# Test input
device = 'cpu'
qkv = torch.randn(1, 512, 3, 4, 64, device=device)

start = time.time()
out = flash_attention_cpu(qkv, block_size=128)
end = time.time()

print("Output shape:", out.shape)
print("Runtime (s):", round(end - start, 3))
