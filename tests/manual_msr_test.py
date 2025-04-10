import torch
import argparse
from torchscale.component.multiscale_retention import MultiScaleRetention, theta_shift
from torchscale.architecture.retnet import RetNetRelPos


# Configuration
args = argparse.Namespace()
args.decoder_embed_dim = 64
args.decoder_value_embed_dim = 128
args.decoder_retention_heads = 4
args.decoder_ffn_embed_dim = 128
args.activation_fn = 'swish'
args.dropout = 0.0
args.activation_dropout = 0.0
args.drop_path_rate = 0.0
args.decoder_layers = 1
args.decoder_normalize_before = True
args.layernorm_eps = 1e-5
args.deepnorm = False
args.subln = True
args.multiway = False
args.vocab_size = 100
args.layernorm_embedding = False
args.no_scale_embedding = False
args.checkpoint_activations = False
args.fsdp = False
args.recurrent_chunk_size = 32
args.moe_freq = 0

batch_size = 16
seq_len = 4096
embed_dim = args.decoder_embed_dim
num_heads = args.decoder_retention_heads
head_dim = args.decoder_value_embed_dim // num_heads
key_dim = embed_dim // num_heads

retention = MultiScaleRetention(args, embed_dim, args.decoder_value_embed_dim, num_heads)
ret_pos = RetNetRelPos(args)
retention.eval()

x = torch.randn(batch_size, seq_len, embed_dim)

# --- Recurrent forward passes ---
incremental_state = {}

recurrent_output = []
for i in range(seq_len):
    rel_pos = ret_pos(slen=i+1, activate_recurrent=True)

    element_i = x[:, i:i+1, :]
    output_i = retention.forward(element_i, rel_pos, False, incremental_state)
    recurrent_output.append(output_i)

recurrent_output = torch.cat(recurrent_output, dim=1)

print(recurrent_output.shape)

# --- Chunkwise Recurrent Computation (Using forward) ---
rel_pos_chunk = ret_pos(slen=seq_len, chunkwise_recurrent=True)
chunkwise_output = retention.forward(x, rel_pos_chunk, chunkwise_recurrent=True)

# --- Parallel Computation (Using forward) ---
rel_pos_parallel = ret_pos(slen=seq_len, activate_recurrent=False, chunkwise_recurrent=False)
parallel_output = retention.forward(x, rel_pos_parallel)

# Comparison
print("\nChunkwise Output Shape:", chunkwise_output.shape)
print("Parallel Output Shape:", parallel_output.shape)
print("Recurrent Output Shape:", recurrent_output.shape)
max_diff_chunk_par = torch.max(torch.abs(chunkwise_output - parallel_output))
mean_diff_chunk_par = torch.mean(torch.abs(chunkwise_output - parallel_output))
print(f"\nChunkwise vs Parallel  | Max abs diff: {max_diff_chunk_par:.6f}, Mean abs diff: {mean_diff_chunk_par:.6f}")

max_diff_recurrent_chunkwise = torch.max(torch.abs(recurrent_output - chunkwise_output))
mean_diff_recurrent_chunkwise = torch.mean(torch.abs(recurrent_output - chunkwise_output))
print(f"\nRecurrent vs Chunkwise  | Max abs diff: {max_diff_recurrent_chunkwise:.6f}, Mean abs diff: {mean_diff_recurrent_chunkwise:.6f}")

max_diff_recurrent_parallel = torch.max(torch.abs(recurrent_output - parallel_output))
mean_diff_recurrent_parallel = torch.mean(torch.abs(recurrent_output - parallel_output))
print(f"\nRecurrent vs Parallel  | Max abs diff: {max_diff_recurrent_parallel:.6f}, Mean abs diff: {mean_diff_recurrent_parallel:.6f}")

# Assert recurrent and chunkwise are close
assert torch.allclose(recurrent_output, chunkwise_output, atol=1e-4, rtol=1e-3), "Recurrent and Chunkwise outputs differ significantly"
print("\nAssertion Passed: Recurrent and Chunkwise outputs are close.")