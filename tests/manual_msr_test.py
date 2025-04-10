import torch
import argparse
from torchscale.component.multiscale_retention import MultiScaleRetention, theta_shift
from torchscale.architecture.retnet import RetNetRelPos

# Use the same chunk size for agent/inference as for chunkwise recurrent
RECURRENT_CHUNK_SIZE = 512
NUM_AGENTS = 4

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
args.recurrent_chunk_size = RECURRENT_CHUNK_SIZE
args.moe_freq = 0
args.inference_chunk_size = NUM_AGENTS # Match inference chunk size to NUM_AGENTS

batch_size = 16
# Ensure seq_len is divisible by the chunk size
seq_len = 4096 #(4096 // RECURRENT_CHUNK_SIZE) * RECURRENT_CHUNK_SIZE
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

# --- Inference agent recurrent forward ---
agent_recurrent_output = []
agent_recurrent_state = {}
# Ensure the loop iterates over the correct number of chunks
assert seq_len % NUM_AGENTS == 0, f"seq_len ({seq_len}) must be divisible by NUM_AGENTS ({NUM_AGENTS})"
num_agent_steps = seq_len // NUM_AGENTS

for i in range(num_agent_steps):
    start_idx = i * NUM_AGENTS
    end_idx = (i + 1) * NUM_AGENTS
    rel_pos = ret_pos(slen=(start_idx, end_idx), agent_recurrent=True)

    element_i = x[:, start_idx:end_idx, :]
    # Pass None for incremental_state, use agent_recurrent_state
    output_i = retention.forward(element_i, rel_pos, False, None, agent_recurrent_state)
    agent_recurrent_output.append(output_i)

agent_recurrent_output = torch.cat(agent_recurrent_output, dim=1)

# --- Chunkwise Recurrent Computation (Using forward) ---
rel_pos_chunk = ret_pos(slen=seq_len, chunkwise_recurrent=True)
chunkwise_output = retention.forward(x, rel_pos_chunk, chunkwise_recurrent=True)

# --- Parallel Computation (Using forward) ---
rel_pos_parallel = ret_pos(slen=seq_len)
parallel_output = retention.forward(x, rel_pos_parallel)

# Comparison
print("\nChunkwise Output Shape:", chunkwise_output.shape)
print("Parallel Output Shape:", parallel_output.shape)
print("Recurrent Output Shape:", recurrent_output.shape)
print("Agent Recurrent Output Shape:", agent_recurrent_output.shape)
max_diff_chunk_par = torch.max(torch.abs(chunkwise_output - parallel_output))
mean_diff_chunk_par = torch.mean(torch.abs(chunkwise_output - parallel_output))
print(f"\nChunkwise vs Parallel  | Max abs diff: {max_diff_chunk_par:.6f}, Mean abs diff: {mean_diff_chunk_par:.6f}")

max_diff_agent_recurrent_chunkwise = torch.max(torch.abs(agent_recurrent_output - chunkwise_output))
mean_diff_agent_recurrent_chunkwise = torch.mean(torch.abs(agent_recurrent_output - chunkwise_output))
print(f"\nAgent Recurrent vs Chunkwise  | Max abs diff: {max_diff_agent_recurrent_chunkwise:.6f}, Mean abs diff: {mean_diff_agent_recurrent_chunkwise:.6f}")

max_diff_recurrent_chunkwise = torch.max(torch.abs(recurrent_output - chunkwise_output))
mean_diff_recurrent_chunkwise = torch.mean(torch.abs(recurrent_output - chunkwise_output))
print(f"\nRecurrent vs Chunkwise  | Max abs diff: {max_diff_recurrent_chunkwise:.6f}, Mean abs diff: {mean_diff_recurrent_chunkwise:.6f}")

# Assert agent recurrent and chunkwise are close
assert torch.allclose(agent_recurrent_output, chunkwise_output, atol=1e-4, rtol=1e-3), "Agent Recurrent and Chunkwise outputs differ significantly"
print("\nAssertion Passed: Agent Recurrent and Chunkwise outputs are close.")