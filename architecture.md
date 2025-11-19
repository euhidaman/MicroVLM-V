# MicroVLM-V Architecture

## Overview

MicroVLM-V is a tiny vision-language model with episodic memory capabilities, designed to achieve total model size under 500MB after quantization. The architecture combines components from four research repositories:

- EVO-1: Image-text alignment methodology
- Larimar: Episodic memory architecture
- BitNet: 1.58-bit quantization
- LeJEPA: Attention visualization and analysis

## Model Components

### 1. Vision Encoder: DeiT-Tiny

The vision encoder uses the Data-efficient Image Transformer (DeiT) Tiny variant.

**Specifications:**
- Model: facebook/deit-tiny-patch16-224
- Image size: 224x224
- Patch size: 16x16
- Number of patches: 196 (14x14 grid)
- Hidden dimension: 192
- Number of layers: 12
- Number of attention heads: 3
- Intermediate size: 768 (4 * hidden_size)

**Output:**
- Patch tokens: (batch_size, 196, 192)
- CLS token: (batch_size, 192) for image-level features

**Size:**
- Original: ~20MB (5.7M parameters)
- With 4-bit quantization: ~5MB

### 2. Multimodal Adapter

Projects vision embeddings to language model space and performs pooling.

**Architecture:**
```
Input: (batch, 196, 192)  # DeiT patch tokens
  |
  v
Projection: Linear(192 -> 896)
  |
  v
MLP: 
  Linear(896 -> 1792)
  GELU()
  Linear(1792 -> 896)
  |
  v
Cross-Attention Pooling:
  Queries: (k_prefix, 896) learned
  Keys/Values: (196, 896) from patches
  Output: (batch, k_prefix, 896)
  |
  v
Positional Embeddings: (k_prefix, 896) learned
  |
  v
LayerNorm(896)
  |
  v
Output: (batch, k_prefix, 896)  # Prefix tokens
```

**K_prefix Computation:**
```
num_patches = 196
k_prefix = clamp(ceil(num_patches / 8), 8, 64)
k_prefix = clamp(ceil(196 / 8), 8, 64)
k_prefix = clamp(25, 8, 64) = 25
```

**Parameters:**
- Projection: 192 * 896 + 896 = 172,928
- MLP: (896 * 1792 + 1792) + (1792 * 896 + 896) = 3,211,264
- Pooling queries: 25 * 896 = 22,400
- Positional embeddings: 25 * 896 = 22,400
- Total: ~3.4M parameters (~14MB FP32, maintained in training)

### 3. EVO-1 Image-Text Alignment

Implements contrastive learning for multimodal alignment following EVO-1 methodology.

**Contrastive Loss:**
```
image_features = CLS_token from DeiT  # (batch, 192)
text_features = mean_pool(text_embeddings)  # (batch, 896)

# Project image features to common space
image_proj = Linear(192 -> 896)(image_features)

# Normalize
image_norm = normalize(image_proj, p=2, dim=-1)
text_norm = normalize(text_features, p=2, dim=-1)

# Similarity matrix
logits = (image_norm @ text_norm.T) / temperature

# Bidirectional cross-entropy
labels = [0, 1, 2, ..., batch_size-1]
loss_i2t = CrossEntropy(logits, labels)
loss_t2i = CrossEntropy(logits.T, labels)

alignment_loss = (loss_i2t + loss_t2i) / 2
```

**Multimodal Fusion:**
```
prefix_tokens: (batch, 25, 896)
text_embeddings: (batch, seq_len, 896)

fused = concat([prefix_tokens, text_embeddings], dim=1)
# Output: (batch, 25 + seq_len, 896)

attention_mask = concat([ones(batch, 25), text_mask], dim=1)
# Prefix tokens always attended to
```

### 4. Language Backbone: Qwen2.5-0.5B

**Specifications:**
- Model: Qwen/Qwen2.5-0.5B
- Hidden size: 896
- Number of layers: 24
- Number of attention heads: 14
- Number of KV heads: 2 (Grouped Query Attention)
- Intermediate size: 4864
- Vocabulary size: 151,936
- Max position embeddings: 32,768
- Head dimension: 64
- KV head dimension: 128

**Size:**
- Original: 988MB (494M parameters)
- With 4-bit quantization (training): ~247MB
- With 1.58-bit quantization (inference): ~98MB

**Architecture:**
- Embedding layer: 151,936 * 896
- 24 Transformer decoder layers with:
  - Grouped Query Attention (14 query heads, 2 KV heads)
  - SwiGLU feedforward networks
  - RMSNorm
  - RoPE positional encoding
- Output head: tied with embedding

### 5. Larimar Episodic Memory

Implements Gaussian Process Memory (GPM) with Sherman-Morrison updates.

**Memory Matrix:**
```
M: (K_mem, C_mem)
K_mem = 512  # Memory slots
C_mem = 896  # Same as Qwen hidden dimension
```

**Write Operation (Larimar Eq. 3):**
```
Given: z_t (fused context embedding)

1. Sequential LSTM ordering:
   z_ordered = LSTM(z_t)  # Bidirectional LSTM

2. For each time step t:
   a. Compute addressing weights w_t:
      w_t = pseudo_inverse(M) @ z_t
   
   b. Sherman-Morrison update:
      Delta = z_t - (w_t.T @ M)
      wU = w_t.T @ Cov
      sigma_z = wU @ w_t + noise_var
      c_z = wU / sigma_z
      
      M_new = M + c_z.T @ Delta
      Cov_new = Cov - c_z.T @ wU

3. Output: updated memory state (M, Cov)
```

**Pseudo-Inverse Approximation (Ben-Cohen Method):**
```
Initialize: A_inv = alpha * A.T  (alpha ~ 5e-4)

For i in range(3):  # 3 iterations
    A_inv = 2 * A_inv - (A_inv @ A @ A_inv)

Return: A_inv
```

**Read Operation (Larimar Eq. 4):**
```
Given: query z, memory state (M, Cov)

1. Compute addressing weights:
   w_mean = pseudo_inverse(M) @ z
   
2. Sample with Gaussian noise:
   w = w_mean + std * randn_like(w_mean)
   std = exp(0.5 * w_logvar)

3. Retrieve:
   z_retrieved = w.T @ M
   
4. Add observation noise:
   z_retrieved += noise_std * randn_like(z_retrieved)

Return: z_retrieved, KL_divergence(w)
```

**Projection to KV Space (W_M):**
```
Input: z_retrieved (episode_size, batch, 896)

W_M: Linear(896 -> num_layers * num_heads * head_dim * 2)
W_M: Linear(896 -> 24 * 14 * 64 * 2)
W_M: Linear(896 -> 43,008)

Output: Reshaped to KV pairs for each layer
  For layer l in [0, 23]:
    K_l: (batch, num_heads, head_dim) = (batch, 14, 64)
    V_l: (batch, num_heads, head_dim) = (batch, 14, 64)

These KV pairs are injected before self-attention in decoder.
```

**Memory Injection via ScopeNet:**
```
ScopeNet: MLP classifier
  Input: fused_context (batch, 896)
  Hidden: Linear(896 -> 256) + ReLU + Dropout(0.1)
  Hidden: Linear(256 -> 256) + ReLU + Dropout(0.1)
  Output: Linear(256 -> 1) + Sigmoid
  
Output: injection_probability (batch, 1)

If probability > threshold:
  Inject memory-derived KV into decoder layers
```

**Memory Parameters:**
- Memory matrix M: 512 * 896 = 458,752
- W_M projection: 896 * 43,008 + 43,008 = 38,579,776
- LSTM ordering: ~1M parameters
- Total: ~40M parameters

**Size with 1.58-bit quantization:**
- M quantized: ~90KB
- W_M quantized: ~8MB
- Total memory system: ~9MB

### 6. Attention Visualization (LeJEPA)

**SlicingUnivariateTest:**
```
Purpose: Analyze distributional differences in attention

1. Random Projection:
   x: (batch, seq_len, hidden_dim)
   A: (hidden_dim, num_slices)  # Random normalized vectors
   A = A / ||A||_2
   
   x_proj = x @ A  # (batch, seq_len, num_slices)

2. FastEppsPulley Test:
   For each slice k:
     Compute characteristic function
     char_fn = exp(i * x_proj[:, :, k] * t)
     
     Statistic = mean(|char_fn|^2)
   
3. Aggregate:
   If reduction='mean': mean over all slices
   If reduction='sum': sum over all slices
```

**Configuration:**
- num_slices: 256
- reduction: 'mean'
- sampler: 'gaussian'
- clip_value: 0.01

**Visualization:**
- Cross-attention heatmaps: text tokens x image tokens
- Memory addressing heatmaps: episode steps x memory slots
- Generated every 5000 training steps
- Logged to WandB

## Quantization Strategy

### Training: 4-bit Quantization

**Symmetric 4-bit:**
```
For weight W:
  qmax = 2^(4-1) - 1 = 7
  scale = max(|W|) / qmax
  W_quant = round(W / scale).clamp(-8, 7)
  W_dequant = W_quant * scale
```

**Applied to:**
- Qwen2.5-0.5B weights: 988MB -> 247MB
- DeiT-Tiny weights: 20MB -> 5MB

**Fake Quantization for QAT:**
```
Forward: Quantize and dequantize
Backward: Straight-through estimator (gradients flow unchanged)
```

### Inference: 1.58-bit Quantization

**Ternary Quantization {-1, 0, +1}:**
```
For weight W:
  scale = mean(|W|)
  W_norm = W / scale
  
  W_quant = {
    +1  if W_norm > 0.5
    -1  if W_norm < -0.5
    0   otherwise
  }
  
  W_dequant = W_quant * scale
```

**Storage:**
- 2 bits per ternary value (4 values in 1 byte with packing)
- Effective: 1.58 bits per parameter

**Applied to:**
- Final Qwen2.5-0.5B: 988MB -> 98MB
- Episodic memory M: 458,752 params -> 90KB

## Model Size Breakdown

### Full Model (Unquantized)
- Qwen2.5-0.5B: 494M params * 4 bytes = 1976 MB
- DeiT-Tiny: 5.7M params * 4 bytes = 23 MB
- Multimodal Adapter: 3.4M params * 4 bytes = 14 MB
- Episodic Memory: 40M params * 4 bytes = 160 MB
- Total: ~2173 MB

### Training (4-bit Quantization)
- Qwen2.5-0.5B: 494M * 0.5 bytes = 247 MB
- DeiT-Tiny: 5.7M * 0.5 bytes = 3 MB
- Multimodal Adapter: 14 MB (FP32)
- Episodic Memory: 160 MB (FP32)
- Total: ~424 MB

### Inference (1.58-bit Quantization)
- Qwen2.5-0.5B: 494M * 0.2 bytes = 98 MB
- DeiT-Tiny: 5.7M * 0.5 bytes = 3 MB
- Multimodal Adapter: 14 MB
- Episodic Memory: 40M * 0.2 bytes = 8 MB
- Total: ~123 MB

**Target achieved: < 500 MB**

## Training Pipeline

### Stage 1: Adapter and Memory Training
- Freeze: DeiT-Tiny (all layers)
- Freeze: Qwen2.5-0.5B (all layers)
- Train:
  - Multimodal adapter
  - W_M projection
  - ScopeNet
  - Episodic memory encoder
- Duration: 10 epochs
- Learning rate: 1e-4

### Stage 2: Language Model Fine-tuning
- Freeze: DeiT-Tiny (all layers)
- Freeze: Qwen2.5-0.5B (first 20 layers)
- Unfreeze: Qwen2.5-0.5B (last 4 layers)
- Train: All from Stage 1 + last 4 Qwen layers
- Duration: 5 epochs
- Learning rate: 1e-5 (lower to prevent drift)

### Loss Function

```
Total Loss = LM_loss + alpha * alignment_loss + beta * memory_KL + gamma * addressing_KL

LM_loss: Cross-entropy language modeling loss
alignment_loss: Bidirectional contrastive loss (EVO-1)
memory_KL: KL(posterior_memory || prior_memory)
addressing_KL: KL(addressing_weights || standard_normal)

Weights:
  alpha = 0.1
  beta = 0.01
  gamma = 0.001
```

## Tensor Shape Flow

**Example forward pass:**

```
Input image: (B, 3, 224, 224)
  |
  v DeiT encoder
Patch tokens: (B, 196, 192)
  |
  v Multimodal adapter
Prefix tokens: (B, 25, 896)
  |
Input text tokens: (B, L)
  |
  v Qwen embedding
Text embeddings: (B, L, 896)
  |
  v Fusion (concatenation)
Fused embeddings: (B, 25+L, 896)
  |
  v Context extraction (mean pool)
Context: (B, 896)
  |
  v Memory write
  Reshape to episode: (E, B/E, 896)  # E = episode_size
  LSTM ordering: (E, B/E, 896)
  Memory update: M (K, C) = (512, 896)
  |
  v Memory read
  Addressing: w (E, B/E, 512)
  Retrieved: z_r (E, B/E, 896)
  |
  v W_M projection
  KV for all layers: (B, 24, 14, 64, 2)
  Split: K_l, V_l for l in [0, 23]
  |
  v ScopeNet
  Injection decision: (B, 1)
  |
  v Qwen decoder with memory KV injection
Output logits: (B, 25+L, 151936)
```

## Performance Characteristics

**Inference Latency (Estimated):**
- Vision encoding: ~50ms (DeiT-Tiny)
- Multimodal adaptation: ~10ms
- Memory retrieval: ~20ms
- Language generation (50 tokens): ~500ms
- Total: ~580ms per image-text pair

**Memory Usage (Runtime):**
- Model weights: ~123 MB
- Activations (batch=1): ~50 MB
- KV cache: ~30 MB
- Total: ~203 MB

**Deployment Target:**
- Raspberry Pi Zero 2 W: 512 MB RAM
- Available for model: ~300 MB
- Fits comfortably with headroom

## Mathematical Foundations

### Episodic Memory (Gaussian Process)

**Prior Distribution:**
```
p(M) = N(M | M_0, Sigma_0)
M_0: (K, C) initialized as identity-like
Sigma_0: (K, K) diagonal covariance
```

**Posterior After Observations:**
```
Given observations: {z_1, z_2, ..., z_T}

p(M | z_{1:T}) ∝ p(z_{1:T} | M) p(M)

Sequential update using Sherman-Morrison:
  M_t = M_{t-1} + (Sigma_{t-1} w_t / (w_t^T Sigma_{t-1} w_t + sigma_noise^2)) 
        * (z_t - w_t^T M_{t-1})
```

**KL Divergence:**
```
KL(posterior || prior) = 0.5 * [
  tr(Sigma_0^{-1} Sigma_t) +
  (M_t - M_0)^T Sigma_0^{-1} (M_t - M_0) -
  K*C + log(det(Sigma_0) / det(Sigma_t))
]
```

### Contrastive Alignment (EVO-1)

**InfoNCE Loss:**
```
sim(i, j) = (f_img(i)^T f_text(j)) / tau

L_i2t = -log(exp(sim(i, i)) / sum_j exp(sim(i, j)))
L_t2i = -log(exp(sim(i, i)) / sum_i exp(sim(i, j)))

L_align = (L_i2t + L_t2i) / 2
```

### Attention Visualization (LeJEPA)

**Sliced Wasserstein Distance:**
```
For distributions P and Q over R^d:

SW(P, Q) = int_{S^{d-1}} W_1(P_theta, Q_theta) dtheta

Where:
  S^{d-1}: unit sphere in R^d
  P_theta, Q_theta: projections onto direction theta
  W_1: 1-Wasserstein distance

Approximated by random projections.
```

## Implementation Notes

**Key Design Decisions:**

1. K_prefix = 25 tokens provides good balance between information preservation and computational efficiency

2. 1.58-bit quantization applied only to final model and memory, not during training

3. Sequential Sherman-Morrison updates more stable than batch matrix inversion

4. Episodic memory with episode_size=4 provides temporal context without excessive computational cost

5. Freezing strategy allows efficient training with limited compute

6. WandB logging organized into sections for comprehensive monitoring

**Critical Constraints Met:**

- Total model size: ~123 MB < 500 MB target
- No markdown files except README.md and architecture.md
- No .bat files - only Python scripts
- No test files to avoid execution errors
- Sequential execution order maintained
- All code directly implemented
- Small-scale testing option included
