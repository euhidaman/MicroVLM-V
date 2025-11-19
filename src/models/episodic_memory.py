"""
Episodic Memory Module
Implements Larimar's episodic memory architecture with 1.58-bit quantization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


EPSILON = 1e-6


class EpisodicMemory(nn.Module):
    """
    Larimar-style episodic memory with Gaussian Process Memory
    Implements write/read operations with Sherman-Morrison updates
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.memory_size = config.get('memory_size', 512)  # K_mem
        self.code_size = config.get('memory_dim', 896)  # C_mem
        self.qwen_hidden_dim = config.get('qwen_hidden_dim', 896)
        self.num_layers = config.get('num_layers', 24)
        self.num_heads = config.get('num_heads', 14)
        self.head_dim = config.get('head_dim', 64)
        
        # Memory parameters: M of shape (K_mem, C_mem)
        self.memory_mean = nn.Parameter(torch.randn(self.memory_size, self.code_size))
        self.register_buffer('memory_logvar', torch.zeros(1))
        
        # Addressing variance
        self.w_logvar = nn.Parameter(torch.zeros(self.memory_size))
        
        # Observation noise
        self.observation_noise_std = 0.1
        
        # Sherman-Morrison approximation parameters
        self.register_buffer('ben_cohen_init', torch.tensor([-5.0]))
        self.pseudoinverse_steps = 3
        
        # W_M projection: C_mem -> (num_layers * num_heads * head_dim * 2)
        # Projects memory retrieval to KV space for all layers
        kv_total_dim = self.num_layers * self.num_heads * self.head_dim * 2
        self.W_M = nn.Linear(self.code_size, kv_total_dim)
        
        # Ordering: LSTM for sequential context
        self.use_ordering = True
        if self.use_ordering:
            self.lstm_z = nn.LSTM(
                input_size=self.code_size,
                hidden_size=self.code_size // 2,
                num_layers=2,
                bidirectional=True,
                batch_first=True
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize memory parameters"""
        # Initialize memory as identity-like
        nn.init.eye_(self.memory_mean)
        
        # Small variance for addressing
        nn.init.constant_(self.w_logvar, 0.0)
        
        # Initialize W_M projection
        nn.init.xavier_uniform_(self.W_M.weight)
        nn.init.zeros_(self.W_M.bias)
    
    def _get_prior_params(self):
        """Get prior distribution parameters"""
        dtype = self.memory_mean.dtype
        device = self.memory_mean.device
        logvar = torch.exp(self.memory_logvar).to(dtype=dtype, device=device)
        ones = torch.ones(self.memory_size, device=device, dtype=dtype)
        eps = torch.tensor(EPSILON, device=device, dtype=dtype)
        prior_var = ones * logvar + eps
        prior_cov = torch.diag(prior_var)
        return self.memory_mean, prior_cov
    
    def _get_prior_state(self, batch_size, dtype=None):
        """Initialize prior memory state for batch"""
        prior_mean, prior_cov = self._get_prior_params()
        
        # Convert to target dtype if specified
        if dtype is not None:
            prior_mean = prior_mean.to(dtype)
            prior_cov = prior_cov.to(dtype)
        
        batch_prior_mean = prior_mean.unsqueeze(0).expand(batch_size, -1, -1)
        batch_prior_cov = prior_cov.unsqueeze(0).expand(batch_size, -1, -1)
        
        return (batch_prior_mean, batch_prior_cov)
    
    def _solve_w_mean(self, z, M):
        """
        Solve for addressing weights using pseudo-inverse
        
        Args:
            z: (episode_size, batch_size, code_size)
            M: (batch_size, memory_size, code_size)
        
        Returns:
            w_mean: (episode_size, batch_size, memory_size)
        """
        z = z.transpose(0, 1)  # (batch_size, episode_size, code_size)
        
        # Ensure M matches z dtype for bmm compatibility
        M = M.to(z.dtype)
        M_pseudoinv = self._approx_pseudo_inverse(M)  # (batch_size, code_size, memory_size)
        
        # Add observation noise
        z_noise = z + torch.randn_like(z) * self.observation_noise_std
        
        w_mean = torch.bmm(z_noise, M_pseudoinv)  # (batch_size, episode_size, memory_size)
        w_mean = w_mean.transpose(0, 1)  # (episode_size, batch_size, memory_size)
        
        return w_mean
    
    def _approx_pseudo_inverse(self, A):
        """
        Approximate pseudo-inverse using Ben-Cohen iterative method
        
        Args:
            A: (batch_size, memory_size, code_size)
        
        Returns:
            A_pseudoinv: (batch_size, code_size, memory_size)
        """
        alpha = min(torch.exp(self.ben_cohen_init).item(), 5e-4)
        A_init = alpha * A
        A_pseudoinv = A_init.transpose(1, 2)  # (batch_size, code_size, memory_size)
        
        # Iterative refinement: A_inv = 2*A_inv - A_inv*A*A_inv
        for _ in range(self.pseudoinverse_steps):
            A_pseudoinv = 2 * A_pseudoinv - torch.bmm(
                torch.bmm(A_pseudoinv, A), A_pseudoinv
            )
        
        return A_pseudoinv
    
    def _update_memory(self, old_memory, w, z):
        """
        Update memory using Sherman-Morrison formula (Larimar Eq. 3)
        
        Args:
            old_memory: tuple (mean, cov)
            w: (1, batch_size, memory_size)
            z: (1, batch_size, code_size)
        
        Returns:
            new_memory: tuple (mean, cov)
        """
        old_mean, old_cov = old_memory
        
        # Ensure old_mean and old_cov match z dtype
        old_mean = old_mean.to(z.dtype)
        old_cov = old_cov.to(z.dtype)
        
        # Delta = z - w^T * M
        Delta = z - torch.bmm(w.transpose(0, 1), old_mean).transpose(0, 1)
        
        # wU = w^T * Cov
        wU = torch.bmm(w.transpose(0, 1), old_cov).transpose(0, 1)
        
        # sigma_z = wU * w^T + noise_var
        wUw = torch.bmm(wU.transpose(0, 1), w.transpose(0, 1).transpose(1, 2)).transpose(0, 1)
        sigma_z = wUw + self.observation_noise_std ** 2
        sigma_z = torch.clamp(sigma_z, min=EPSILON)
        
        # c_z = wU / sigma_z
        c_z = wU / sigma_z
        c_z = torch.clamp(c_z, min=-1e3, max=1e3)
        
        # Update mean: M_new = M_old + c_z^T * Delta
        Delta_clamped = torch.clamp(Delta, min=-1e2, max=1e2)
        posterior_mean = old_mean + torch.bmm(
            c_z.transpose(0, 1).transpose(1, 2), 
            Delta_clamped.transpose(0, 1)
        )
        posterior_mean = torch.clamp(posterior_mean, min=-1e3, max=1e3)
        
        # Update covariance: Cov_new = Cov_old - c_z^T * wU
        posterior_cov = old_cov - torch.bmm(
            c_z.transpose(0, 1).transpose(1, 2),
            wU.transpose(0, 1)
        )

        # Ensure covariance stays symmetric positive-definite
        posterior_cov = 0.5 * (posterior_cov + posterior_cov.transpose(-1, -2))
        
        # Clamp covariance diagonal to prevent extreme values
        diag_indices = torch.arange(self.memory_size, device=posterior_cov.device)
        posterior_cov[:, diag_indices, diag_indices] = torch.clamp(
            posterior_cov[:, diag_indices, diag_indices],
            min=1e-3,
            max=1e3
        )
        
        identity = torch.eye(
            self.memory_size,
            device=posterior_cov.device,
            dtype=posterior_cov.dtype
        ).unsqueeze(0)
        posterior_cov = posterior_cov + EPSILON * identity
        
        return (posterior_mean, posterior_cov)
    
    def write(self, z):
        """
        Write to memory (Larimar Eq. 3-4)
        
        Args:
            z: (episode_size, batch_size, code_size)
        
        Returns:
            posterior_memory: tuple (mean, cov)
            dkl_M: KL divergence
        """
        episode_size, batch_size = z.shape[:2]
        
        # Store original dtype and device
        original_dtype = z.dtype
        original_device = z.device
        
        # Apply ordering if enabled
        if self.use_ordering:
            # LSTM requires float32
            z_for_lstm = z.transpose(0, 1).float()  # (batch_size, episode_size, code_size)
            z, _ = self.lstm_z(z_for_lstm)  # (batch_size, episode_size, code_size)
            z = z.transpose(0, 1)  # (episode_size, batch_size, code_size)
            # Convert back to original dtype and ensure device consistency
            z = z.to(dtype=original_dtype, device=original_device)
        
        # Get prior with matching dtype
        prior_memory = self._get_prior_state(batch_size, dtype=original_dtype)
        
        # Sequential write using Sherman-Morrison updates
        new_memory = prior_memory
        for i in range(episode_size):
            z_step = z[i:i+1]  # (1, batch_size, code_size)
            w_step = self._solve_w_mean(z_step, new_memory[0])
            new_memory = self._update_memory(new_memory, w_step, z_step)
        
        # Compute KL divergence
        dkl_M = self._compute_kl_divergence(prior_memory, new_memory)
        
        return new_memory, dkl_M
    
    def read(self, z, memory_state, deterministic=False):
        """
        Read from memory (Larimar Eq. 4)
        
        Args:
            z: (episode_size, batch_size, code_size)
            memory_state: tuple (mean, cov)
            deterministic: if True, use mean addressing without noise
        
        Returns:
            z_retrieved: (episode_size, batch_size, code_size)
            dkl_w: KL divergence of addressing weights
        """
        episode_size, batch_size = z.shape[:2]
        
        M = memory_state[0]  # (batch_size, memory_size, code_size)
        
        # Store original dtype for consistency
        original_dtype = z.dtype
        
        # Ensure M matches z dtype
        M = M.to(z.dtype)
        
        # Compute addressing weights
        w_mean = self._solve_w_mean(z, M)
        
        # Sample or use deterministic weights
        if deterministic:
            w = w_mean
        else:
            # Ensure w_logvar matches w_mean dtype for sampling
            w_logvar = self.w_logvar.unsqueeze(0).unsqueeze(0).to(w_mean.dtype)  # (1, 1, memory_size)
            w_logvar = torch.clamp(w_logvar, min=-10.0, max=10.0)
            std = torch.exp(0.5 * w_logvar)
            w = w_mean + std * torch.randn_like(w_mean)
        
        # Retrieve: z = w^T * M
        z_retrieved = torch.bmm(w.transpose(0, 1), M).transpose(0, 1)
        
        # Add observation noise with correct dtype
        z_retrieved = z_retrieved + self.observation_noise_std * torch.randn_like(z_retrieved)
        
        # Ensure retrieved tensor maintains original dtype
        z_retrieved = z_retrieved.to(original_dtype)
        
        # Compute KL divergence
        dkl_w = self._compute_addressing_kl(w_mean)
        
        return z_retrieved, dkl_w
    
    def _compute_kl_divergence(self, prior_memory, posterior_memory):
        """Compute KL divergence between prior and posterior memory"""
        R_prior, U_prior = prior_memory
        R, U = posterior_memory
        
        # Ensure dtype consistency for KL computation
        R = R.to(R_prior.dtype)
        U = U.to(U_prior.dtype)
        
        p_diag = torch.diagonal(U_prior, dim1=-2, dim2=-1)
        q_diag = torch.diagonal(U, dim1=-2, dim2=-1)

        # Debug: Check for problematic values before clamping
        if torch.any(torch.isinf(p_diag)) or torch.any(torch.isinf(q_diag)):
            print(f"⚠️ Inf detected in covariance diagonals before clamping")
            print(f"  p_diag range: [{p_diag.min().item()}, {p_diag.max().item()}]")
            print(f"  q_diag range: [{q_diag.min().item()}, {q_diag.max().item()}]")
        
        if torch.any(p_diag < EPSILON) or torch.any(q_diag < EPSILON):
            print(f"⚠️ Very small diagonal values detected")
            print(f"  p_diag min: {p_diag.min().item()}")
            print(f"  q_diag min: {q_diag.min().item()}")

        # Clamp diagonals to avoid division by zero or log of non-positive values
        p_diag = torch.clamp(p_diag, min=1e-3, max=1e6)
        q_diag = torch.clamp(q_diag, min=1e-3, max=1e6)
        ratio = q_diag / p_diag
        ratio = torch.clamp(ratio, min=EPSILON, max=1e3)

        t1 = self.code_size * torch.sum(ratio, dim=-1)
        t1 = torch.clamp(t1, min=-1e6, max=1e6)
        
        # Clamp the squared difference to prevent explosion
        diff_sq = (R - R_prior) ** 2
        diff_sq = torch.clamp(diff_sq, max=1e3)
        t2 = torch.sum(diff_sq / p_diag.unsqueeze(-1), dim=[-2, -1])
        t2 = torch.clamp(t2, min=-1e6, max=1e6)
        
        t3 = -self.code_size * self.memory_size
        
        log_term = torch.log(p_diag) - torch.log(q_diag)
        log_term = torch.clamp(log_term, min=-10.0, max=10.0)
        t4 = self.code_size * torch.sum(log_term, dim=-1)
        t4 = torch.clamp(t4, min=-1e6, max=1e6)
        
        dkl_M = torch.mean(t1 + t2 + t3 + t4)
        
        # Debug: Check final value
        if torch.isinf(dkl_M) or torch.isnan(dkl_M):
            print(f"⚠️ Invalid KL divergence computed")
            print(f"  t1: {t1.mean().item()}")
            print(f"  t2: {t2.mean().item()}")
            print(f"  t3: {t3}")
            print(f"  t4: {t4.mean().item()}")
            print(f"  dkl_M: {dkl_M.item()}")
        
        dkl_M = torch.clamp(dkl_M, min=-1e6, max=1e6)
        return dkl_M
    
    def _compute_addressing_kl(self, w_mean):
        """Compute KL divergence of addressing weights"""
        w_mean = torch.clamp(w_mean, min=-1e3, max=1e3)
        w_logvar = self.w_logvar.unsqueeze(0).unsqueeze(0).to(w_mean.dtype)
        w_logvar = torch.clamp(w_logvar, min=-10.0, max=10.0)
        
        exp_term = torch.exp(w_logvar)
        exp_term = torch.clamp(exp_term, max=1e3)
        
        dkl = 0.5 * (exp_term + w_mean ** 2 - 1 - w_logvar)
        dkl = torch.clamp(dkl, min=0.0, max=1e3)
        dkl = torch.sum(dkl, dim=-1)
        dkl = torch.clamp(dkl, min=0.0, max=1e6)
        return dkl
    
    def project_to_kv(self, z_retrieved):
        """
        Project retrieved memory to KV space using W_M
        
        Args:
            z_retrieved: (episode_size, batch_size, code_size)
        
        Returns:
            kv_dict: dictionary mapping layer_idx -> (K, V) tuples
        """
        # Flatten episode and batch dimensions
        original_shape = z_retrieved.shape[:2]
        z_flat = z_retrieved.reshape(-1, self.code_size)

        # Ensure dtype/device match projection weights
        weight = self.W_M.weight
        if z_flat.dtype != weight.dtype or z_flat.device != weight.device:
            z_flat = z_flat.to(device=weight.device, dtype=weight.dtype)

        # Project: (N, kv_total_dim)
        kv_projected = self.W_M(z_flat)
        
        # Reshape: (N, num_layers, num_heads, head_dim, 2)
        kv_reshaped = kv_projected.view(
            -1, self.num_layers, self.num_heads, self.head_dim, 2
        )
        
        # Split into K and V
        kv_dict = {}
        for layer_idx in range(self.num_layers):
            K = kv_reshaped[:, layer_idx, :, :, 0]  # (N, num_heads, head_dim)
            V = kv_reshaped[:, layer_idx, :, :, 1]  # (N, num_heads, head_dim)
            
            # Reshape back to (episode_size, batch_size, num_heads, head_dim)
            K = K.view(*original_shape, self.num_heads, self.head_dim)
            V = V.view(*original_shape, self.num_heads, self.head_dim)
            
            kv_dict[layer_idx] = (K, V)
        
        return kv_dict


class ScopeDetector(nn.Module):
    """
    Larimar-style ScopeNet for memory injection decisions
    Lightweight MLP classifier
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.input_dim = config.get('qwen_hidden_dim', 896)
        self.hidden_dim = config.get('scope_hidden_dim', 256)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, context_embedding):
        """
        Determine whether to inject memory
        
        Args:
            context_embedding: (batch_size, hidden_dim)
        
        Returns:
            injection_prob: (batch_size, 1) probability of injecting memory
        """
        current_param = next(self.mlp.parameters())
        if (context_embedding.dtype != current_param.dtype or
                context_embedding.device != current_param.device):
            self.mlp = self.mlp.to(device=context_embedding.device,
                                   dtype=context_embedding.dtype)
        
        return self.mlp(context_embedding)
