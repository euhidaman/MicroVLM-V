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


def _sanitize_tensor(t: torch.Tensor, min_val: float = -1e6, max_val: float = 1e6) -> torch.Tensor:
    """Clamp tensor to finite range and replace NaN/Inf with safe defaults."""
    t = torch.clamp(t, min=min_val, max=max_val)
    return torch.nan_to_num(t, nan=0.0, posinf=max_val, neginf=min_val)


class EpisodicMemory(nn.Module):
    """
    Larimar-style episodic memory with Gaussian Process Memory
    Implements write/read operations with Sherman-Morrison updates
    
    NOTE: Dimensions reduced for compact model (<1GB target):
    - memory_size: 512 -> 64 (8x reduction)
    - W_M targets only 6 layers with 4 heads (vs 24 layers, 14 heads)
    
    w_logvar_setting options (from Larimar):
    - 0: Single scalar variance for all dimensions and inputs (default, most lightweight)
    - 1: Per-dimension variance, same for all inputs (lightweight)
    - 2: Variance is a function of z (input-dependent, small linear layer)
    - 3: Variance is a function of memory M (memory-dependent, small linear layer)
    - 4: Variance is a function of both z and M (most expressive, small linear layer)
    """
    
    def __init__(self, config):
        super().__init__()
        
        # Reduced memory dimensions for compact model
        self.memory_size = getattr(config, 'memory_size', 64)  # K_mem (reduced from 512)
        self.code_size = getattr(config, 'memory_dim', 896)  # C_mem (matches Qwen hidden)
        self.qwen_hidden_dim = getattr(config, 'qwen_hidden_dim', 896)

        # Validate dimension match - code_size must match language hidden for fused embeddings
        lang_hidden = getattr(config, 'language_hidden_size', 896)
        if self.code_size != lang_hidden:
            print(f"⚠️ EpisodicMemory: code_size ({self.code_size}) != language_hidden ({lang_hidden})")
            print(f"   Forcing code_size = {lang_hidden} to avoid dimension mismatch")
            self.code_size = lang_hidden
        
        # W_M projection targets fewer layers/heads for size reduction
        self.num_layers = getattr(config, 'memory_num_layers', 6)  # Reduced from 24
        self.num_heads = getattr(config, 'memory_num_heads', 4)    # Reduced from 14
        self.head_dim = getattr(config, 'head_dim', 64)
        self.compute_dtype = torch.float32
        
        # Memory parameters: M of shape (K_mem, C_mem)
        self.memory_mean = nn.Parameter(torch.randn(self.memory_size, self.code_size))
        self.register_buffer('memory_logvar', torch.zeros(1))
        
        # ===== Larimar-style w_logvar_setting configuration =====
        # Controls how addressing weight variance is computed
        self._w_logvar_setting = getattr(config, 'w_logvar_setting', 1)  # Default: setting 1 (per-dim)
        self.deterministic = getattr(config, 'deterministic_memory', False)

        if self._w_logvar_setting == 0:
            # Setting 0: Single scalar variance (most lightweight)
            self.w_logvar = nn.Parameter(torch.zeros(1))
        elif self._w_logvar_setting == 1:
            # Setting 1: Per-dimension variance (lightweight, default)
            self.w_logvar = nn.Parameter(torch.zeros(self.memory_size))
        elif self._w_logvar_setting == 2:
            # Setting 2: Variance is a function of z (input-dependent)
            # Uses small linear layer for compact model
            self.w_logvar = nn.Linear(self.code_size, self.memory_size)
            nn.init.xavier_uniform_(self.w_logvar.weight, gain=0.1)
            nn.init.zeros_(self.w_logvar.bias)
        elif self._w_logvar_setting == 3:
            # Setting 3: Variance is a function of memory M
            # Flatten memory and project (uses small linear layer)
            self.w_logvar = nn.Linear(self.code_size * self.memory_size, self.memory_size)
            nn.init.xavier_uniform_(self.w_logvar.weight, gain=0.1)
            nn.init.zeros_(self.w_logvar.bias)
        elif self._w_logvar_setting == 4:
            # Setting 4: Variance is a function of both z and M (most expressive)
            # Takes concatenated z and flattened M
            self.w_logvar = nn.Linear(self.code_size + self.code_size * self.memory_size, self.memory_size)
            nn.init.xavier_uniform_(self.w_logvar.weight, gain=0.1)
            nn.init.zeros_(self.w_logvar.bias)
        else:
            raise ValueError(f"Invalid w_logvar_setting: {self._w_logvar_setting}. Must be 0-4.")
        
        # Observation noise
        self.observation_noise_std = 0.1
        
        # Sherman-Morrison approximation parameters
        self.register_buffer('ben_cohen_init', torch.tensor([-5.0]))
        self.pseudoinverse_steps = 3
        
        # W_M projection: C_mem -> (num_layers * num_heads * head_dim * 2)
        # Projects memory retrieval to KV space for all layers
        kv_total_dim = self.num_layers * self.num_heads * self.head_dim * 2
        self.W_M = nn.Linear(self.code_size, kv_total_dim)
        
        # Ordering: LSTM for sequential context (reduced for compact model)
        self.use_ordering = True
        if self.use_ordering:
            lstm_hidden = self.code_size // 4  # Reduced from //2 for compact model
            lstm_output_dim = lstm_hidden * 2  # Bidirectional doubles the output
            self.lstm_z = nn.LSTM(
                input_size=self.code_size,
                hidden_size=lstm_hidden,
                num_layers=1,  # Reduced from 2 for compact model
                bidirectional=True,
                batch_first=True
            )
            # Project LSTM output back to code_size
            self.lstm_proj = nn.Linear(lstm_output_dim, self.code_size)
            nn.init.xavier_uniform_(self.lstm_proj.weight)
            nn.init.zeros_(self.lstm_proj.bias)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize memory parameters"""
        # Initialize memory as identity-like
        nn.init.eye_(self.memory_mean)
        
        # Initialize w_logvar based on setting type
        if self._w_logvar_setting in [0, 1]:
            # Scalar or per-dimension parameter
            nn.init.constant_(self.w_logvar, 0.0)
        # Settings 2-4 use Linear layers, already initialized in __init__
        
        # Initialize W_M projection
        nn.init.xavier_uniform_(self.W_M.weight)
        nn.init.zeros_(self.W_M.bias)
    
    def _compute_w_logvar(self, z: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """
        Compute w_logvar based on the configured setting (Larimar-style).
        
        Args:
            z: (episode_size, batch_size, code_size) - input encoding
            M: (batch_size, memory_size, code_size) - current memory
        
        Returns:
            w_logvar: variance for addressing weights, shape depends on setting
        """
        episode_size, batch_size, _ = z.shape
        
        if self._w_logvar_setting == 0:
            # Setting 0: Single scalar for all
            return self.w_logvar  # Shape: (1,)
        
        elif self._w_logvar_setting == 1:
            # Setting 1: Per-dimension, same for all inputs
            return self.w_logvar  # Shape: (memory_size,)
        
        elif self._w_logvar_setting == 2:
            # Setting 2: Function of z
            z_flat = z.reshape(episode_size * batch_size, self.code_size)
            w_logvar = self.w_logvar(z_flat)  # (episode_size * batch_size, memory_size)
            return w_logvar.view(episode_size, batch_size, self.memory_size)
        
        elif self._w_logvar_setting == 3:
            # Setting 3: Function of memory M
            M_flat = M.reshape(batch_size, -1)  # (batch_size, memory_size * code_size)
            w_logvar = self.w_logvar(M_flat)  # (batch_size, memory_size)
            return w_logvar.unsqueeze(0).expand(episode_size, -1, -1)
        
        elif self._w_logvar_setting == 4:
            # Setting 4: Function of both z and M
            M_flat = M.reshape(batch_size, -1)  # (batch_size, memory_size * code_size)
            # Expand M to match episode_size
            M_expanded = M_flat.unsqueeze(0).expand(episode_size, -1, -1)
            M_expanded = M_expanded.reshape(episode_size * batch_size, -1)
            z_flat = z.reshape(episode_size * batch_size, self.code_size)
            # Concatenate z and M
            combined = torch.cat([z_flat, M_expanded], dim=-1)
            w_logvar = self.w_logvar(combined)  # (episode_size * batch_size, memory_size)
            return w_logvar.view(episode_size, batch_size, self.memory_size)
        
        else:
            raise ValueError(f"Invalid w_logvar_setting: {self._w_logvar_setting}")
    
    def _sample_w(self, w_mean: torch.Tensor, w_logvar: torch.Tensor) -> torch.Tensor:
        """
        Sample addressing weights from Gaussian distribution (Larimar-style).
        
        Args:
            w_mean: (episode_size, batch_size, memory_size)
            w_logvar: variance, shape depends on w_logvar_setting
        
        Returns:
            w: sampled addressing weights (episode_size, batch_size, memory_size)
        """
        w_mean = _sanitize_tensor(w_mean, min_val=-1e3, max_val=1e3)
        std = torch.exp(0.5 * torch.clamp(w_logvar, min=-10.0, max=10.0))
        std = _sanitize_tensor(std, min_val=1e-6, max_val=1e3)

        # Handle different shapes of w_logvar
        if self._w_logvar_setting == 0:
            # Single scalar - broadcast to w_mean shape
            std = std.expand_as(w_mean)
        elif self._w_logvar_setting == 1:
            # Per-dimension - broadcast to (episode_size, batch_size, memory_size)
            std = std.view(1, 1, -1).expand_as(w_mean)
        # Settings 2-4 already have correct shape
        
        # Reparameterization trick
        w_sample = w_mean + std * torch.randn_like(w_mean)
        return _sanitize_tensor(w_sample, min_val=-1e3, max_val=1e3)

    def _get_prior_params(self):
        """Get prior distribution parameters"""
        device = self.memory_mean.device
        dtype = self.compute_dtype
        mean = self.memory_mean.to(dtype=dtype, device=device)
        logvar = torch.exp(self.memory_logvar.to(dtype=dtype, device=device))
        ones = torch.ones(self.memory_size, device=device, dtype=dtype)
        eps = torch.tensor(EPSILON, device=device, dtype=dtype)
        prior_var = ones * logvar + eps
        prior_cov = torch.diag(prior_var)
        return mean, prior_cov
    
    def _get_prior_state(self, batch_size, dtype=None):
        """Initialize prior memory state for batch"""
        prior_mean, prior_cov = self._get_prior_params()
        
        # Convert to target dtype if specified, otherwise use compute dtype
        target_dtype = dtype or self.compute_dtype
        prior_mean = prior_mean.to(target_dtype)
        prior_cov = prior_cov.to(target_dtype)
        
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
        z_noise = _sanitize_tensor(z_noise, min_val=-1e2, max_val=1e2)

        w_mean = torch.bmm(z_noise, M_pseudoinv)  # (batch_size, episode_size, memory_size)
        w_mean = w_mean.transpose(0, 1)  # (episode_size, batch_size, memory_size)
        
        return _sanitize_tensor(w_mean, min_val=-1e3, max_val=1e3)

    def _approx_pseudo_inverse(self, A):
        """
        Approximate pseudo-inverse using Ben-Cohen iterative method
        
        Args:
            A: (batch_size, memory_size, code_size)
        
        Returns:
            A_pseudoinv: (batch_size, code_size, memory_size)
        """
        # Sanitize input
        A = _sanitize_tensor(A, min_val=-1e2, max_val=1e2)

        alpha = min(torch.exp(self.ben_cohen_init).item(), 5e-4)
        A_init = alpha * A
        A_pseudoinv = A_init.transpose(1, 2)  # (batch_size, code_size, memory_size)
        
        # Iterative refinement: A_inv = 2*A_inv - A_inv*A*A_inv
        # Use inplace operations and delete intermediate tensors to save memory
        for _ in range(self.pseudoinverse_steps):
            temp = torch.bmm(A_pseudoinv, A)
            temp = _sanitize_tensor(temp, min_val=-1e2, max_val=1e2)
            temp2 = torch.bmm(temp, A_pseudoinv)
            del temp  # Free memory immediately
            temp2 = _sanitize_tensor(temp2, min_val=-1e2, max_val=1e2)
            A_pseudoinv = 2 * A_pseudoinv - temp2
            del temp2  # Free memory immediately
            A_pseudoinv = _sanitize_tensor(A_pseudoinv, min_val=-1e2, max_val=1e2)

        # Clear CUDA cache to prevent fragmentation
        if A.is_cuda:
            torch.cuda.empty_cache()

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
        Delta_clamped = _sanitize_tensor(Delta, min_val=-1e2, max_val=1e2)
        posterior_mean = old_mean + torch.bmm(
            c_z.transpose(0, 1).transpose(1, 2), 
            Delta_clamped.transpose(0, 1)
        )
        posterior_mean = _sanitize_tensor(posterior_mean, min_val=-1e3, max_val=1e3)

        # Update covariance: Cov_new = Cov_old - c_z^T * wU
        posterior_cov = old_cov - torch.bmm(
            c_z.transpose(0, 1).transpose(1, 2),
            wU.transpose(0, 1)
        )

        # Ensure covariance stays symmetric positive-definite
        posterior_cov = 0.5 * (posterior_cov + posterior_cov.transpose(-1, -2))
        posterior_cov = _sanitize_tensor(posterior_cov, min_val=-1e3, max_val=1e3)

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

        # Sanitize input before processing
        z = _sanitize_tensor(z.to(self.compute_dtype), min_val=-1e2, max_val=1e2)

        # Apply ordering if enabled
        if self.use_ordering:
            # LSTM requires float32 and sanitized input
            z_for_lstm = z.transpose(0, 1).float()  # (batch_size, episode_size, code_size)
            z_for_lstm = _sanitize_tensor(z_for_lstm, min_val=-1e2, max_val=1e2)

            # LSTM forward with gradient checkpointing for stability
            z_lstm, _ = self.lstm_z(z_for_lstm)  # (batch_size, episode_size, lstm_output_dim)

            # Sanitize LSTM output before projection
            z_lstm = _sanitize_tensor(z_lstm, min_val=-1e2, max_val=1e2)

            z_lstm = self.lstm_proj(z_lstm)  # Project back to (batch_size, episode_size, code_size)

            # Sanitize after projection
            z_lstm = _sanitize_tensor(z_lstm, min_val=-1e2, max_val=1e2)

            z = z_lstm.transpose(0, 1)  # (episode_size, batch_size, code_size)
            z = z.to(self.compute_dtype)
        
        # Get prior with matching dtype
        prior_memory = self._get_prior_state(batch_size, dtype=self.compute_dtype)
        
        # Sequential write using Sherman-Morrison updates
        new_memory = prior_memory
        for i in range(episode_size):
            z_step = z[i:i+1]  # (1, batch_size, code_size)
            z_step = _sanitize_tensor(z_step, min_val=-1e2, max_val=1e2)
            w_step = self._solve_w_mean(z_step, new_memory[0])
            new_memory = self._update_memory(new_memory, w_step, z_step)
        
        # Compute KL divergence with safety
        dkl_M = self._compute_kl_divergence(prior_memory, new_memory)
        
        # Final sanitization of KL
        dkl_M = _sanitize_tensor(dkl_M, min_val=0.0, max_val=1e3)

        return new_memory, dkl_M.to(original_dtype)
    
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
        
        # Perform memory operations in compute dtype with sanitization
        z = _sanitize_tensor(z.to(self.compute_dtype), min_val=-1e2, max_val=1e2)
        M = _sanitize_tensor(M.to(self.compute_dtype), min_val=-1e2, max_val=1e2)
        cov = _sanitize_tensor(memory_state[1].to(self.compute_dtype), min_val=1e-4, max_val=1e3)
        memory_state = (M, cov)
        
        # Compute addressing weights
        w_mean = self._solve_w_mean(z, M)
        
        # Sample or use deterministic weights (using Larimar-style w_logvar)
        if deterministic or self.deterministic:
            w = w_mean
        else:
            # Compute w_logvar based on setting (Larimar-style)
            w_logvar = self._compute_w_logvar(z, M)
            w_logvar = torch.clamp(w_logvar, min=-10.0, max=10.0)
            # Sample using reparameterization trick
            w = self._sample_w(w_mean, w_logvar.to(w_mean.dtype))
        
        # Sanitize w before retrieval
        w = _sanitize_tensor(w, min_val=-1e2, max_val=1e2)

        # Retrieve: z = w^T * M
        z_retrieved = torch.bmm(w.transpose(0, 1), M).transpose(0, 1)
        
        # Sanitize retrieved values
        z_retrieved = _sanitize_tensor(z_retrieved, min_val=-1e2, max_val=1e2)

        # Add observation noise with correct dtype (scaled down)
        noise = 0.01 * torch.randn_like(z_retrieved)  # Reduced noise
        z_retrieved = z_retrieved + noise

        # Final sanitization
        z_retrieved = _sanitize_tensor(z_retrieved.to(original_dtype), min_val=-1e2, max_val=1e2)

        # Compute KL divergence (with w_logvar for accurate computation)
        if deterministic or self.deterministic:
            dkl_w = self._compute_addressing_kl(w_mean)
        else:
            dkl_w = self._compute_addressing_kl(w_mean, w_logvar)
        
        # Sanitize KL output
        dkl_w = _sanitize_tensor(dkl_w, min_val=0.0, max_val=1e3)

        return z_retrieved, dkl_w.to(original_dtype)
    
    def _compute_kl_divergence(self, prior_memory, posterior_memory):
        """Compute KL divergence between prior and posterior memory"""
        R_prior, U_prior = prior_memory
        R, U = posterior_memory
        
        # Ensure dtype consistency for KL computation
        R = R.to(R_prior.dtype)
        U = U.to(U_prior.dtype)
        
        p_diag = _sanitize_tensor(torch.diagonal(U_prior, dim1=-2, dim2=-1), min_val=1e-4, max_val=1e4)
        q_diag = _sanitize_tensor(torch.diagonal(U, dim1=-2, dim2=-1), min_val=1e-4, max_val=1e4)
        ratio = _sanitize_tensor(q_diag / p_diag, min_val=1e-4, max_val=1e4)

        t1 = self.code_size * torch.sum(ratio, dim=-1)
        t1 = torch.clamp(t1, min=-1e6, max=1e6)
        
        # Clamp the squared difference to prevent explosion
        diff_sq = _sanitize_tensor((R - R_prior) ** 2, min_val=0.0, max_val=1e3)
        t2 = torch.sum(diff_sq / p_diag.unsqueeze(-1), dim=[-2, -1])
        t2 = torch.clamp(t2, min=-1e6, max=1e6)
        
        t3 = -self.code_size * self.memory_size
        
        log_term = _sanitize_tensor(torch.log(p_diag) - torch.log(q_diag), min_val=-10.0, max_val=10.0)
        t4 = self.code_size * torch.sum(log_term, dim=-1)
        t4 = torch.clamp(t4, min=-1e6, max=1e6)
        
        dkl_M = torch.mean(t1 + t2 + t3 + t4)
        
        # Debug: Check for NaN or Inf
        if torch.isnan(dkl_M) or torch.isinf(dkl_M):
            print("⚠️ Invalid KL divergence detected – forcing to zero.")
            dkl_M = torch.zeros_like(dkl_M)

        return torch.clamp(dkl_M, min=-1e3, max=1e3)

    def _compute_addressing_kl(self, w_mean, w_logvar=None):
        """
        Compute KL divergence of addressing weights (Larimar-style).
        
        KL(q(w) || p(w)) where p(w) = N(0, I) is the prior.
        
        Args:
            w_mean: (episode_size, batch_size, memory_size)
            w_logvar: variance, shape depends on w_logvar_setting
                     If None, uses self.w_logvar (for backward compatibility)
        """
        w_mean = torch.clamp(w_mean, min=-1e3, max=1e3)
        
        # Get w_logvar if not provided
        if w_logvar is None:
            if self._w_logvar_setting in [0, 1]:
                w_logvar = self.w_logvar.to(w_mean.dtype)
            else:
                # For settings 2-4, need z and M which we don't have here
                # Use zeros as fallback (equivalent to unit variance)
                w_logvar = torch.zeros(self.memory_size, device=w_mean.device, dtype=w_mean.dtype)
        
        w_logvar = torch.clamp(w_logvar, min=-10.0, max=10.0)
        
        # Handle different shapes based on w_logvar_setting
        if self._w_logvar_setting == 0:
            # Scalar: broadcast to full shape
            w_logvar_expanded = w_logvar.expand_as(w_mean)
        elif self._w_logvar_setting == 1:
            # Per-dimension: reshape for broadcasting
            w_logvar_expanded = w_logvar.view(1, 1, -1).expand_as(w_mean)
        else:
            # Settings 2-4: already have correct shape or use provided w_logvar
            if w_logvar.dim() == 1:
                w_logvar_expanded = w_logvar.view(1, 1, -1).expand_as(w_mean)
            else:
                w_logvar_expanded = w_logvar
        
        exp_term = _sanitize_tensor(torch.exp(w_logvar_expanded), min_val=1e-6, max_val=1e3)
        kl_core = _sanitize_tensor(exp_term + w_mean ** 2 - 1 - w_logvar_expanded, min_val=-1e3, max_val=1e3)
        dkl = 0.5 * kl_core
        dkl = _sanitize_tensor(dkl, min_val=0.0, max_val=1e3)
        dkl = torch.sum(dkl, dim=-1)
        return _sanitize_tensor(dkl, min_val=0.0, max_val=1e4)

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
    Lightweight MLP classifier (reduced for compact model)
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.input_dim = getattr(config, 'qwen_hidden_dim', 896)
        self.hidden_dim = getattr(config, 'scope_hidden_dim', 64)  # Reduced from 256

        # Simplified 2-layer MLP (reduced from 3 layers)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
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
