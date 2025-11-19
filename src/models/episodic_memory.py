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
        prior_var = torch.ones(self.memory_size, device=self.memory_mean.device, dtype=self.memory_mean.dtype) * \
                   torch.exp(self.memory_logvar) + EPSILON
        prior_cov = torch.diag(prior_var)
        return self.memory_mean, prior_cov
    
    def _get_prior_state(self, batch_size):
        """Initialize prior memory state for batch"""
        prior_mean, prior_cov = self._get_prior_params()
        
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
        
        # Delta = z - w^T * M
        Delta = z - torch.bmm(w.transpose(0, 1), old_mean).transpose(0, 1)
        
        # wU = w^T * Cov
        wU = torch.bmm(w.transpose(0, 1), old_cov).transpose(0, 1)
        
        # sigma_z = wU * w^T + noise_var
        wUw = torch.bmm(wU.transpose(0, 1), w.transpose(0, 1).transpose(1, 2)).transpose(0, 1)
        sigma_z = wUw + self.observation_noise_std ** 2
        
        # c_z = wU / sigma_z
        c_z = wU / sigma_z
        
        # Update mean: M_new = M_old + c_z^T * Delta
        posterior_mean = old_mean + torch.bmm(
            c_z.transpose(0, 1).transpose(1, 2), 
            Delta.transpose(0, 1)
        )
        
        # Update covariance: Cov_new = Cov_old - c_z^T * wU
        posterior_cov = old_cov - torch.bmm(
            c_z.transpose(0, 1).transpose(1, 2),
            wU.transpose(0, 1)
        )
        
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
        
        # Get prior
        prior_memory = self._get_prior_state(batch_size)
        
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
        
        # Compute addressing weights
        w_mean = self._solve_w_mean(z, M)
        
        # Sample or use deterministic weights
        if deterministic:
            w = w_mean
        else:
            w_logvar = self.w_logvar.unsqueeze(0).unsqueeze(0)  # (1, 1, memory_size)
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
        
        p_diag = torch.diagonal(U_prior, dim1=-2, dim2=-1)
        q_diag = torch.diagonal(U, dim1=-2, dim2=-1)
        
        t1 = self.code_size * torch.sum(q_diag / p_diag, dim=-1)
        t2 = torch.sum((R - R_prior) ** 2 / p_diag.unsqueeze(-1), dim=[-2, -1])
        t3 = -self.code_size * self.memory_size
        t4 = self.code_size * torch.sum(torch.log(p_diag) - torch.log(q_diag), dim=-1)
        
        dkl_M = torch.mean(t1 + t2 + t3 + t4)
        return dkl_M
    
    def _compute_addressing_kl(self, w_mean):
        """Compute KL divergence of addressing weights"""
        w_logvar = self.w_logvar.unsqueeze(0).unsqueeze(0)
        dkl = 0.5 * (torch.exp(w_logvar) + w_mean ** 2 - 1 - w_logvar)
        return torch.sum(dkl, dim=-1)
    
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
        return self.mlp(context_embedding)
