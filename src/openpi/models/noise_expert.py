"""
Noise Expert – Gemma 300M architecture (Flax Linen).

Learns (mu, log_sigma) for both coarse and fine action noise.
Used via nnx_bridge.ToNNX in the NNX model.

noise = mu + exp(0.5 * log_sigma) * randn

Placed in: src/openpi/models/noise_expert.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import flax.linen as nn
import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class NoiseExpertConfig:
    action_dim: int = 32
    coarse_action_horizon: int = 30
    action_horizon: int = 30
    width: int = 1024
    depth: int = 18
    mlp_dim: int = 4096
    num_heads: int = 8
    head_dim: int = 256
    num_kv_heads: int = 1
    dtype: str = "bfloat16"


class RMSNorm(nn.Module):
    @nn.compact
    def __call__(self, x):
        dtype = x.dtype
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        normed = x * jnp.reciprocal(jnp.sqrt(var + 1e-06))
        scale = self.param("scale", nn.initializers.zeros, (x.shape[-1],))
        return (normed * (1 + scale)).astype(dtype)


class Attention(nn.Module):
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    width: int = 1024
    dtype: str = "bfloat16"

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        dtype = getattr(jnp, self.dtype)
        b, s, _ = x.shape

        qkv = nn.Dense(
            (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
            dtype=dtype,
            name="qkv_einsum",
        )(x)

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        q, k, v = jnp.split(qkv, [q_size, q_size + kv_size], axis=-1)

        q = q.reshape(b, s, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, s, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        if self.num_kv_heads != self.num_heads:
            k = jnp.repeat(k, self.num_heads // self.num_kv_heads, axis=1)
            v = jnp.repeat(v, self.num_heads // self.num_kv_heads, axis=1)

        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        if mask is not None:
            # Defensive: ensure mask is 4D (B, 1, Q, K) for broadcasting with attn (B, H, Q, K)
            while mask.ndim > 4:
                mask = mask.squeeze(0)
            if mask.ndim == 2:
                mask = mask[:, None, :, None]  # (B, S) -> (B, 1, S, S)
            elif mask.ndim == 3:
                mask = mask[:, None, :, :]  # (B, Q, K) -> (B, 1, Q, K)
            elif mask.ndim == 4 and mask.shape[1] != 1:
                # Already 4D but head dim is not 1 — insert axis
                mask = mask[:, None, :, :]
            attn = jnp.where(mask, attn, -2.3819763e38)
        attn = jax.nn.softmax(attn, axis=-1)

        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, self.num_heads * self.head_dim)
        return nn.Dense(self.width, dtype=dtype, name="attn_vec_einsum")(out)


class FeedForward(nn.Module):
    hidden_dim: int = 4096
    width: int = 1024
    dtype: str = "bfloat16"

    @nn.compact
    def __call__(self, x):
        dtype = getattr(jnp, self.dtype)
        gating = nn.Dense(2 * self.hidden_dim, dtype=dtype, name="gating_einsum")(x)
        gate, value = jnp.split(gating, 2, axis=-1)
        return nn.Dense(self.width, dtype=dtype, name="linear")(nn.gelu(gate) * value)


class TransformerBlock(nn.Module):
    width: int = 1024
    mlp_dim: int = 4096
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    dtype: str = "bfloat16"

    @nn.compact
    def __call__(self, x, mask=None, deterministic=True):
        h = Attention(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            width=self.width,
            dtype=self.dtype,
            name="attn",
        )(RMSNorm(name="pre_attention_norm")(x), mask=mask, deterministic=deterministic)
        x = x + h
        h = FeedForward(
            hidden_dim=self.mlp_dim,
            width=self.width,
            dtype=self.dtype,
            name="mlp",
        )(RMSNorm(name="pre_ffw_norm")(x))
        return x + h


class NoiseExpertTransformer(nn.Module):
    config: NoiseExpertConfig

    def setup(self):
        cfg = self.config
        self.layers = [
            TransformerBlock(
                width=cfg.width,
                mlp_dim=cfg.mlp_dim,
                num_heads=cfg.num_heads,
                num_kv_heads=cfg.num_kv_heads,
                head_dim=cfg.head_dim,
                dtype=cfg.dtype,
                name=f"block_{i}",
            )
            for i in range(cfg.depth)
        ]
        self.final_norm = RMSNorm(name="final_norm")

    def __call__(self, x, mask=None, deterministic=True):
        for layer in self.layers:
            x = layer(x, mask=mask, deterministic=deterministic)
        return self.final_norm(x)


class NoiseExpert(nn.Module):
    """
    Predicts (mu, log_sigma) for coarse and fine action noise.

    Input: prefix tokens (B, S, D) + mask (B, S)
    Output: mu_coarse, log_sigma_coarse, mu_fine, log_sigma_fine
    """

    config: NoiseExpertConfig

    def setup(self):
        cfg = self.config
        dtype = getattr(jnp, cfg.dtype)

        self.input_proj = nn.Dense(cfg.width, dtype=dtype, name="input_proj")
        self.transformer = NoiseExpertTransformer(config=cfg)

        total_coarse = cfg.coarse_action_horizon * cfg.action_dim
        total_fine = cfg.action_horizon * cfg.action_dim

        self.feature_proj = nn.Sequential(
            [
                nn.Dense(cfg.width, dtype=dtype),
                nn.gelu,
                nn.Dense(cfg.width, dtype=dtype),
                nn.gelu,
            ]
        )

        self.coarse_mu_head = nn.Dense(total_coarse, dtype=dtype, name="coarse_mu")
        self.coarse_log_sigma_head = nn.Dense(total_coarse, dtype=dtype, name="coarse_log_sigma")
        self.fine_mu_head = nn.Dense(total_fine, dtype=dtype, name="fine_mu")
        self.fine_log_sigma_head = nn.Dense(total_fine, dtype=dtype, name="fine_log_sigma")

        self.coarse_action_horizon = cfg.coarse_action_horizon
        self.action_horizon = cfg.action_horizon
        self.action_dim = cfg.action_dim

    def __call__(self, prefix_tokens: jnp.ndarray, prefix_mask: jnp.ndarray):
        # Ensure prefix_mask is 2D (B, S) — real data may have extra dims like (1, B, S)
        if prefix_mask.ndim > 2:
            prefix_mask = prefix_mask.reshape(prefix_tokens.shape[0], -1)
        prefix_tokens = self.input_proj(prefix_tokens)
        # attn_mask: (B, 1, S, S) — explicit head dim for broadcasting with (B, H, Q, K) attention
        attn_mask = (prefix_mask[:, None, :] * prefix_mask[:, :, None])[:, None, :, :]
        hidden = self.transformer(prefix_tokens, mask=attn_mask, deterministic=False)

        mask_expanded = prefix_mask[:, :, None]
        pooled = jnp.sum(hidden * mask_expanded, axis=1) / jnp.maximum(jnp.sum(mask_expanded, axis=1), 1.0)

        features = self.feature_proj(pooled)

        mu_c = self.coarse_mu_head(features).reshape(-1, self.coarse_action_horizon, self.action_dim)
        ls_c = self.coarse_log_sigma_head(features).reshape(-1, self.coarse_action_horizon, self.action_dim)
        mu_f = self.fine_mu_head(features).reshape(-1, self.action_horizon, self.action_dim)
        ls_f = self.fine_log_sigma_head(features).reshape(-1, self.action_horizon, self.action_dim)
        return mu_c, ls_c, mu_f, ls_f
