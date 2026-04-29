"""
ACoT-VLA with Temporal SigLIP Encoding + Noise Expert.

Integrates into the official ACoT-VLA codebase (openpi imports).

Key innovations over baseline:
1. Reuses the official SigLIP encoder but processes T frames per camera,
   mixes with a TemporalMixer, then pools back to single-frame token count.
2. NoiseExpert (Gemma 300M Linen module) replaces standard Gaussian noise
   in flow matching. Enabled at both training and inference — during inference
   the expert generates the initial noise distribution for ODE denoising.
"""

import dataclasses
import logging

import augmax
import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

import openpi.models.acot_vla as _acot_vla
import openpi.models.gemma as _gemma
import openpi.models.model as _model
from openpi.models.noise_expert import NoiseExpert
from openpi.models.noise_expert import NoiseExpertConfig
from openpi.models.pi0 import make_attn_mask
from openpi.shared import array_typing as at
from openpi.shared import image_tools
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("ACOT_VLA_MyModal")


# ---------------------------------------------------------------------------
# Temporal Mixer (pure NNX) – replaces the Linen TemporalSigLIPEncoder
# by reusing the official SigLIP backbone and only adding temporal mixing.
# ---------------------------------------------------------------------------


class TemporalBlockNNX(nnx.Module):
    """Single transformer block for temporal self-attention."""

    def __init__(self, width: int, num_heads: int, mlp_dim: int, *, rngs: nnx.Rngs):
        self.num_heads = num_heads
        self.head_dim = width // num_heads

        self.ln1 = nnx.LayerNorm(width, rngs=rngs)
        self.q_proj = nnx.Linear(width, width, rngs=rngs)
        self.k_proj = nnx.Linear(width, width, rngs=rngs)
        self.v_proj = nnx.Linear(width, width, rngs=rngs)
        self.out_proj = nnx.Linear(width, width, rngs=rngs)

        self.ln2 = nnx.LayerNorm(width, rngs=rngs)
        self.fc1 = nnx.Linear(width, mlp_dim, rngs=rngs)
        self.fc2 = nnx.Linear(mlp_dim, width, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        B, S, D = x.shape
        h = self.ln1(x)

        q = self.q_proj(h).reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(h).reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(h).reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        scale = 1.0 / jnp.sqrt(jnp.float32(self.head_dim))
        attn = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale
        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("bhqk,bhkd->bhqd", attn, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
        out = self.out_proj(out)
        x = x + out

        h = self.ln2(x)
        h = jax.nn.gelu(self.fc1(h))
        h = self.fc2(h)
        return x + h


class TemporalMixerNNX(nnx.Module):
    """
    Mixes temporal information across frames via self-attention, then
    average-pools across the time dimension so the output has the same
    token count as a single-frame SigLIP encoding.

    Input:  (B, T*P, D)  – tokens from T frames × P patches
    Output: (B, P, D)    – temporally-mixed, time-pooled tokens
    """

    def __init__(self, width: int, num_layers: int = 6, num_heads: int = 16, *, rngs: nnx.Rngs):
        mlp_dim = 4 * width
        self.blocks = [TemporalBlockNNX(width, num_heads, mlp_dim, rngs=rngs) for _ in range(num_layers)]
        self.final_norm = nnx.LayerNorm(width, rngs=rngs)

    def __call__(self, x: jnp.ndarray, num_frames: int, num_patches: int) -> jnp.ndarray:
        B = x.shape[0]
        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)
        x = x.reshape(B, num_frames, num_patches, -1)
        return x.mean(axis=1)  # (B, P, D)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ACOTMyModalConfig(_acot_vla.ACOTConfig):
    """Extends the baseline ACOTConfig with temporal + noise expert flags."""

    num_history_frames: int = 4
    num_cameras: int = 3
    temporal_encoder_layers: int = 6
    adopt_noise_expert: bool = True

    def __post_init__(self):
        super().__post_init__()

    @override
    def create(self, rng: at.KeyArrayLike) -> "ACOT_VLA_MyModal":
        return ACOT_VLA_MyModal(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        if self.num_history_frames > 1:
            image_spec = jax.ShapeDtypeStruct(
                [batch_size, self.num_history_frames, *_model.IMAGE_RESOLUTION, 3], jnp.float32
            )
        else:
            image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)
        return observation_spec, action_spec

    def get_freeze_filter(
        self,
        freeze_llm=False,
        freeze_llm_embedder=True,
        freeze_vision=False,
        freeze_dual_ae=None,
        freeze_temporal=False,
        freeze_noise_expert=False,
    ):
        if freeze_dual_ae is None:
            freeze_dual_ae = [False, False]
        base_filter = super().get_freeze_filter(
            freeze_llm=freeze_llm,
            freeze_llm_embedder=freeze_llm_embedder,
            freeze_vision=freeze_vision,
            freeze_dual_ae=freeze_dual_ae,
        )
        extra_freeze = []
        if freeze_temporal:
            extra_freeze.append(nnx_utils.PathRegex(".*temporal_mixer.*"))
        if freeze_noise_expert:
            extra_freeze.append(nnx_utils.PathRegex(".*noise_expert.*"))

        if not extra_freeze:
            return base_filter
        if base_filter is nnx.Nothing:
            return nnx.Any(*extra_freeze)
        return nnx.Any(base_filter, *extra_freeze)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class ACOT_VLA_MyModal(_acot_vla.ACOT_VLA):
    """
    ACoT-VLA with two innovations:
    1. Temporal multi-frame SigLIP encoding (reuses official SigLIP backbone)
    2. Learned noise via NoiseExpert (replaces Gaussian in flow matching)
    """

    def __init__(self, config: ACOTMyModalConfig, rngs: nnx.Rngs):
        # Initialize the baseline ACOT_VLA first – this sets up PaliGemma (llm+img),
        # action projections, action reasoners, etc.
        super().__init__(config, rngs=rngs)
        self.mymodal_config = config

        paligemma_config = _gemma.get_config(config.paligemma_variant)

        # ---- Innovation 1: Temporal Mixer ----
        self.temporal_mixer = TemporalMixerNNX(
            width=paligemma_config.width,
            num_layers=config.temporal_encoder_layers,
            num_heads=16,
            rngs=rngs,
        )

        # ---- Innovation 2: Noise Expert ----
        self.adopt_noise_expert = config.adopt_noise_expert
        if config.adopt_noise_expert:
            noise_cfg = NoiseExpertConfig(
                action_dim=config.action_dim,
                coarse_action_horizon=config.coarse_action_horizon,
                action_horizon=config.action_horizon,
                dtype=config.dtype,
            )
            noise_expert_linen = NoiseExpert(config=noise_cfg)
            self.noise_expert = nnx_bridge.ToNNX(noise_expert_linen, rngs=rngs)
            sample_tok = jnp.ones((1, 100, paligemma_config.width))
            sample_msk = jnp.ones((1, 100))
            self.noise_expert.lazy_init(sample_tok, sample_msk)

    # ------------------------------------------------------------------
    # Override embed_prefix for temporal image encoding
    # ------------------------------------------------------------------
    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []

        num_history = self.mymodal_config.num_history_frames

        for name in obs.images:
            image = obs.images[name]

            if image.ndim == 5 and num_history > 1:
                # Temporal images: (B, T, H, W, C)
                B, T, H, W, C = image.shape
                flat_images = image.reshape(B * T, H, W, C)
                image_tokens, _ = self.PaliGemma.img(flat_images, train=False)
                num_patches = image_tokens.shape[1]
                D = image_tokens.shape[2]
                image_tokens = image_tokens.reshape(B, T, num_patches, D)
                flat_tok = image_tokens.reshape(B, T * num_patches, D)
                image_tokens = self.temporal_mixer(flat_tok, T, num_patches)
            else:
                # Single frame: (B, H, W, C) – behaves like baseline
                if image.ndim == 5:
                    image = image[:, -1]  # take most recent frame
                image_tokens, _ = self.PaliGemma.img(image, train=False)

            tokens.append(image_tokens)
            input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1]))
            ar_mask += [False] * image_tokens.shape[1]

        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            ar_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    # ------------------------------------------------------------------
    # Noise Expert helpers
    # ------------------------------------------------------------------
    def _sample_noise_from_expert(self, rng, prefix_tokens, prefix_mask):
        mu_c, log_s_c, mu_f, log_s_f = self.noise_expert(prefix_tokens, prefix_mask)
        rng_c, rng_f = jax.random.split(rng)
        coarse_noise = mu_c + jnp.exp(0.5 * log_s_c) * jax.random.normal(rng_c, mu_c.shape)
        fine_noise = mu_f + jnp.exp(0.5 * log_s_f) * jax.random.normal(rng_f, mu_f.shape)
        return coarse_noise, fine_noise

    # ------------------------------------------------------------------
    # Override preprocess to handle temporal images
    # ------------------------------------------------------------------
    def _preprocess_obs(self, rng, observation, *, train=False):
        """Preprocessing that handles (B,T,H,W,C) temporal images with augmentation."""
        batch_shape = observation.state.shape[:-1]
        image_resolution = _model.IMAGE_RESOLUTION

        out_images = {}
        for key in _model.IMAGE_KEYS:
            if key not in observation.images:
                continue
            image = observation.images[key]

            if image.ndim == 5:
                B, T, H, W, C = image.shape
                flat = image.reshape(B * T, H, W, C)
                if image_resolution != (H, W):
                    flat = image_tools.resize_with_pad(flat, *image_resolution)
                if train and rng is not None:
                    flat = flat / 2.0 + 0.5
                    transforms = []
                    if "wrist" not in key:
                        h, w = flat.shape[1], flat.shape[2]
                        transforms += [
                            augmax.RandomCrop(int(w * 0.95), int(h * 0.95)),
                            augmax.Resize(w, h),
                            augmax.Rotate((-5, 5)),
                        ]
                    transforms += [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]
                    sub_rngs = jax.random.split(rng, flat.shape[0])
                    flat = jax.vmap(augmax.Chain(*transforms))(sub_rngs, flat)
                    flat = flat * 2.0 - 1.0
                image = flat.reshape(B, T, *flat.shape[1:])
            else:
                if image.shape[1:3] != image_resolution:
                    image = image_tools.resize_with_pad(image, *image_resolution)
                if train and rng is not None:
                    image = image / 2.0 + 0.5
                    transforms = []
                    if "wrist" not in key:
                        h, w = image.shape[1], image.shape[2]
                        transforms += [
                            augmax.RandomCrop(int(w * 0.95), int(h * 0.95)),
                            augmax.Resize(w, h),
                            augmax.Rotate((-5, 5)),
                        ]
                    transforms += [augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5)]
                    sub_rngs = jax.random.split(rng, image.shape[0])
                    image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)
                    image = image * 2.0 - 1.0

            out_images[key] = image

        out_masks = {}
        for key in out_images:
            if key not in observation.image_masks:
                out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool_)
            else:
                out_masks[key] = jnp.asarray(observation.image_masks[key])

        return _model.Observation(
            images=out_images,
            image_masks=out_masks,
            state=observation.state,
            tokenized_prompt=observation.tokenized_prompt,
            tokenized_prompt_mask=observation.tokenized_prompt_mask,
        )

    # ------------------------------------------------------------------
    # Override compute_loss
    # ------------------------------------------------------------------
    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        coarse_actions: _model.CoarseActions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, time_rng, coarse_noise_rng, expert_noise_rng = jax.random.split(rng, 4)

        if self.mymodal_config.num_history_frames > 1:
            observation = self._preprocess_obs(preprocess_rng, observation, train=train)
        else:
            observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]

        # --- Embed prefix (temporal encoding happens here) ---
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)

        # --- Noise: learned or standard Gaussian ---
        if self.adopt_noise_expert and hasattr(self, "noise_expert"):
            coarse_action_noise, expert_action_noise = self._sample_noise_from_expert(
                coarse_noise_rng, prefix_tokens, prefix_mask
            )
        else:
            coarse_action_noise = jax.random.normal(coarse_noise_rng, coarse_actions.shape)
            expert_action_noise = jax.random.normal(expert_noise_rng, actions.shape)

        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]

        x_ref_t = time_expanded * coarse_action_noise + (1.0 - time_expanded) * coarse_actions
        u_ref_t = coarse_action_noise - coarse_actions
        x_expert_t = time_expanded * expert_action_noise + (1.0 - time_expanded) * actions
        u_expert_t = expert_action_noise - actions

        # --- KV cache from prefix ---
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None, None], mask=prefix_attn_mask, positions=positions_prefix)

        # --- Explicit action reasoner (coarse trajectory) ---
        if self.adopt_explicit_action_reasoner:
            suf_ref_tok, suf_ref_mask, suf_ref_ar, adarms_ref = self.embed_suffix(
                observation, x_ref_t, time, suf_type="reasoner"
            )
            im = jnp.concatenate([prefix_mask, suf_ref_mask], axis=1)
            am = jnp.concatenate([prefix_ar_mask, suf_ref_ar], axis=0)
            attn_m = make_attn_mask(im, am)
            pos = jnp.cumsum(im, axis=1) - 1
            (_, suf_ref_out, _), _ = self.PaliGemma.llm(
                [prefix_tokens, suf_ref_tok, None],
                mask=attn_m,
                positions=pos,
                adarms_cond=[None, adarms_ref, None],
            )
            explicit_action_reason = coarse_actions
        else:
            explicit_action_reason = None

        # --- Implicit action reasoner ---
        if self.adopt_implicit_action_reasoner:
            K_all, V_all = kv_cache
            K_r = einops.rearrange(K_all, "L B T 1 D -> B L T D")
            V_r = einops.rearrange(V_all, "L B T 1 D -> B L T D")
            implicit_action_reason = self.implicit_action_reasoner(K_r, V_r)
        else:
            implicit_action_reason = None

        # --- Expert suffix forward ---
        suf_exp_tok, suf_exp_mask, suf_exp_ar, adarms_exp = self.embed_suffix(
            observation,
            x_expert_t,
            time,
            explicit_action_reason=explicit_action_reason,
            implicit_action_reason=implicit_action_reason,
            suf_type="expert",
        )
        im = jnp.concatenate([prefix_mask, suf_exp_mask], axis=1)
        am = jnp.concatenate([prefix_ar_mask, suf_exp_ar], axis=0)
        attn_m = make_attn_mask(im, am)
        pos = jnp.cumsum(im, axis=1) - 1
        (_, _, suf_exp_out), _ = self.PaliGemma.llm(
            [prefix_tokens, None, suf_exp_tok],
            mask=attn_m,
            positions=pos,
            adarms_cond=[None, None, adarms_exp],
        )

        # --- Loss ---
        if self.adopt_explicit_action_reasoner:
            v_ref = self.coarse_action_out_proj(suf_ref_out[:, -self.coarse_action_horizon :])
            v_exp = self.action_out_proj(suf_exp_out[:, -self.action_horizon :])
            return jnp.mean(jnp.square(u_ref_t - v_ref)) + jnp.mean(jnp.square(u_expert_t - v_exp))
        v_exp = self.action_out_proj(suf_exp_out[:, -self.action_horizon :])
        return jnp.mean(jnp.square(u_expert_t - v_exp))

    # ------------------------------------------------------------------
    # Override sample_actions – use noise expert for initial distribution
    # ------------------------------------------------------------------
    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
    ) -> _model.Actions:
        if self.mymodal_config.num_history_frames > 1:
            observation = self._preprocess_obs(None, observation, train=False)
        else:
            observation = _model.preprocess_observation(None, observation, train=False)

        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]

        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None, None], mask=prefix_attn_mask, positions=positions)

        # --- Initial noise: from noise expert (learned distribution) or standard Gaussian ---
        ref_rng, exp_rng = jax.random.split(rng, 2)
        if self.adopt_noise_expert and hasattr(self, "noise_expert"):
            ref_noise, exp_noise = self._sample_noise_from_expert(ref_rng, prefix_tokens, prefix_mask)
        else:
            ref_noise = jax.random.normal(ref_rng, (batch_size, self.coarse_action_horizon, self.action_dim))
            exp_noise = jax.random.normal(exp_rng, (batch_size, self.action_horizon, self.action_dim))

        if self.adopt_implicit_action_reasoner:
            K_all, V_all = kv_cache
            K_r = einops.rearrange(K_all, "L B T 1 D -> B L T D")
            V_r = einops.rearrange(V_all, "L B T 1 D -> B L T D")
            implicit_action_reason = self.implicit_action_reasoner(K_r, V_r)
        else:
            implicit_action_reason = None

        # --- Explicit action reasoner ODE: denoise from learned initial distribution ---
        def step_ref(carry):
            x_t, t, idx = carry
            stok, smask, sar, ada = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(t, batch_size), suf_type="reasoner"
            )
            s_attn = make_attn_mask(smask, sar)
            p_attn = einops.repeat(prefix_mask, "b p -> b s p", s=stok.shape[1])
            full = jnp.concatenate([p_attn, s_attn], axis=-1)
            spos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(smask, axis=-1) - 1
            (_, sout, _), _ = self.PaliGemma.llm(
                [None, stok, None],
                mask=full,
                positions=spos,
                kv_cache=kv_cache,
                adarms_cond=[None, ada, None],
            )
            v = self.coarse_action_out_proj(sout[:, -self.coarse_action_horizon :])
            return x_t + dt * v, t + dt, idx + 1

        def cond_ref(carry):
            return carry[1] >= -dt / 2

        if self.adopt_explicit_action_reasoner:
            explicit_action_reason, _, _ = jax.lax.while_loop(cond_ref, step_ref, (ref_noise, 1.0, 1))
        else:
            explicit_action_reason = None

        # --- Expert ODE: denoise from learned initial distribution ---
        def step_exp(carry):
            x_t, t, idx = carry
            stok, smask, sar, ada = self.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(t, batch_size),
                explicit_action_reason=explicit_action_reason,
                implicit_action_reason=implicit_action_reason,
                suf_type="expert",
            )
            s_attn = make_attn_mask(smask, sar)
            p_attn = einops.repeat(prefix_mask, "b p -> b s p", s=stok.shape[1])
            full = jnp.concatenate([p_attn, s_attn], axis=-1)
            spos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(smask, axis=-1) - 1
            (_, _, sout), _ = self.PaliGemma.llm(
                [None, None, stok],
                mask=full,
                positions=spos,
                kv_cache=kv_cache,
                adarms_cond=[None, None, ada],
            )
            v = self.action_out_proj(sout[:, -self.action_horizon :])
            return x_t + dt * v, t + dt, idx + 1

        def cond_exp(carry):
            return carry[1] >= -dt / 2

        x0_exp, _, _ = jax.lax.while_loop(cond_exp, step_exp, (exp_noise, 1.0, 1))

        if self.adopt_explicit_action_reasoner:
            return {"actions": x0_exp, "coarse_actions": explicit_action_reason}
        return {"actions": x0_exp}
