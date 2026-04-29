"""
Training step smoke test for ACoT-VLA MyModal.

Tests one forward + backward pass with minimal data to validate
the full training pipeline before running full training.

Usage:
    python scripts/test_train_step.py --config-name acot_r2a_mymodal_temporal_noise
"""

import dataclasses
import logging
import os
import sys

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import optax

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)d)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def init_train_state(config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh):
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    partial_params = load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    return init(init_rng, partial_params)


def make_fake_batch(config: _config.TrainConfig):
    """Create a minimal fake batch matching the expected data shapes."""
    model_config = config.model
    bs = 1  # minimal batch size
    T_img = getattr(model_config, "num_history_frames", 4)
    T_act = model_config.action_horizon
    T_cact = model_config.coarse_action_horizon
    H, W = _model.IMAGE_RESOLUTION
    C = 3
    action_dim = model_config.action_dim
    coarse_action_dim = action_dim  # same as action dim for this model

    # Get image keys from the model's inputs_spec
    fake_obs_tuple = model_config.inputs_spec(batch_size=bs)
    image_keys = list(fake_obs_tuple[0].images.keys())

    images = {k: jnp.ones((bs, T_img, H, W, C), dtype=jnp.float32) for k in image_keys}
    image_masks = {k: jnp.ones((bs,), dtype=jnp.bool_) for k in image_keys}
    state = jnp.ones((bs, action_dim), dtype=jnp.float32)

    observation = _model.Observation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=None,
        tokenized_prompt_mask=None,
        token_ar_mask=None,
        token_loss_mask=None,
    )
    actions = jnp.ones((bs, T_act, action_dim), dtype=jnp.float32)
    coarse_actions = jnp.ones((bs, T_cact, coarse_action_dim), dtype=jnp.float32)

    return (observation, actions, coarse_actions)


@dataclasses.dataclass
class Args:
    config_name: str = "acot_r2a_mymodal_temporal_noise"
    exp_name: str = "r2a_mymodal_v1"


def main():
    import tyro

    args: Args = tyro.cli(Args, args=sys.argv[1:])
    config = _config.get_config(args.config_name)
    exp_name = args.exp_name

    init_logging()

    # Disable typechecking to avoid optax.OptState ForwardRef issue
    from openpi.shared.array_typing import disable_typechecking

    disable_typechecking().__enter__()

    logging.info(f"Config: {args.config_name}, Experiment: {exp_name}")

    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)

    # Step 1: Initialize model + load pretrained weights
    logging.info("Step 1/3: Initializing model & loading pretrained weights...")
    train_state = init_train_state(config, init_rng, mesh)
    jax.block_until_ready(train_state)
    num_params = training_utils.count_parameters(train_state.params)
    logging.info(f"Model initialized! Total parameters: {num_params:,}")
    logging.info("Init PASSED ✅")

    # Step 2: Create minimal fake batch
    logging.info("Step 2/3: Creating fake batch...")
    batch = make_fake_batch(config)
    obs, actions, coarse_actions = batch
    logging.info(f"  observation.images: { {k: v.shape for k, v in obs.images.items()} }")
    logging.info(f"  observation.state:   {obs.state.shape}")
    logging.info(f"  actions:             {actions.shape}")
    logging.info(f"  coarse_actions:      {coarse_actions.shape}")
    logging.info("Fake batch PASSED ✅")

    # Step 3: Run one forward + backward pass (train_step)
    logging.info("Step 3/3: Running forward + backward pass...")

    def train_step_fn(state, batch):
        """Single training step: forward loss + backward grad + optimizer update."""
        model = nnx.merge(state.model_def, state.params)
        model.train()

        obs_batch, act_batch, cact_batch = batch
        train_step_rng = jax.random.fold_in(jax.random.PRNGKey(0), state.step)

        def loss_fn(model, rng, observation, actions, coarse_actions):
            return model.compute_loss(rng, observation, actions, coarse_actions, train=True)

        diff_state = nnx.DiffState(0, config.trainable_filter)
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(
            model, train_step_rng, obs_batch, act_batch, cact_batch
        )

        params = state.params.filter(config.trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)

        nnx.update(model, new_params)
        new_params_full = nnx.state(model)

        new_state = dataclasses.replace(state, step=state.step + 1, params=new_params_full, opt_state=new_opt_state)

        return new_state, loss, grads

    # Run with jax.jit (params already on GPU, so no OOM risk)
    logging.info("  Running with jax.jit...")
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    state_sharding = sharding.fsdp_sharding(jax.eval_shape(lambda s: s, train_state), mesh)

    jit_train_step = jax.jit(
        train_step_fn,
        in_shardings=(state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated_sharding, replicated_sharding),
    )
    try:
        new_state, loss, grads = jit_train_step(train_state, batch)
        jax.block_until_ready(new_state)
        jax.block_until_ready(loss)
        logging.info(f"  Loss value: {loss:.6f}")
        grad_norm = optax.global_norm(grads)
        logging.info(f"  Grad norm: {grad_norm:.6f}")
        logging.info(f"  New step: {new_state.step}")
        logging.info("Forward + Backward PASSED ✅")
    except Exception as e:
        logging.error(f"Forward + Backward FAILED ❌: {e}", exc_info=True)
        sys.exit(1)

    logging.info("=" * 60)
    logging.info("ALL TRAINING PIPELINE TESTS PASSED!")
    logging.info("The model can successfully run forward + backward propagation.")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
