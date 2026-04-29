"""
Real-data training smoke test for ACoT-VLA MyModal.

Loads ONE real dataset (not all 13), runs a few training steps
with real data to validate the full pipeline end-to-end.

Usage:
    python scripts/test_real_data_train.py --config-name acot_r2a_mymodal_temporal_noise --exp-name r2a_mymodal_v1
"""

import dataclasses
import functools
import logging
import os
import sys
import time

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def load_weights_and_validate(loader, params_shape):
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def init_train_state(config, init_rng, mesh):
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng, partial_params=None):
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
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)
    train_state = init(init_rng, partial_params)
    return train_state, state_sharding


@dataclasses.dataclass
class Args:
    config_name: str = "acot_r2a_mymodal_temporal_noise"
    exp_name: str = "r2a_mymodal_v1"
    test_steps: int = 5  # run N training steps with real data
    single_dataset: bool = True  # only use first dataset from config
    single_repo_id: str = ""  # override: use this specific dataset (empty = use config's first)


def main():
    import tyro

    args: Args = tyro.cli(Args, args=sys.argv[1:])
    config = _config.get_config(args.config_name)

    init_logging()
    from openpi.shared.array_typing import disable_typechecking

    disable_typechecking().__enter__()

    # Override dataset to use only one (for fast testing)
    target_repo = args.single_repo_id or config.data.repo_id[0]
    logging.info(f"Config: {args.config_name}, Exp: {args.exp_name}, Steps: {args.test_steps}")
    logging.info(f"  Using single dataset: {target_repo}")

    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # ---- Step 1: Init model ----
    logging.info("Step 1/5: Initializing model...")
    t0 = time.time()
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh)
    jax.block_until_ready(train_state)
    num_params = training_utils.count_parameters(train_state.params)
    logging.info(f"Init done! Params: {num_params:,}, Time: {time.time() - t0:.1f}s")

    # ---- Step 2: Create data loader with ONE dataset ----
    logging.info("Step 2/5: Creating minimal dataset config (single dataset)...")
    logging.info(f"  Using dataset: {target_repo}")

    import copy

    test_config = copy.deepcopy(config)
    # Bypass frozen: modify at the data_config level
    _dc = test_config.data
    while not hasattr(_dc, "repo_id"):
        _dc = getattr(_dc, "__pytree_state__", None) or getattr(_dc, "base", None)
    if hasattr(_dc, "repo_id"):
        object.__setattr__(_dc, "repo_id", [target_repo])
    else:
        logging.error("Cannot find repo_id in config data chain")
        sys.exit(1)
    object.__setattr__(test_config, "batch_size", min(config.batch_size, 4))
    object.__setattr__(test_config, "num_workers", 0)
    object.__setattr__(test_config, "num_train_steps", args.test_steps)

    t_data = time.time()
    data_loader = _data_loader.create_data_loader(test_config, sharding=data_sharding, shuffle=False)
    data_iter = iter(data_loader)
    batch = next(data_iter)
    jax.block_until_ready(batch)
    logging.info(f"  Data loaded! Time: {time.time() - t_data:.1f}s")
    logging.info(f"  Batch keys: {type(batch).__name__}")
    obs, actions, coarse_actions = batch
    img_shapes = {k: v.shape for k, v in obs.images.items()}
    logging.info(f"  obs.images: {img_shapes}")
    logging.info(f"  obs.state:   {obs.state.shape}")
    logging.info(f"  actions:      {actions.shape}")
    logging.info(f"  coarse_act:  {coarse_actions.shape}")

    # ---- Step 3: Compile train_step ----
    logging.info("Step 3/5: Compiling train_step (XLA)...")

    def acot_train_step(cfg, rng, state, batch):
        model = nnx.merge(state.model_def, state.params)
        model.train()
        obs_b, act_b, cact_b = batch
        train_step_rng = jax.random.fold_in(rng, state.step)

        @at.typecheck
        def loss_fn(model, rng, observation, actions, coarse_actions):
            return model.compute_loss(rng, observation, actions, coarse_actions, train=True)

        diff_state = nnx.DiffState(0, cfg.trainable_filter)
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_step_rng, obs_b, act_b, cact_b)
        params = state.params.filter(cfg.trainable_filter)
        updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
        new_params = optax.apply_updates(params, updates)
        nnx.update(model, new_params)
        new_params_full = nnx.state(model)
        new_state = dataclasses.replace(state, step=state.step + 1, params=new_params_full, opt_state=new_opt_state)
        if state.ema_decay is not None:
            # ema_params was created from nnx.state(model) with freeze_filter applied,
            # so we must use the same structure for EMA update.
            # Skip PRNG keys and other non-numeric types in EMA.
            def _ema_update(old, new):
                if hasattr(old, "dtype") and jax.numpy.issubdtype(old.dtype, jax.numpy.floating):
                    return state.ema_decay * old + (1 - state.ema_decay) * new
                return new

            new_state = dataclasses.replace(
                new_state,
                ema_params=jax.tree.map(_ema_update, state.ema_params, new_params_full),
            )
        return new_state, {"loss": loss, "grad_norm": optax.global_norm(grads)}

    # ---- Step 3: Compile train_step (JIT) ----
    logging.info("Step 3/5: Compiling train_step (XLA)...")
    t_compile = time.time()
    ptrain_step = jax.jit(
        functools.partial(acot_train_step, test_config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
    )

    # Warmup (compilation)
    with sharding.set_mesh(mesh):
        train_state, info = ptrain_step(jax.random.PRNGKey(42), train_state, batch)
    jax.block_until_ready(train_state)
    compile_time = time.time() - t_compile
    logging.info(f"  Compilation done! Time: {compile_time:.1f}s")
    logging.info(f"  Step 0 loss: {jax.device_get(info['loss']):.6f}")

    # ---- Step 4: Train loop ----
    logging.info(f"Step 4/5: Running {args.test_steps} real training steps...")
    t_train = time.time()
    train_rng = jax.random.PRNGKey(123)
    losses = []
    for step in range(args.test_steps):
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
            batch = next(data_iter)  # get next real batch
        jax.block_until_ready(train_state)
        loss_val = float(jax.device_get(info["loss"]))
        grad_val = float(jax.device_get(info["grad_norm"]))
        losses.append(loss_val)
        logging.info(f"  Step {step + 1}: loss={loss_val:.6f}, grad_norm={grad_val:.4f}")
    train_time = time.time() - t_train
    avg_loss = sum(losses) / len(losses)
    logging.info(
        f"  Done! Avg loss: {avg_loss:.6f}, Total: {train_time:.1f}s ({train_time / args.test_steps:.2f}s/step)"
    )

    # ---- Step 5: Summary ----
    logging.info("=" * 60)
    logging.info("ALL REAL-DATA TRAINING TESTS PASSED")
    logging.info(f"  Dataset:          {target_repo}")
    logging.info(f"  Params:            {num_params:,}")
    logging.info(f"  Compile time:      {compile_time:.1f}s")
    logging.info(f"  Training:         {args.test_steps} steps, {train_time:.1f}s total")
    logging.info(f"  Loss curve:        {' -> '.join(f'{l:.4f}' for l in losses)}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
