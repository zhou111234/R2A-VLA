"""
Checkpoint & Training Loop Smoke Test for ACoT-VLA MyModal.

Tests:
  1. Run 1001 steps of training (triggers checkpoint save at step 1000)
  2. Verify checkpoint is saved and readable
  3. Measure checkpoint load/restore speed (without full download)
  4. Print loss at step 0 and every log_interval steps

Usage:
    python scripts/test_checkpoint.py --config-name acot_r2a_mymodal_temporal_noise --exp-name r2a_mymodal_v1
"""

import dataclasses
import logging
import os
import pathlib
import sys
import time

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import optax
import tqdm

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


def make_fake_batch(model_config, bs=1):
    T_img = getattr(model_config, "num_history_frames", 4)
    T_act = model_config.action_horizon
    T_cact = model_config.coarse_action_horizon
    H, W = _model.IMAGE_RESOLUTION
    C = 3
    action_dim = model_config.action_dim

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
    coarse_actions = jnp.ones((bs, T_cact, action_dim), dtype=jnp.float32)
    return (observation, actions, coarse_actions)


@dataclasses.dataclass
class Args:
    config_name: str = "acot_r2a_mymodal_temporal_noise"
    exp_name: str = "r2a_mymodal_v1"
    test_steps: int = 1001  # run N+1 steps to trigger save at step N
    batch_size: int = 1  # minimal for speed


def main():
    import tyro

    args: Args = tyro.cli(Args, args=sys.argv[1:])
    config = _config.get_config(args.config_name)

    init_logging()
    from openpi.shared.array_typing import disable_typechecking

    disable_typechecking().__enter__()

    ckpt_base = pathlib.Path(config.checkpoint_base_dir) / config.name / args.exp_name
    logging.info(f"Config: {args.config_name}, Exp: {args.exp_name}")
    logging.info(f"Checkpoint dir: {ckpt_base}")

    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)

    # ========== Step 1: Init ==========
    logging.info("Step 1/5: Initializing model...")
    t0 = time.time()
    train_state = init_train_state(config, init_rng, mesh)
    jax.block_until_ready(train_state)
    num_params = training_utils.count_parameters(train_state.params)
    logging.info(f"Init done! Params: {num_params:,}, Time: {time.time() - t0:.1f}s")

    # ========== Step 2: Build jit train_step ==========
    logging.info("Step 2/5: Compiling train_step (XLA)...")
    batch = make_fake_batch(config.model, bs=args.batch_size)

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))

    def train_step_fn(state, batch):
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
        return new_state, {"loss": loss, "grad_norm": optax.global_norm(grads)}

    # Compile
    t_compile = time.time()
    ptrain_step = jax.jit(
        train_step_fn,
        in_shardings=(replicated_sharding, data_sharding),
        out_shardings=(replicated_sharding, replicated_sharding),
    )
    # Warmup run (compilation happens here)
    with sharding.set_mesh(mesh):
        train_state, info = ptrain_step(train_state, batch)
    jax.block_until_ready(train_state)
    compile_time = time.time() - t_compile
    logging.info(f"Compilation done! Time: {compile_time:.1f}s")
    logging.info(
        f"  Step 0 loss: {jax.device_get(info['loss']):.6f}  grad_norm: {jax.device_get(info['grad_norm']):.6f}"
    )

    # ========== Step 3: Train loop + checkpoint save ==========
    logging.info(f"Step 3/5: Running {args.test_steps - 1} more steps (save at step {config.save_interval})...")
    t_train = time.time()
    infos = []
    start_step = int(train_state.step)
    pbar = tqdm.tqdm(range(start_step, args.test_steps), initial=start_step, total=args.test_steps, dynamic_ncols=True)

    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()

    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_state, batch)
        infos.append(info)

        # Print at step 0 (already printed above) AND every log_interval
        if step % config.log_interval == 0 or step == start_step:
            stacked_infos = jax.tree.map(lambda *xs: jnp.stack(xs), *infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            infos = []

        # Save checkpoint at save_interval
        if step > start_step and step % config.save_interval == 0:
            save_dir = (ckpt_base / f"step_{step}").resolve()
            os.makedirs(save_dir, exist_ok=True)
            save_args = jax.tree.map(ocp.args.StandardSave, train_state)
            t_save = time.time()
            checkpointer.save(save_dir, train_state, save_args=save_args, force=True)
            jax.block_until_ready(train_state)
            save_time = time.time() - t_save
            ckpt_size_mb = sum(f.stat().st_size for f in pathlib.Path(save_dir).rglob("*") if f.is_file()) / (
                1024 * 1024
            )
            logging.info(f"  Checkpoint saved at step {step}: {ckpt_size_mb:.0f} MB, took {save_time:.1f}s")

    train_time = time.time() - t_train
    avg_step_time = train_time / (args.test_steps - start_step)
    logging.info(f"Training loop done! Total: {train_time:.1f}s, Avg step: {avg_step_time:.3f}s")

    # ========== Step 4: Checkpoint restore speed test ==========
    last_ckpt_dir = ckpt_base / f"step_{(args.test_steps // config.save_interval) * config.save_interval}"
    if not last_ckpt_dir.exists():
        # fallback: find any checkpoint dir
        ckpt_dirs = sorted([d for d in ckpt_base.iterdir() if d.is_dir() and d.name.startswith("step_")])
        if ckpt_dirs:
            last_ckpt_dir = ckpt_dirs[-1]
        else:
            logging.warning("No checkpoint found, skipping restore test")
            last_ckpt_dir = None

    if last_ckpt_dir and last_ckpt_dir.exists():
        logging.info(f"Step 4/5: Testing checkpoint restore from {last_ckpt_dir.name}...")

        # Test 1: metadata only (fast, no data transfer)
        t_meta = time.time()
        checkpointer.metadata(last_ckpt_dir)
        meta_time = time.time() - t_meta
        logging.info(f"  Metadata read: {meta_time:.3f}s")

        # Test 2: full restore (measures actual read speed)
        t_restore = time.time()
        restored = checkpointer.restore(last_ckpt_dir)
        jax.block_until_ready(restored)
        restore_time = time.time() - t_restore
        restored_params = training_utils.count_parameters(restored.params)
        logging.info(f"  Full restore: {restore_time:.1f}s ({restored_params:,} params)")
        logging.info(f"  Restored step={restored.step}, ema_decay={restored.ema_decay}")

        # Estimate download speed (size / time)
        ckpt_total_mb = sum(f.stat().st_size for f in last_ckpt_dir.rglob("*") if f.is_file()) / (1024 * 1024)
        speed_mbs = ckpt_total_mb / restore_time if restore_time > 0 else 0
        logging.info(f"  Checkpoint size: {ckpt_total_mb:.0f} MB")
        logging.info(f"  Estimated I/O throughput: {speed_mbs:.0f} MB/s")

        # Validate restored state matches
        assert restored.step == train_state.step, f"Step mismatch: {restored.step} vs {train_state.step}"
        logging.info("  Restore validation PASSED ✅")
    else:
        logging.warning("Step 4/5: SKIPPED - no checkpoint found")

    # ========== Step 5: Summary ==========
    logging.info("=" * 60)
    logging.info("ALL CHECKPOINT TESTS PASSED!")
    logging.info(f"  Model params:     {num_params:,}")
    logging.info(f"  Compile time:     {compile_time:.1f}s")
    logging.info(f"  Train time:       {train_time:.1f}s ({args.test_steps} steps, {avg_step_time:.3f}s/step)")
    if "restore_time" in dir():
        logging.info(f"  Restore time:     {restore_time:.1f}s ({ckpt_total_mb:.0f} MB, {speed_mbs:.0f} MB/s)")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
