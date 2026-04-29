"""
Checkpoint save/restore speed test for ACoT-VLA MyModal.

No training involved. Just:
  1. Initialize model + load pretrained weights
  2. Save checkpoint
  3. Restore checkpoint
  4. Measure size/speed

Usage:
    python scripts/test_checkpoint_only.py --config-name acot_r2a_mymodal_temporal_noise --exp-name r2a_mymodal_v1
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

import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from openpi.training import checkpoints as _checkpoints
import openpi.training.config as _config
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
    return init(init_rng, partial_params)


@dataclasses.dataclass
class Args:
    config_name: str = "acot_r2a_mymodal_temporal_noise"
    exp_name: str = "r2a_mymodal_v1"


def main():
    import tyro

    args: Args = tyro.cli(Args, args=sys.argv[1:])
    config = _config.get_config(args.config_name)

    init_logging()
    from openpi.shared.array_typing import disable_typechecking

    disable_typechecking().__enter__()

    ckpt_base = (pathlib.Path(config.checkpoint_base_dir) / config.name / args.exp_name).resolve()
    save_dir = (ckpt_base / "ckpt_test").resolve()
    logging.info(f"Config: {args.config_name}, Exp: {args.exp_name}")
    logging.info(f"Save dir: {save_dir}")

    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)

    # Step 1: Init
    logging.info("Step 1/4: Initializing model...")
    t0 = time.time()
    train_state = init_train_state(config, init_rng, mesh)
    jax.block_until_ready(train_state)
    num_params = training_utils.count_parameters(train_state.params)
    init_time = time.time() - t0
    logging.info(f"Init done! Params: {num_params:,}, Time: {init_time:.1f}s")

    # Step 2: Save checkpoint (using same logic as train.py)
    logging.info("Step 2/4: Saving checkpoint (CheckpointManager, same as train.py)...")
    ckpt_dir = save_dir.parent
    os.makedirs(ckpt_dir, exist_ok=True)
    mngr, _ = _checkpoints.initialize_checkpoint_dir(ckpt_dir, keep_period=None, overwrite=True, resume=False)

    t_save = time.time()

    class MockDataLoader:
        def data_config(self):
            class C:
                norm_stats = None
                asset_id = None

            return C()

    _checkpoints.save_state(mngr, train_state, data_loader=MockDataLoader(), step=0)
    mngr.wait_until_finished()
    save_time = time.time() - t_save

    ckpt_files = list(ckpt_dir.rglob("*"))
    ckpt_size_bytes = sum(f.stat().st_size for f in ckpt_files if f.is_file())
    ckpt_size_mb = ckpt_size_bytes / (1024 * 1024)
    ckpt_size_gb = ckpt_size_mb / 1024
    save_speed = ckpt_size_mb / save_time if save_time > 0 else 0
    logging.info(f"  Saved! Size: {ckpt_size_gb:.1f} GB ({ckpt_size_mb:.0f} MB)")
    logging.info(f"  Save time: {save_time:.1f}s, Speed: {save_speed:.0f} MB/s")

    # Step 3: Metadata read (fast, no data transfer)
    logging.info("Step 3/4: Reading metadata only...")
    t_meta = time.time()
    steps = mngr.all_steps()
    meta_time = time.time() - t_meta
    logging.info(f"  Metadata read: {meta_time:.3f}s, available steps: {list(steps)}")

    # Step 4: Full restore (using same logic as train.py --resume)
    logging.info("Step 4/4: Full restore (same as train.py --resume)...")
    t_restore = time.time()
    restored = _checkpoints.restore_state(mngr, train_state, data_loader=MockDataLoader(), step=0)
    jax.block_until_ready(restored)
    restore_time = time.time() - t_restore
    restore_speed = ckpt_size_mb / restore_time if restore_time > 0 else 0
    restored_params = training_utils.count_parameters(restored.params)

    logging.info(f"  Restored! Time: {restore_time:.1f}s, Speed: {restore_speed:.0f} MB/s")
    logging.info(f"  Step={restored.step}, Params={restored_params:,}, ema_decay={restored.ema_decay}")

    assert restored.step == train_state.step, "Step mismatch"
    assert restored_params == num_params, "Param count mismatch"
    logging.info("  Validation PASSED ✅")

    logging.info("=" * 60)
    logging.info("ALL CHECKPOINT TESTS PASSED")
    logging.info(f"  Params:          {num_params:,}")
    logging.info(f"  Init time:       {init_time:.1f}s")
    logging.info(f"  Checkpoint size: {ckpt_size_gb:.1f} GB")
    logging.info(f"  Save speed:      {save_speed:.0f} MB/s ({save_time:.1f}s)")
    logging.info(f"  Restore speed:   {restore_speed:.0f} MB/s ({restore_time:.1f}s)")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
