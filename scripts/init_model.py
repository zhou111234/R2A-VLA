"""
Model initialization script for ACoT-VLA MyModal.

Separates model initialization (create model, load pretrained weights,
build optimizer state) from data loading and training loop.

Usage:
    python scripts/init_model.py --config-name acot_r2a_mymodal_temporal_noise

Output:
    Saves initialized TrainState checkpoint to:
    checkpoints/<config_name>/<exp_name>/
"""

import dataclasses
import logging
import os
import pathlib
import sys

import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp

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


def init_train_state(
    config: _config.TrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
):
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
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    partial_params = load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = init(init_rng, partial_params)

    return train_state, state_sharding


@dataclasses.dataclass
class Args:
    config_name: str = "acot_r2a_mymodal_temporal_noise"
    exp_name: str = "r2a_mymodal_v1"


def main():
    import tyro

    args: Args = tyro.cli(Args, args=sys.argv[1:])
    config = _config.get_config(args.config_name)
    # Override exp_name from CLI arg (config is frozen, rebuild checkpoint_dir logic)
    exp_name = args.exp_name

    logging.info(f"Config: {args.config_name}, Experiment: {exp_name}")
    ckpt_base = pathlib.Path(config.checkpoint_base_dir) / config.name / exp_name
    logging.info(f"Checkpoint dir: {ckpt_base}")

    # Workaround: optax.OptState contains unresolvable ForwardRef('ArrayTree')
    # which causes jaxtyping.TypeCheckError when TrainState is instantiated.
    from openpi.shared.array_typing import disable_typechecking

    disable_typechecking().__enter__()

    jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))

    rng = jax.random.key(config.seed)
    _, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)

    logging.info("Step 1/3: Initializing model & loading pretrained weights...")
    train_state, state_sharding = init_train_state(config, init_rng, mesh)
    jax.block_until_ready(train_state)
    num_params = training_utils.count_parameters(train_state.params)
    logging.info(f"Total parameters: {num_params:,}")
    logging.info("Model init PASSED ✅")

    logging.info("Step 2/3: Saving checkpoint...")
    import orbax.checkpoint as ocp

    checkpointer = ocp.PyTreeCheckpointer()
    save_dir = ckpt_base / "model_init"
    os.makedirs(save_dir, exist_ok=True)

    save_args = jax.tree.map(lambda x: ocp.args.StandardSave(x), train_state)
    checkpointer.save(save_dir, train_state, save_args=save_args, force=True)
    logging.info(f"Checkpoint saved to: {save_dir}")
    logging.info("Save PASSED ✅")

    logging.info("=" * 60)
    logging.info("ALL DONE! Model initialized and saved successfully.")
    logging.info(f"Run training with: python scripts/train.py {config_name} --exp-name={exp_name} --resume")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
