#!/usr/bin/env python3
"""
Test script to verify R2A checkpoint can be loaded and served for LIBERO evaluation.

Usage:
    python scripts/test_libero_r2a_service.py
"""

import logging
import pathlib
import sys

# Add project root to path
project_root = pathlib.Path(__file__).parent
sys.path.insert(0, str(project_root))

import numpy as np

from openpi.policies import policy_config as _policy_config
from openpi.policies.libero_r2a_policy import LiberoR2AInputs
from openpi.policies.libero_r2a_policy import LiberoR2AOutputs
from openpi.training import config as _config
import openpi.transforms as transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def test_config_loading():
    """Test that the config can be loaded."""
    logger.info("Testing config loading...")
    try:
        config = _config.get_config("acot_r2a_libero_eval_v2")
        logger.info(f"✓ Config loaded: {config.name}")
        logger.info(f"  - Model: {config.model}")
        logger.info(f"  - Action dim: {config.model.action_dim}")
        logger.info(f"  - Action horizon: {config.model.action_horizon}")
        logger.info(f"  - Num history frames: {config.model.num_history_frames}")
        return config
    except Exception as e:
        logger.error(f"✗ Config loading failed: {e}")
        return None


def test_input_transform():
    """Test the LIBERO to R2A input transform."""
    logger.info("Testing input transform...")

    # Create a fake LIBERO observation
    liberto_obs = {
        "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(8).astype(np.float32),
        "prompt": "pick up the block",
    }

    try:
        # Create transform
        transform = LiberoR2AInputs(
            action_dim=32,
            num_history_frames=4,
        )

        # Apply transform
        result = transform(liberto_obs)

        # Check outputs
        assert "image" in result
        assert "state" in result
        assert "prompt" in result

        # Check image shapes
        for cam_key in ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]:
            assert cam_key in result["image"], f"Missing camera: {cam_key}"
            img = result["image"][cam_key]
            assert img.ndim == 4, f"Expected 4D (T,H,W,C), got {img.ndim}D"
            assert img.shape[0] == 4, f"Expected T=4, got {img.shape[0]}"
            logger.info(f"  ✓ {cam_key}: {img.shape}")

        # Check state shape
        assert result["state"].shape[0] == 32, f"Expected state dim 32, got {result['state'].shape[0]}"
        logger.info(f"  ✓ state: {result['state'].shape}")

        # Check prompt
        assert result["prompt"] == "pick up the block"
        logger.info(f"  ✓ prompt: {result['prompt']}")

        logger.info("✓ Input transform test passed!")
        return True
    except Exception as e:
        logger.error(f"✗ Input transform test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_output_transform():
    """Test the R2A to LIBERO output transform."""
    logger.info("Testing output transform...")

    # Create fake R2A model outputs
    r2a_output = {
        "actions": np.random.randn(30, 32).astype(np.float32),  # (action_horizon, action_dim)
        "coarse_actions": np.random.randn(50, 32).astype(np.float32),  # (coarse_action_horizon, action_dim)
    }

    try:
        # Create transform
        transform = LiberoR2AOutputs()

        # Apply transform
        result = transform(r2a_output)

        # Check outputs
        assert "actions" in result
        assert "coarse_actions" in result

        # Check action shapes
        assert result["actions"].shape == (30, 7), f"Expected (30, 7), got {result['actions'].shape}"
        assert result["coarse_actions"].shape == (50, 7), f"Expected (50, 7), got {result['coarse_actions'].shape}"

        logger.info(f"  ✓ actions: {result['actions'].shape}")
        logger.info(f"  ✓ coarse_actions: {result['coarse_actions'].shape}")
        logger.info("✓ Output transform test passed!")
        return True
    except Exception as e:
        logger.error(f"✗ Output transform test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_policy_creation():
    """Test creating a policy with the checkpoint."""
    logger.info("Testing policy creation...")

    checkpoint_dir = pathlib.Path("/mnt/nas/R2A-VLA/checkpoints/acot_r2a_mymodal_temporal_noise/r2a_mymodal_v1/4000")

    if not checkpoint_dir.exists():
        logger.error(f"✗ Checkpoint directory not found: {checkpoint_dir}")
        return False

    try:
        config = _config.get_config("acot_r2a_libero_eval_v2")

        # Create repack transforms for LIBERO
        repack_transforms = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "images": {
                            "top_head": "observation.image",
                            "hand_left": "observation.wrist_image",
                            "hand_right": "observation.wrist_image",
                        },
                        "state": "observation.state",
                        "prompt": "prompt",
                    }
                ),
                LiberoR2AInputs(action_dim=32, num_history_frames=4),
            ],
            outputs=[
                LiberoR2AOutputs(),
            ],
        )

        logger.info("  Creating policy (this may take a minute)...")
        policy = _policy_config.create_trained_policy(
            config,
            checkpoint_dir,
            repack_transforms=repack_transforms,
        )

        logger.info("✓ Policy created successfully!")
        logger.info(f"  - Policy metadata: {policy.metadata}")
        return policy
    except Exception as e:
        logger.error(f"✗ Policy creation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def test_policy_inference(policy):
    """Test running inference with the policy."""
    logger.info("Testing policy inference...")

    if policy is None:
        logger.error("✗ No policy provided")
        return False

    # Create a fake LIBERO observation
    liberto_obs = {
        "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.rand(8).astype(np.float32),
        "prompt": "pick up the block",
    }

    try:
        result = policy.infer(liberto_obs)

        logger.info("  ✓ Inference succeeded!")
        logger.info(f"  - Result keys: {result.keys()}")

        if "actions" in result:
            logger.info(f"  - actions shape: {result['actions'].shape}")
            assert result["actions"].shape[1] == 7, f"Expected 7 action dims, got {result['actions'].shape[1]}"

        if "coarse_actions" in result:
            logger.info(f"  - coarse_actions shape: {result['coarse_actions'].shape}")

        logger.info("✓ Policy inference test passed!")
        return True
    except Exception as e:
        logger.error(f"✗ Policy inference failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    logger.info("=" * 60)
    logger.info("R2A to LIBERO Service Test")
    logger.info("=" * 60)

    # Test 1: Config loading
    config = test_config_loading()
    if config is None:
        logger.error("Config loading failed - cannot continue")
        return False

    # Test 2: Input transform
    if not test_input_transform():
        logger.error("Input transform test failed")
        return False

    # Test 3: Output transform
    if not test_output_transform():
        logger.error("Output transform test failed")
        return False

    # Test 4: Policy creation
    policy = test_policy_creation()
    if policy is None:
        logger.error("Policy creation failed")
        return False

    # Test 5: Policy inference
    if not test_policy_inference(policy):
        logger.error("Policy inference test failed")
        return False

    logger.info("=" * 60)
    logger.info("All tests passed! ✓")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Start the policy server:")
    logger.info("   uv run scripts/serve_policy.py --env LIBERO policy:checkpoint \\")
    logger.info("       --policy.config acot_r2a_libero_eval_v2 \\")
    logger.info("       --policy.dir ./checkpoints/acot_r2a_mymodal_temporal_noise/r2a_mymodal_v1/4000")
    logger.info("")
    logger.info("2. In another terminal, run LIBERO client:")
    logger.info("   python examples/libero/main.py --task-suite-name libero_spatial")

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
