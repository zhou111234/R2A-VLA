"""
R2A to LIBERO Policy Adapter.

This module adapts R2A checkpoint (trained on Go2/Reasoning2Action data) to work with LIBERO benchmark.

Key adaptations:
1. Input transform: LIBERO observations -> R2A model expected format
2. Image handling: 2 cameras -> 3 cameras (duplicate wrist for right_wrist)
3. Temporal handling: Single-frame -> 4-frame history (repeat for compatibility)
4. State padding: LIBERO 8-dim state -> R2A 32-dim state
5. Action output: R2A 32-dim -> LIBERO 7-dim (slice first 7 dims)

Usage:
    from openpi.policies.libero_r2a_policy import LiberoR2AInputs, LiberoR2AOutputs
    
    # In serve_policy.py:
    inputs_transform = LiberoR2AInputs(
        action_dim=32,
        num_history_frames=4,
    )
"""

import dataclasses
import copy
import numpy as np
from collections.abc import Sequence
from typing import ClassVar

import openpi.transforms as transforms
from openpi.policies.r2a_temporal_policy import R2ATemporalInputs


@dataclasses.dataclass(frozen=True)
class LiberoR2AInputs(transforms.DataTransformFn):
    """
    Input transform for adapting LIBERO observations to R2A model format.
    
    Handles:
    - Image format conversion: LIBERO (H,W,C) -> R2A expected format
    - Camera slot filling: 2 LIBERO cameras -> 3 R2A cameras
    - Temporal framing: Single-frame -> Multi-frame (T=4)
    - State padding: 8-dim -> 32-dim
    
    LIBERO provides:
    - observation/image: (224, 224, 3) uint8 - agentview camera
    - observation/wrist_image: (224, 224, 3) uint8 - wrist camera
    - observation/state: (8,) float32 - robot state (7 joint + gripper)
    - prompt: str - language instruction
    
    R2A model expects:
    - image.base_0_rgb: (T, H, W, 3) uint8 - base camera
    - image.left_wrist_0_rgb: (T, H, W, 3) uint8 - left wrist
    - image.right_wrist_0_rgb: (T, H, W, 3) uint8 - right wrist
    - state: (32,) float32 - padded state
    - prompt: str
    """

    action_dim: int = 32
    num_history_frames: int = 4
    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None
    prompt_map_inject_to_training: dict | None = None
    
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")
    rename_map: ClassVar[dict[str, str]] = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb",
    }

    def _convert_image_to_uint8(self, img: np.ndarray) -> np.ndarray:
        """Convert image to uint8 format (H, W, C)."""
        if np.issubdtype(img.dtype, np.floating):
            img = (255 * np.clip(img, 0, 1)).astype(np.uint8)
        # Ensure (H, W, C) format
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img

    def _create_temporal_frames(self, img: np.ndarray) -> np.ndarray:
        """
        Create temporal frames by repeating single image T times.
        
        Args:
            img: Single image (H, W, C)
            
        Returns:
            Temporal frames (T, H, W, C)
        """
        # Add time dimension and repeat
        frames = np.stack([img] * self.num_history_frames, axis=0)
        return frames

    def __call__(self, data: dict) -> dict:
        # Extract LIBERO observations
        base_image = self._convert_image_to_uint8(data["observation/image"])
        wrist_image = self._convert_image_to_uint8(data["observation/wrist_image"])
        
        # LIBERO state is 8-dim (7 joint angles + 1 gripper)
        # R2A expects 32-dim, so we pad with zeros
        libero_state = data["observation/state"]
        assert len(libero_state) == 8, f"Expected LIBERO state to be 8-dim, got {len(libero_state)}"
        
        # Pad state to 32 dimensions
        state = np.zeros(self.action_dim, dtype=np.float32)
        state[:8] = libero_state
        
        # Apply state mask if provided (for ablation studies)
        if self.state_mask is not None:
            state[np.array(self.state_mask)] = 0

        # Create temporal frames for each camera
        # LIBERO has 2 cameras, R2A expects 3
        # Strategy: Use base_image for top_head, wrist_image for both left and right
        images = {}
        
        # Base camera (top_head) <- LIBERO agentview
        images[self.rename_map["top_head"]] = self._create_temporal_frames(base_image)
        
        # Left wrist <- LIBERO wrist_image
        images[self.rename_map["hand_left"]] = self._create_temporal_frames(wrist_image)
        
        # Right wrist <- Duplicate of LIBERO wrist_image
        # (R2A trained with bimanual data, but LIBERO is single-arm)
        images[self.rename_map["hand_right"]] = self._create_temporal_frames(wrist_image)

        # Create image masks (all True since we have valid images)
        image_mask = {self.rename_map[c]: np.True_ for c in self.EXPECTED_CAMERAS}

        # Build inputs dict
        inputs = {
            "image": images,
            "image_mask": image_mask,
            "state": state,
        }

        # Add prompt
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        # Apply prompt injection if configured (for training data augmentation)
        if self.prompt_map_inject_to_training is not None and "task" in data:
            task_name = data.get("task", "")
            if task_name in self.prompt_map_inject_to_training:
                prompt, prob = self.prompt_map_inject_to_training[task_name]
                if np.random.rand() < prob:
                    inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoR2AOutputs(transforms.DataTransformFn):
    """
    Output transform for adapting R2A model outputs to LIBERO action format.
    
    R2A model outputs:
    - actions: (action_horizon=30, action_dim=32) float32
    - coarse_actions: (coarse_action_horizon=50, action_dim=32) float32 (optional)
    
    LIBERO expects:
    - actions: (action_horizon=30, 7) float32
      - First 6 dims: end-effector pose (x, y, z, roll, pitch, yaw)
      - 7th dim: gripper open/close
    
    Note: This assumes R2A action space is structured similarly to LIBERO.
    If the semantic mapping is different, additional transformation may be needed.
    """

    def __call__(self, data: dict) -> dict:
        result = {}
        
        # Extract first 7 dimensions from actions
        if "actions" in data:
            actions = np.asarray(data["actions"])
            # Slice: keep all time steps, first 7 action dimensions
            result["actions"] = actions[:, :7]
        
        # Also process coarse_actions if present
        if "coarse_actions" in data:
            coarse_actions = np.asarray(data["coarse_actions"])
            result["coarse_actions"] = coarse_actions[:, :7]
        
        return result


@dataclasses.dataclass(frozen=True)
class LiberoR2AWithActionMapping(LiberoR2AOutputs):
    """
    Enhanced output transform with explicit action space mapping.
    
    R2A action space (32-dim) structure (based on Go2 robot):
    - [0:6]: End-effector pose (x, y, z, roll, pitch, yaw)
    - [6]: Gripper (left)
    - [7:14]: Left arm joint positions (7 DOF)
    - [14]: Gripper (right)
    - [15:22]: Right arm joint positions (7 DOF)
    - [22:27]: Waist/head/chassis (5 DOF)
    - [27:32]: Reserved/padding
    
    LIBERO action space (7-dim):
    - [0:6]: End-effector pose (x, y, z, roll, pitch, yaw)
    - [6]: Gripper
    
    This mapping extracts the relevant dimensions for LIBERO.
    """
    
    # Mapping from R2A action indices to LIBERO action indices
    # R2A [0:6] -> LIBERO [0:6] (end-effector pose)
    # R2A [6] -> LIBERO [6] (gripper)
    r2a_to_libero_indices: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    
    def __call__(self, data: dict) -> dict:
        result = {}
        
        if "actions" in data:
            actions = np.asarray(data["actions"])
            # Extract specific indices for LIBERO compatibility
            result["actions"] = actions[:, self.r2a_to_libero_indices]
        
        if "coarse_actions" in data:
            coarse_actions = np.asarray(data["coarse_actions"])
            result["coarse_actions"] = coarse_actions[:, self.r2a_to_libero_indices]
        
        return result
