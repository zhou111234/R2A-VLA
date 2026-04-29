"""
Policy transforms for the Reasoning2Action competition data with temporal frames.

Handles multi-frame (T, C, H, W) images from TemporalFrameWrapper and
converts them to (T, H, W, C) uint8/float for the model.

Placed in: src/openpi/policies/r2a_temporal_policy.py
"""

from collections.abc import Sequence
import copy
import dataclasses
from typing import ClassVar

import numpy as np
import torch

import openpi.transforms as transforms


@dataclasses.dataclass(frozen=True)
class R2ATemporalInputs(transforms.DataTransformFn):
    """
    Input transform for R2A competition data with temporal (multi-frame) images.

    Handles both single-frame (C,H,W) and multi-frame (T,C,H,W) image tensors.
    Always outputs (T, H, W, C) uint8/float regardless of input format.
    """

    action_dim: int
    state_mask: np.ndarray | None = None
    action_mask: np.ndarray | None = None
    prompt_map_inject_to_training: dict | None = None
    acot_action_generation: Sequence[Sequence[int]] | None = None

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("top_head", "hand_left", "hand_right")
    rename_map = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb",
    }

    def _convert_image(self, img):
        """Convert image from dataset format to model format (T, H, W, C).

        Handles:
          - (C, H, W) single frame → repeat T times → (T, H, W, C)
          - (T, C, H, W) multi frame → (T, H, W, C)
          - (H, W, C) single frame → (1, H, W, C)
          - (1, H, W, 3) → transpose to (1, H, W, C)
        """
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()

        if img.ndim == 2:
            # (H, W) or (W, C) — treat as single frame, add time dim
            if img.shape[-1] == 3 or (img.ndim == 3 and img.shape[0] != 3):
                img = img[np.newaxis]  # (1, H, W, C)
            else:
                img = img.transpose(1, 2, 0)[np.newaxis]  # (1, C, H, W) -> (1, H,, W, C)
        elif img.ndim == 3:
            if img.shape[-1] == 3 or (img.shape[0] == 3 and img.shape[2] != 3):
                img = img[np.newaxis]  # (1, H, W, C)
            else:
                img = np.transpose(img, (0, 2, 3, 1))  # (C, H, W) -> (1, H, W, C)
        elif img.ndim == 4:
            # Already has time dim, just ensure dtype and shape
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * np.clip(img, 0, 1)).astype(np.uint8)
            # Only transpose if input is (T, C, H, W); skip if already (T, H, W, C)
            if img.shape[1] == 3 and img.shape[-1] != 3:
                img = np.transpose(img, (0, 2, 3, 1))
        return img

    def _slice_state_and_action(self, data):
        """Slice state/action dimensions based on dataset format."""
        state_len = len(data["state"])
        state_indices = None
        if state_len == 183:
            state_indices = list(range(54, 68)) + [0, 1] + list(range(99, 104))
        elif state_len == 159:
            state_indices = list(range(30, 44)) + [0, 1] + list(range(75, 80))
        if state_indices is not None:
            data["state"] = data["state"][state_indices]
        if "actions" in data and data["actions"].shape[1] == 40:
            data["actions"] = np.column_stack(
                (
                    data["actions"][:, 16:30],
                    data["actions"][:, 0:2],
                    data["actions"][:, 33:38],
                )
            )
        return data

    def _random_inject_prompt(self, data):
        if self.prompt_map_inject_to_training is None:
            return data
        task_name = data.get("task", "")
        if task_name in self.prompt_map_inject_to_training:
            prompt, prob = self.prompt_map_inject_to_training[task_name]
            if np.random.rand() < prob:
                data["prompt"] = prompt
        return data

    def __call__(self, data: dict) -> dict:
        data = self._slice_state_and_action(data)
        state = copy.deepcopy(transforms.pad_to_dim(data["state"], self.action_dim))
        if self.state_mask is not None:
            state[np.array(self.state_mask)] = 0

        images = {}
        for camera in self.EXPECTED_CAMERAS:
            if camera in data["images"]:
                images[self.rename_map[camera]] = self._convert_image(data["images"][camera])
            else:
                raise ValueError(f"Camera {camera} not found in data")

        image_mask = {self.rename_map[c]: np.True_ for c in self.EXPECTED_CAMERAS}

        inputs = {
            "image": images,
            "image_mask": image_mask,
            "state": state,
        }

        if self.acot_action_generation is not None and "actions" in data:
            action_horizons = self.acot_action_generation[0]
            joint_action_shifts = self.acot_action_generation[1]
            raw = data["actions"]
            for idx, key in enumerate(["coarse_actions", "actions"]):
                ah = action_horizons[idx]
                js = joint_action_shifts[idx]
                required = (ah - 1) * js + 1
                data[key] = copy.deepcopy(raw[:required:js])
                assert len(data[key]) == ah

            for key in ["coarse_actions", "actions"]:
                if key in data:
                    if self.action_mask is not None:
                        data[key][:, np.array(self.action_mask)[: data[key].shape[1]]] = 0
                    data[key] = transforms.pad_to_dim(data[key], self.action_dim)
                    inputs[key] = data[key]

        if "task" in data:
            data = self._random_inject_prompt(data)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class R2ATemporalOutputs(transforms.DataTransformFn):
    """Output transform: pass through full action predictions (no truncation).

    Action dimension alignment is handled by post_process / policy config,
    not by hard-coding a slice here.
    """

    def __call__(self, data: dict) -> dict:
        return {key: np.asarray(data[key]) for key in ["coarse_actions", "actions"] if key in data}


# ---------------------------------------------------------------------------
# TemporalBufferedPolicy — wraps a Policy to handle single-frame Genie3 observations
# and stack them into (T, H, W, C) for the model's temporal input.
# Only active when config.num_history_frames > 1.
# ---------------------------------------------------------------------------

from openpi.policies.temporal_policy_server import TemporalFrameBuffer


class TemporalBufferedPolicy:
    """
    Wraps a trained Policy so that Genie3's single-frame observations
    are buffered into multi-frame (T, H, W, C) before inference.

    This is transparent to the caller — infer() still takes/returns
    the same single-frame dict format.

    Automatically detects episode boundaries via task_name changes
    and resets the temporal frame buffer.
    """

    def __init__(self, inner_policy, T: int = 4, camera_keys=None):
        self._inner = inner_policy
        self.T = T
        self._buffer = TemporalFrameBuffer(
            num_history_frames=T,
            camera_keys=camera_keys
            or [
                "base_0_rgb",
                "left_wrist_0_rgb",
                "right_wrist_0_rgb",
            ],
        )
        self._last_task_name: str | None = None

    def reset(self, episode_id=None):
        """Clear frame buffer (call at episode boundaries)."""
        self._buffer.reset(episode_id)
        self._last_task_name = None

    @property
    def metadata(self):
        return self._inner.metadata

    def infer(self, obs: dict) -> dict:
        """Accepts single-frame obs from Genie3, returns same-format output."""
        current_task = obs.get("task", obs.get("task_name"))
        if current_task is not None and current_task != self._last_task_name:
            self._buffer.reset()
            self._last_task_name = current_task
        obs = self._buffer.update(obs)
        result = self._inner.infer(obs)
        return result
