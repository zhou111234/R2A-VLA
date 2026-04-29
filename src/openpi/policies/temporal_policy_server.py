"""
Temporal frame buffer for the policy server.

During inference (Genie Sim evaluation), the environment sends one observation
at a time. This wrapper maintains a circular buffer of the last T frames per
camera and stacks them before passing to the model.

Placed in: src/openpi/policies/temporal_policy_server.py

Usage: Wrap the standard policy server's predict function.
"""

import collections
import copy
from typing import Any

import numpy as np


class TemporalFrameBuffer:
    """
    Maintains per-camera frame history for temporal inference.

    At each step, call `update(observation)` to store the current frame,
    then `get_temporal_observation()` to retrieve an observation dict with
    multi-frame images stacked as (T, H, W, C).
    """

    def __init__(self, num_history_frames: int = 4, camera_keys=None):
        self.T = num_history_frames
        self.camera_keys = camera_keys or ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
        self._buffers: dict[str, collections.deque] = {k: collections.deque(maxlen=self.T) for k in self.camera_keys}
        self._episode_id: Any = None

    def reset(self, episode_id: Any = None):
        """Clear buffers (call at episode boundaries)."""
        for buf in self._buffers.values():
            buf.clear()
        self._episode_id = episode_id

    def update(self, observation: dict) -> dict:
        """
        Push current observation's images into the buffer and return
        a modified observation with multi-frame images.

        Args:
            observation: Dict with keys like "image"/{camera_key}: (H,W,C) arrays

        Returns:
            Modified observation with "image"/{camera_key}: (T, H, W, C)
        """
        obs = copy.copy(observation)
        images = obs.get("image", obs.get("images", {}))
        new_images = {}

        for cam_key in self.camera_keys:
            if cam_key not in images:
                continue
            frame = np.asarray(images[cam_key])
            self._buffers[cam_key].append(frame)

            buf = self._buffers[cam_key]
            if len(buf) < self.T:
                # Pad with repeats of the oldest available frame
                pad = [buf[0]] * (self.T - len(buf))
                frames = pad + list(buf)
            else:
                frames = list(buf)

            new_images[cam_key] = np.stack(frames, axis=0)  # (T, H, W, C)

        if "image" in obs:
            obs["image"] = new_images
        elif "images" in obs:
            obs["images"] = new_images

        return obs
