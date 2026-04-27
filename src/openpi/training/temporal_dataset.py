"""
Temporal frame wrapper for LeRobot datasets.

Wraps a standard LeRobot dataset so that each __getitem__ returns
multi-frame images (T, C, H, W) instead of single-frame (C, H, W)
by loading previous frames from the same episode.

Placed in: src/openpi/training/temporal_dataset.py
"""

import logging

import numpy as np
import torch

logger = logging.getLogger("temporal_dataset")


class TemporalFrameWrapper:
    """
    Wraps a LeRobot dataset to provide history frames.

    For each sample at dataset index *i*, it also loads frames at
    i-1, i-2, ..., i-(T-1) from the same episode and stacks them along
    a new leading dimension for every image/video key.

    When the requested frame is before the episode start, the current
    frame is repeated (zero-padding would hurt SigLIP encoding).
    """

    def __init__(self, dataset, num_history_frames: int = 4, camera_keys=None):
        self.dataset = dataset
        self._dataset = dataset  # alias for get_base_dataset() traversal compatibility
        self.T = num_history_frames
        self.camera_keys = camera_keys  # auto-detected if None
        self._episode_starts = self._build_episode_starts()
        if self.camera_keys is None:
            self.camera_keys = self._detect_camera_keys()

    # ---- index helpers --------------------------------------------------

    def _build_episode_starts(self) -> dict[int, int]:
        """Map episode_index → first dataset index of that episode."""
        hf = self._get_hf_dataset()
        if hf is None:
            return {}
        starts: dict[int, int] = {}
        for i in range(len(hf)):
            ep = self._to_int(hf[i].get("episode_index", 0))
            if ep not in starts:
                starts[ep] = i
        return starts

    def _get_hf_dataset(self):
        """Robustly fetch the underlying HF dataset."""
        if hasattr(self.dataset, "hf_dataset"):
            return self.dataset.hf_dataset
        if hasattr(self.dataset, "_dataset") and hasattr(self.dataset._dataset, "hf_dataset"):
            return self.dataset._dataset.hf_dataset
        return None

    def _detect_camera_keys(self):
        """Auto-detect image keys from first sample."""
        try:
            sample = self.dataset[0]
        except Exception:
            return []
        keys = []
        for k, v in sample.items():
            if not isinstance(v, (torch.Tensor, np.ndarray)):
                continue
            if isinstance(v, torch.Tensor) and v.ndim >= 3:
                if v.shape[0] == 3 or v.shape[-1] == 3:
                    keys.append(k)
            elif isinstance(v, np.ndarray) and v.ndim >= 3:
                if v.shape[0] == 3 or v.shape[-1] == 3:
                    keys.append(k)
        logger.info(f"TemporalFrameWrapper detected camera keys: {keys}")
        return keys

    @staticmethod
    def _to_int(x):
        if hasattr(x, "item"):
            return x.item()
        return int(x)

    # ---- core -----------------------------------------------------------

    def __getitem__(self, index):
        item = self.dataset[index]
        if self.T <= 1 or not self.camera_keys:
            return item

        ep = self._to_int(item.get("episode_index", 0))
        ep_start = self._episode_starts.get(ep, 0)

        for cam_key in self.camera_keys:
            if cam_key not in item:
                continue
            current = item[cam_key]
            frames = []
            for t in range(self.T - 1, 0, -1):
                hist_idx = index - t
                if hist_idx >= ep_start:
                    try:
                        hist = self.dataset[hist_idx]
                        hist_ep = self._to_int(hist.get("episode_index", -1))
                        if hist_ep == ep and cam_key in hist:
                            frames.append(hist[cam_key])
                            continue
                    except Exception:
                        pass
                frames.append(self._clone(current))
            frames.append(current)

            if isinstance(current, torch.Tensor):
                item[cam_key] = torch.stack(frames, dim=0)
            else:
                item[cam_key] = np.stack(frames, axis=0)
        return item

    @staticmethod
    def _clone(x):
        if isinstance(x, torch.Tensor):
            return x.clone()
        return x.copy()

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name in ("dataset", "_dataset", "T", "camera_keys", "_episode_starts"):
            raise AttributeError
        return getattr(self.dataset, name)
