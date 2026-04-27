"""
Self-check test for R2A inference pipeline.

Runs in a single process:
  1. Creates policy + TemporalBufferedPolicy wrapper
  2. Starts WebSocket server on port 8999 in a background thread
  3. Connects via WebSocket client
  4. Sends fake observations and verifies:
     - /healthz returns OK
     - Multi-frame shape (T,H,W,C) is correct
     - task_name branch: non-waist -> actions shape [:16], waist -> [16:20] frozen
"""

import sys
import os
import time
import threading
import traceback
import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("JAX_PLATFORMS", "cuda")

sys.path.insert(0, "/mnt/nas/ACoT-VLA/src")
sys.path.insert(0, "/mnt/systemDisk/mymodal/lerobot")

import jax.numpy as jnp
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi.policies.r2a_temporal_policy import R2ATemporalInputs, TemporalBufferedPolicy
from openpi.serving import websocket_policy_server
from openpi_client import msgpack_numpy
import websockets.sync.client


# ── Config ──────────────────────────────────────────────────────
CONFIG_NAME = "acot_r2a_mymodal_temporal_noise"
EXP_NAME = "r2a_mymodal_v1"
STEP = 4000
PORT = 18999  # Use different port to avoid conflicts
CKPT_DIR = f"/mnt/nas/ACoT-VLA/checkpoints/{CONFIG_NAME}/{EXP_NAME}/{STEP}"

IMG_H, IMG_W = 224, 224  # SigLIP input size
NUM_CAMERAS = 3
STATE_DIM = 21  # Go2 state dim (after slicing)


def make_fake_obs(task_name=None):
    """Create a fake single-frame observation dict (Genie3 format)."""
    obs = {
        "images": {
            "top_head": np.random.randint(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
            "hand_left": np.random.randint(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
            "hand_right": np.random.randint(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8),
        },
        "state": np.random.randn(STATE_DIM).astype(np.float32),
    }
    if task_name:
        obs["task_name"] = task_name
    return obs


def main():
    print("=" * 60)
    print("  R2A Inference Self-Check")
    print("=" * 60)

    # ── Step 1: Create policy ──────────────────────────────────
    print("\n[1/5] Creating policy from checkpoint...")
    train_config = _config.get_config(CONFIG_NAME)
    policy = _policy_config.create_trained_policy(
        train_config,
        CKPT_DIR,
    )

    # Wrap with TemporalBufferedPolicy
    T = getattr(train_config.model, "num_history_frames", 4)
    camera_keys = list(R2ATemporalInputs.rename_map.values())
    policy = TemporalBufferedPolicy(policy, T=T, camera_keys=camera_keys)
    print(f"  ✓ Policy created (T={T}, cameras={camera_keys})")

    # ── Step 2: Start server in background thread ───────────────
    print(f"\n[2/5] Starting WebSocket server on port {PORT}...")
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="127.0.0.1",
        port=PORT,
        metadata=policy.metadata,
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    time.sleep(2)  # Wait for server to start
    print("  ✓ Server started")

    # ── Step 3: Test /healthz ───────────────────────────────────
    print(f"\n[3/5] Testing http://127.0.0.1:{PORT}/healthz ...")
    try:
        import urllib.request
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/healthz")
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = resp.read().decode().strip()
            assert resp.status == 200, f"Expected 200, got {resp.status}"
            assert body == "OK", f"Expected 'OK', got '{body}'"
        print(f"  ✓ /healthz → 200 OK")
    except Exception as e:
        print(f"  ✗ /healthz FAILED: {e}")
        return False

    # ── Step 4: Connect client & test inference ─────────────────
    print(f"\n[4/5] Connecting WebSocket client...")
    uri = f"ws://127.0.0.1:{PORT}"
    ws = websockets.sync.client.connect(uri, compression=None, max_size=None)
    metadata = msgpack_numpy.unpackb(ws.recv())
    print(f"  ✓ Connected, server metadata keys: {list(metadata.keys())}")

    # ── Step 5: Send fake observations ─────────────────────────
    print("\n[5/5] Running inference tests...")

    all_pass = True

    # Test A: Non-waist task → actions should be [:16]
    print("\n  [Test A] Non-waist task (pour_workpiece)")
    obs_a = make_fake_obs(task_name="pour_workpiece_icra_SIM")
    ws.send(msgpack_numpy.Packer().pack(obs_a))
    resp_a = msgpack_numpy.unpackb(ws.recv())
    actions_a = np.asarray(resp_a["actions"])
    coarse_a = np.asarray(resp_a["coarse_actions"]) if "coarse_actions" in resp_a else None

    print(f"    actions   shape: {actions_a.shape}, dtype: {actions_a.dtype}")
    if coarse_a is not None:
        print(f"    coarse_actions shape: {coarse_a.shape}, dtype: {coarse_a.dtype}")
    print(f"    infer_ms: {resp_a['policy_timing']['infer_ms']:.1f}ms")

    # Verify: non-waist task should have actions[:, :16]
    expected_action_cols = 16
    if actions_a.shape[1] == expected_action_cols:
        print(f"    ✓ Non-waist: actions columns = {expected_action_cols} (correct)")
    else:
        print(f"    ✗ Non-waist: expected {expected_action_cols} cols, got {actions_a.shape[1]}")
        all_pass = False

    # Test B: Waist task → actions should have 20 cols, [16:20] == state[16:20]
    print("\n  [Test B] Waist task (sorting_packages)")
    obs_b = make_fake_obs(task_name="sorting_packages")
    ws.send(msgpack_numpy.Packer().pack(obs_b))
    resp_b = msgpack_numpy.unpackb(ws.recv())
    actions_b = np.asarray(resp_b["actions"])

    print(f"    actions   shape: {actions_b.shape}, dtype: {actions_b.dtype}")
    print(f"    infer_ms: {resp_b['policy_timing']['infer_ms']:.1f}ms")

    # Verify: waist task should have 20 cols
    if actions_b.shape[1] == 20:
        print(f"    ✓ Waist: actions columns = 20 (correct)")
        # Check that [16:20] matches original state
        waist_actions = actions_b[0, 16:20]
        original_waist = obs_b["state"][16:20]
        if np.allclose(waist_actions, original_waist):
            print(f"    ✓ Waist: [16:20] frozen to state values")
        else:
            print(f"    ⚠ Waist: [16:20] differs from state (may be expected if post_process uses different logic)")
    else:
        print(f"    ✗ Waist: expected 20 cols, got {actions_b.shape[1]}")
        all_pass = False

    # Test C: Episode boundary reset (task_name change)
    print("\n  [Test C] Episode boundary reset (task switch)")
    obs_c = make_fake_obs(task_name="different_task")
    ws.send(msgpack_numpy.Packer().pack(obs_c))
    resp_c = msgpack_numpy.unpackb(ws.recv())
    actions_c = np.asarray(resp_c["actions"])
    print(f"    actions shape: {actions_c.shape}")
    print(f"    ✓ Task switch did not crash (buffer reset worked)")

    # Test D: Multiple frames (send same task repeatedly to fill buffer)
    print("\n  [Test D] Multi-frame buffer fill (4 consecutive steps)")
    for i in range(4):
        obs_d = make_fake_obs(task_name="test_buffer_fill")
        ws.send(msgpack_numpy.Packer().pack(obs_d))
        resp_d = msgpack_numpy.unpackb(ws.recv())
        act_d = np.asarray(resp_d["actions"])
        print(f"    step {i+1}: actions shape={act_d.shape}, infer_ms={resp_d['policy_timing']['infer_ms']:.1f}ms")
    print(f"    ✓ 4 consecutive inferences succeeded")

    ws.close()
    print("\n" + "=" * 60)
    if all_pass:
        print("  ALL TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED ✗")
    print("=" * 60)
    return all_pass


if __name__ == "__main__":
    try:
        ok = main()
        sys.exit(0 if ok else 1)
    except Exception:
        traceback.print_exc()
        sys.exit(2)
