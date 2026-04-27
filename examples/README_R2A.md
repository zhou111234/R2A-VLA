# R2A-VLA 示例代码说明

## 📂 examples/ 目录说明

由于 R2A-VLA 模型使用**特殊的输入格式**（3 相机 +4 帧历史 +32 维状态），原始 ACOT-VLA 的示例代码（LIBERO、ALOHA 等）**不能直接使用**。

### 当前提供的示例

#### simple_client/ - 简单客户端示例 ✅

**用途**：测试策略服务器连接和基本推理功能

**运行方法**：
```bash
# 1. 先启动策略服务器
uv run scripts/serve_policy.py --env G2SIM --port 8999

# 2. 在另一个终端运行简单客户端
uv run examples/simple_client/main.py --env R2A --num-steps 10
```

**输入格式**：
```python
{
    "image": {
        "base_0_rgb": (4, 224, 224, 3),       # 顶视相机，4 帧历史
        "left_wrist_0_rgb": (4, 224, 224, 3),  # 左手相机，4 帧历史
        "right_wrist_0_rgb": (4, 224, 224, 3)  # 右手相机，4 帧历史
    },
    "image_mask": {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": True,
    },
    "state": (32,),  # 32 维状态向量
    "prompt": "Pick up the block and place it into the box",
}
```

**输出格式**：
```python
{
    "actions": (30, 32),  # 30 步动作，每步 32 维
    "coarse_actions": (30, 32),  # 30 步粗动作
}
```

---

## 🚫 未提供的示例

以下示例由于**输入格式不匹配**，暂未提供：

### LIBERO 示例
- **原因**：LIBERO 使用 2 相机（主视 + 腕部），而 R2A-VLA 需要 3 相机
- **状态维度**：LIBERO 8 维 vs R2A-VLA 32 维
- **时间维度**：LIBERO 单帧 vs R2A-VLA 4 帧历史

### ALOHA 示例
- **原因**：ALOHA 使用 4 相机（高、低、左腕、右腕）
- **动作空间**：ALOHA 14 维 vs R2A-VLA 32 维

### DROID 示例
- **原因**：DROID 使用 2 相机（外部 + 腕部）
- **状态空间**：DROID 7 维关节 +1 维夹爪 vs R2A-VLA 32 维

---

## 🔧 如何为新平台创建示例

如果您想在其他机器人平台上使用 R2A-VLA，需要：

### 1. 定义输入转换

```python
from openpi.policies.r2a_temporal_policy import R2ATemporalInputs

# 定义您的机器人输入转换
class MyRobotInputs(R2ATemporalInputs):
    EXPECTED_CAMERAS = ("top_head", "hand_left", "hand_right")
    rename_map = {
        "top_head": "base_0_rgb",
        "hand_left": "left_wrist_0_rgb",
        "hand_right": "right_wrist_0_rgb",
    }
    
    def __call__(self, data: dict) -> dict:
        # 将您的机器人数据转换为 R2A-VLA 格式
        return super().__call__(data)
```

### 2. 定义策略包装器

```python
from openpi.policies.r2a_temporal_policy import TemporalBufferedPolicy

# 创建策略
policy = create_policy(...)

# 包装为时间缓冲策略（处理 4 帧历史）
wrapped_policy = TemporalBufferedPolicy(
    policy,
    T=4,
    camera_keys=["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
)
```

### 3. 运行推理

```python
# 准备观察数据（单帧）
obs = {
    "base_0_rgb": (224, 224, 3),  # 单帧
    "left_wrist_0_rgb": (224, 224, 3),
    "right_wrist_0_rgb": (224, 224, 3),
    "state": (32,),
    "prompt": "your task",
}

# TemporalBufferedPolicy 会自动处理为 4 帧历史
result = wrapped_policy.infer(obs)
actions = result["actions"]  # (30, 32)
```

---

## 📦 packages/openpi-client/ 说明

### ✅ 完全兼容 R2A-VLA

`openpi-client` 包提供**通用的 WebSocket 客户端**，与模型架构无关。

### 核心功能

#### 1. WebsocketClientPolicy
```python
from openpi_client import websocket_client_policy

# 连接远程策略服务器
client = websocket_client_policy.WebsocketClientPolicy(
    host="localhost",
    port=8999
)

# 运行推理
result = client.infer(obs)
```

#### 2. ActionChunkBroker
```python
from openpi_client import action_chunk_broker

# 缓存动作块，减少网络延迟影响
broker = action_chunk_broker.ActionChunkBroker(
    policy=client,
    chunk_size=30
)

# 逐步执行动作
for _ in range(30):
    action = broker.get_next_action()
    env.step(action)
```

#### 3. Runtime
```python
from openpi_client.runtime import runtime

# 完整的运行时环境
rt = runtime.Runtime(
    agent=policy_agent,
    environment=env,
    subscriber=subscriber
)
rt.run()
```

---

## 🎯 推荐使用方式

### 快速测试（无机器人）
```bash
# 使用 simple_client 测试
uv run examples/simple_client/main.py --env R2A
```

### 实际部署（有机器人）
1. 参考 `simple_client/main.py` 创建您的机器人接口
2. 使用 `openpi-client` 连接策略服务器
3. 确保输入格式符合 R2A-VLA 要求（3 相机 +4 帧 +32 维状态）

---

## 📝 总结

| 组件 | 兼容性 | 说明 |
|------|--------|------|
| **simple_client/** | ✅ 兼容 | 已修改为 R2A-VLA 格式 |
| **openpi-client/** | ✅ 完全兼容 | 通用 WebSocket 客户端 |
| **libero/** | ❌ 不兼容 | 输入格式差异大 |
| **aloha_*/** | ❌ 不兼容 | 输入格式差异大 |
| **droid/** | ❌ 不兼容 | 输入格式差异大 |

**建议**：从 `simple_client/` 开始，逐步适配您的机器人平台。
