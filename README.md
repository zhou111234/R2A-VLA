# R2A-VLA: Reasoning-to-Action Vision-Language-Action Model

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-TBD-red.svg)](https://arxiv.org/)
[![AgiBot Challenge](https://img.shields.io/badge/AgiBot-ICRA2026-blue.svg)](https://agibot-world.com/challenge2026/)

**官方基线实现**：AgiBot World Challenge @ ICRA 2026 - Reasoning to Action Track

R2A-VLA 是一个端到端的视觉 - 语言 - 动作模型，专为机器人操作任务设计。本模型在 AgiBot Reasoning2Action 数据集上训练，支持从视觉观察和语言指令直接生成机器人关节动作。

---

## 📋 目录

- [核心特性](#-核心特性)
- [模型架构](#-模型架构)
- [安装指南](#-安装指南)
- [快速开始](#-快速开始)
- [训练指南](#-训练指南)
- [推理指南](#-推理指南)
- [Docker 部署](#-docker-部署)
- [数据集](#-数据集)
- [模型 Checkpoint](#-模型-checkpoint)
- [基准测试](#-基准测试)
- [故障排查](#-故障排查)
- [许可证](#-许可证)
- [引用](#-引用)
- [致谢](#-致谢)

---

## 🌟 核心特性

### 主要创新
- **Action Chain-of-Thought (ACoT)**: 将推理过程转化为结构化的动作意图，实现长视距策略学习
- **显式动作推理器 (EAR)**: 轻量级 Transformer，合成粗粒度运动轨迹，提供直接运动线索
- **隐式动作推理器 (IAR)**: 通过交叉注意力机制从 VLM 主干的内部表示中提取潜在动作先验

### 技术特点
- ✅ **端到端策略**: 从视觉 + 语言输入直接输出机器人关节动作
- ✅ **多相机支持**: 支持 3 个相机输入（顶视、左手、右手）
- ✅ **时间帧缓冲**: 使用 4 帧历史输入，捕捉时间动态
- ✅ **噪声专家**: 增强模型在不确定性下的决策能力
- ✅ **LoRA 微调**: 支持参数高效微调，减少显存需求
- ✅ **WebSocket 服务**: 支持远程推理，策略服务器与机器人分离部署

---

## 🔧 模型架构

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ 视觉输入        │     │ 语言输入         │     │ 状态输入        │
│ (3 相机×4 帧)    │     │ (自然语言指令)    │     │ (关节位置)      │
└────────┬────────┘     └────────┬─────────┘     └────────┬────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────────────────────────────────────────────┐
│              PaliGemma 2B + LoRA (冻结)                  │
│                    (视觉 - 语言编码器)                    │
└─────────────────────────┬───────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌────────────────┐ ┌──────────────┐ ┌──────────────┐
│ 显式动作推理器 │ │ 隐式动作推理器│ │   噪声专家   │
│   (EAR)        │ │   (IAR)      │ │  (可选)      │
└───────┬────────┘ └──────┬───────┘ └──────┬───────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          ▼
                 ┌─────────────────┐
                 │   动作输出      │
                 │ (32 维动作块)    │
                 │ (30 步预测)      │
                 └─────────────────┘
```

### 模型规格

| 组件 | 规格 |
|------|------|
| **VLM 主干** | PaliGemma 2B (gemma_2b_lora) |
| **动作维度** | 32 (14 关节×2 + 夹爪×2 + 基座×4) |
| **动作范围** | 30 步 |
| **粗动作范围** | 30 步 |
| **历史帧数** | 4 帧 |
| **相机数量** | 3 (top_head, hand_left, hand_right) |
| **时间编码器** | 6 层 Transformer |
| **最大 token 长度** | 200 |
| **数据类型** | bfloat16 |

---

## 📦 安装指南

### 系统要求

| 模式 | GPU 显存 | 示例 GPU | 说明 |
|------|----------|----------|------|
| **推理** | > 8 GB | RTX 4090 | 运行预训练模型 |
| **微调 (LoRA)** | > 22.5 GB | RTX 4090 | 参数高效微调 |
| **微调 (全量)** | > 70 GB | A100 (80GB) | 全参数微调 |

**操作系统**: Ubuntu 22.04 已测试，其他 Linux 发行版可能需调整

### 步骤 1: 克隆仓库

```bash
git clone https://github.com/YOUR_USERNAME/R2A-VLA.git
cd R2A-VLA
git submodule update --init --recursive
```

### 步骤 2: 安装依赖

我们使用 [uv](https://docs.astral.sh/uv/) 管理 Python 依赖：

```bash
# 安装 uv (如果尚未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 同步依赖
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

**注意**: `GIT_LFS_SKIP_SMUDGE=1` 是必需的，用于正确拉取 LeRobot 依赖。

### 步骤 3: 验证安装

```bash
# 检查 Python 环境
uv run python -c "import openpi; print('OpenPI imported successfully')"

# 检查 JAX
uv run python -c "import jax; print(f'JAX devices: {jax.device_count()}')"
```

---

## 🚀 快速开始

### 推理示例（单步）

```python
from openpi.training import config as _config
from openpi.policies import policy_config
import numpy as np

# 加载配置
config = _config.get_config("acot_r2a_mymodal_temporal_noise")

# 加载 checkpoint
checkpoint_dir = "./checkpoints/acot_r2a_mymodal_temporal_noise/r2a_mymodal_v1/4000"

# 创建策略
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 准备输入（示例数据）- R2A-VLA 需要 3 相机 +4 帧历史
example = {
    "image": {
        "base_0_rgb": np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8),
        "left_wrist_0_rgb": np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8),
        "right_wrist_0_rgb": np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8),
    },
    "image_mask": {
        "base_0_rgb": True,
        "left_wrist_0_rgb": True,
        "right_wrist_0_rgb": True,
    },
    "state": np.random.randn(32).astype(np.float32),
    "prompt": "Pick up the block and place it into the box"
}

# 运行推理
action_chunk = policy.infer(example)["actions"]
print(f"Generated actions shape: {action_chunk.shape}")  # (30, 32)
```

### 测试推理（无需机器人）

我们提供了简单客户端示例，用于测试推理功能：

```bash
# 1. 启动策略服务器
export CUDA_VISIBLE_DEVICES=0
uv run scripts/serve_policy.py --env G2SIM --port 8999

# 2. 在另一个终端运行简单客户端
uv run examples/simple_client/main.py --env R2A --num-steps 20
```

**输入格式说明**：
```python
{
    "image": {
        "base_0_rgb": (4, 224, 224, 3),       # 顶视相机，4 帧历史
        "left_wrist_0_rgb": (4, 224, 224, 3),  # 左手相机，4 帧历史
        "right_wrist_0_rgb": (4, 224, 224, 3)  # 右手相机，4 帧历史
    },
    "image_mask": {...},  # 相机掩码
    "state": (32,),       # 32 维状态向量
    "prompt": "任务描述"
}
```

详见 [`examples/simple_client/README.md`](examples/simple_client/README.md) 和 [`examples/README_R2A.md`](examples/README_R2A.md)。

### 启动策略服务器

```bash
# 启动 WebSocket 策略服务器
export CUDA_VISIBLE_DEVICES=0
uv run scripts/serve_policy.py --env G2SIM --port 8999
```

服务器将在 `localhost:8999` 监听 WebSocket 连接。

---

## 📚 训练指南

### 1. 数据集准备

数据集需要转换为 LeRobot 格式。示例：

```bash
# 转换 LIBERO 数据（示例）
uv run examples/libero/convert_libero_data_to_lerobot.py --data_dir /path/to/libero
```

对于 Reasoning2Action 数据集，数据已位于 `/mnt/nas/Reasoning2Action-Sim/`。

### 2. 计算归一化统计

训练前需要计算数据集的归一化统计：

```bash
uv run scripts/compute_norm_stats.py --config-name acot_r2a_mymodal_temporal_noise
```

这将从数据集中采样约 10% 并计算状态和动作的归一化统计量。

### 3. 启动训练

```bash
# 单卡训练（推荐）
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py acot_r2a_mymodal_temporal_noise \
    --exp-name=my_experiment \
    --overwrite
```

**训练参数说明：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--exp-name` | 必需 | 实验名称 |
| `--overwrite` | 可选 | 覆盖已有 checkpoint |
| `--resume` | 可选 | 从断点恢复训练 |

### 4. 监控训练

训练日志会输出到：
- **控制台**: 实时训练进度
- **Weights & Biases**: `./wandb/` 目录（离线模式）
- **Checkpoint**: `./checkpoints/<config>/<exp_name>/<step>/`

**W&B 在线监控**（可选）：
```bash
export WANDB_API_KEY=your_api_key
uv run scripts/train.py acot_r2a_mymodal_temporal_noise --exp-name=my_experiment
```

### 5. 训练配置详情

当前配置 (`acot_r2a_mymodal_temporal_noise`)：

| 超参数 | 值 |
|--------|-----|
| **训练步数** | 50,000 |
| **批量大小** | 4 |
| **学习率** | 5e-5 (峰值) |
| **热身步数** | 10,000 |
| **衰减步数** | 1,000,000 |
| **保存间隔** | 1,000 步 |
| **日志间隔** | 200 步 |
| **优化器** | AdamW (β₁=0.9, β₂=0.95) |
| **梯度裁剪** | 1.0 |
| **EMA 衰减** | None |

### 6. 多 GPU 训练

```bash
# FSDP 训练（减少显存占用）
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
uv run scripts/train.py acot_r2a_mymodal_temporal_noise \
    --exp-name=my_experiment \
    --fsdp-devices=2
```

---

## 🔍 推理指南

### 方法 1: 本地推理

```python
from openpi.training import config as _config
from openpi.policies import policy_config

# 加载配置和 checkpoint
config = _config.get_config("acot_r2a_mymodal_temporal_noise")
checkpoint_dir = "./checkpoints/acot_r2a_mymodal_temporal_noise/r2a_mymodal_v1/4000"

# 创建策略
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 推理
example = {...}  # 准备输入
result = policy.infer(example)
actions = result["actions"]
```

### 方法 2: 远程推理（WebSocket）

**步骤 1: 启动策略服务器**
```bash
export CUDA_VISIBLE_DEVICES=0
uv run scripts/serve_policy.py --env G2SIM --port 8999
```

**步骤 2: 客户端连接**
```python
import asyncio
import websockets
import json

async def query_policy(observations):
    async with websockets.connect("ws://localhost:8999") as ws:
        await ws.send(json.dumps(observations))
        response = await ws.recv()
        return json.loads(response)

# 使用示例
actions = asyncio.run(query_policy(observations))
```

### 方法 3: 使用 GenieSim 评估

```bash
# 启动策略服务器
bash scripts/serve_r2a_mymodal.sh 0 8999

# 在另一个终端，启动 GenieSim 评估
cd /path/to/genie_sim
bash scripts/run_minimal.sh
# 在容器内运行评估
bash scripts/run_official_eval.sh
```

---

## 🐳 Docker 部署

### 构建 Docker 镜像

```bash
# 构建推理镜像
docker build -f Dockerfile.r2a -t r2a-vla:latest .

# 或使用构建脚本
bash build_docker.sh
```

### 运行 Docker 容器

```bash
# 运行推理服务器
docker run -d --gpus all --network=host \
    -e STEP=4000 \
    -e PORT=8999 \
    -v /path/to/checkpoints:/app/checkpoints \
    r2a-vla:latest
```

### Docker 配置说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `STEP` | Checkpoint 步数 | 4000 |
| `PORT` | WebSocket 端口 | 8999 |
| `CUDA_VISIBLE_DEVICES` | GPU 选择 | 0 |

---

## 📊 数据集

### 支持的数据集格式

- **LeRobot**: 主要支持的数据格式
- **RLDS**: 通过适配器支持

### Reasoning2Action 数据集

包含 13 个长视距操作任务：

| 任务 ID | 任务名称 | 说明 |
|--------|----------|------|
| 1 | Pouring workpieces | 倒工件（单臂） |
| 2 | Opening a door | 开门（单臂） |
| 3 | Scooping popcorn | 舀爆米花（单臂） |
| 4 | Carrying a pot | 端锅（双臂） |
| 5 | Grabbing toys | 抓玩具（双臂） |
| 6 | Supermarket item retrieval | 超市取物（双臂） |
| 7 | Supermarket restocking | 超市补货（双臂） |
| 8 | Packages sorting | 包裹分类（单臂 + 腰部） |
| 9 | Arranging the table | 整理桌面（双臂） |

**数据访问**: 数据位于 `/mnt/nas/Reasoning2Action-Sim/dataset_without_depth/`

### 数据转换示例

```python
# 转换自定义数据到 LeRobot 格式
from lerobot.common.datasets.lerobot_dataset import LeroNetDataset

dataset = LeroNetDataset.create(
    repo_id="my_dataset",
    robot_type="agibot_genie",
    fps=30,
)
```

---

## 🏆 模型 Checkpoint

### 可用 Checkpoint

| 模型 | 用途 | 路径 | 步数 |
|------|------|------|------|
| **R2A-VLA (MyModal)** | 推理 | `./checkpoints/acot_r2a_mymodal_temporal_noise/r2a_mymodal_v1/4000` | 4000 |
| **R2A-VLA (Base)** | 微调 | (待提供) | - |

### Checkpoint 结构

```
checkpoints/
└── acot_r2a_mymodal_temporal_noise/
    └── r2a_mymodal_v1/
        ├── 1000/          # 1000 步 checkpoint
        ├── 2000/          # 2000 步 checkpoint
        ├── 3000/          # 3000 步 checkpoint
        ├── 4000/          # 4000 步 checkpoint (推荐)
        └── wandb_id.txt   # W&B run ID
```

**注意**: Checkpoint 文件较大（约 12GB），开源时不上传到 GitHub。请通过以下方式获取：
1. 自行训练
2. 联系作者获取下载链接
3. (待添加) ModelScope 下载链接

### 加载 Checkpoint

```python
from openpi.training import config as _config
from openpi.policies import policy_config

config = _config.get_config("acot_r2a_mymodal_temporal_noise")
checkpoint_dir = "./checkpoints/acot_r2a_mymodal_temporal_noise/r2a_mymodal_v1/4000"

policy = policy_config.create_trained_policy(config, checkpoint_dir)
```

---

## 📈 基准测试

### GenieSim-Instruction 基准

| 任务 | 成功率 (%) |
|------|-----------|
| pick_block_color | TBD |
| pick_block_number | TBD |
| pick_block_shape | TBD |
| pick_block_size | TBD |
| pick_common_sense | TBD |
| pick_follow_logic_or | TBD |
| pick_object_type | TBD |
| pick_specific_object | TBD |
| straighten_object | TBD |
| pick_billiards_color | TBD |
| **平均** | **TBD** |

### GenieSim-Robust 基准

| 干扰类型 | 成功率 (%) |
|----------|-----------|
| 指令变化 | TBD |
| 机器人初始位置 | TBD |
| 机器人初始关节 | TBD |
| 末端执行器干扰 | TBD |
| 控制延迟 | TBD |
| 相机帧丢弃 | TBD |
| 相机噪声 | TBD |
| 相机遮挡 | TBD |
| 相机外参 | TBD |
| 环境光照 | TBD |
| 背景变化 | TBD |
| **平均** | **TBD** |

*结果将在完整评估后更新*

---

## 🔧 故障排查

### 常见问题

#### 1. GPU 显存不足

**错误**: `RESOURCE_EXHAUSTED: Out of memory`

**解决方案**:
```bash
# 增加 JAX 显存分配比例
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# 或使用 FSDP 减少显存
uv run scripts/train.py <config> --fsdp-devices=2
```

#### 2. Checkpoint 加载失败

**错误**: `Checkpoint directory does not exist`

**解决方案**:
```bash
# 检查 checkpoint 路径
ls -la ./checkpoints/acot_r2a_mymodal_temporal_noise/r2a_mymodal_v1/

# 确认归一化统计存在
ls -la ./assets/acot_r2a_mymodal_temporal_noise/norm_stats/
```

#### 3. WebSocket 连接失败

**错误**: `Connection refused`

**解决方案**:
```bash
# 检查服务器是否运行
netstat -tlnp | grep 8999

# 查看服务器日志
docker logs <container_id>

# 检查防火墙
sudo ufw allow 8999
```

#### 4. 数据加载错误

**错误**: `Dataset not found` 或 `KeyError`

**解决方案**:
- 确认数据路径正确
- 检查 LeRobot 数据集格式
- 运行 `compute_norm_stats.py` 生成归一化统计

#### 5. 训练损失发散

**错误**: 训练损失突然增大

**解决方案**:
- 检查 `norm_stats.json` 中的统计量
- 降低学习率
- 增加梯度裁剪阈值
- 检查数据质量

---

## 📜 许可证

本项目采用 **MIT License** 许可。详见 [LICENSE](LICENSE) 文件。

---

## 📖 引用

如果您在研究中使用了本模型或代码，请引用：

```bibtex
@article{zhong2026acot,
  title={ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models},
  author={Zhong, Linqing and Liu, Yi and Wei, Yifei and Xiong, Ziyu and Yao, Maoqing and Liu, Si and Ren, Guanghui},
  journal={arXiv preprint arXiv:2601.11404},
  year={2026}
}
```

---

## 🙏 致谢

- 本项目基于 [OpenPI](https://github.com/Physical-Intelligence/openpi) 框架构建
- 感谢 AgiBot 团队提供的 GenieSim 仿真平台
- 感谢 ICRA 2026 AgiBot World Challenge 组织者

---

## 📞 联系方式

- **GitHub Issues**: 技术问题请提 issue
- **邮箱**: (待添加)
- **微信**: (待添加)

---

## 📝 更新日志

- **2026-04-27**: 初始开源版本
  - 发布 ACoT-VLA 模型代码
  - 发布训练和推理脚本
  - 发布 Reasoning2Action 配置

- **待发布**:
  - 添加 PyTorch 支持
  - 添加更多示例
  - 发布预训练权重

---

*最后更新：2026-04-27*
