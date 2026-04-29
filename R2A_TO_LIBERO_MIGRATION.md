# R2A 模型迁移到 LIBERO 测试指南

## 📋 迁移概述

本指南说明如何将 R2A (Reasoning2Action) checkpoint 迁移到 LIBERO 基准测试中使用。

### 核心差异

| 维度 | R2A (Go2 机器人) | LIBERO (机械臂) | 适配策略 |
|------|-----------------|-----------------|----------|
| **动作空间** | 32 维 | 7 维 | 输出截取前 7 维 |
| **状态空间** | 32 维 (pad 后) | 8 维 | 输入 pad 到 32 维 |
| **相机输入** | 3 路 (top_head, hand_left, hand_right) | 2 路 (agentview, wrist) | 复制 wrist 到 right |
| **时序帧** | 4 帧 | 1 帧 | 单帧重复 4 次 |
| **action_horizon** | 30 | - | 保持 30 |

### 动作空间映射

**R2A 32 维动作结构：**
- `[0:6]`: 末端位姿 (x, y, z, roll, pitch, yaw)
- `[6]`: 夹爪 (左)
- `[7:14]`: 左臂关节位置 (7 DOF)
- `[14]`: 夹爪 (右)
- `[15:22]`: 右臂关节位置 (7 DOF)
- `[22:27]`: 头部/腰部/底盘 (5 DOF)
- `[27:32]`: 保留/填充

**LIBERO 7 维动作：**
- `[0:6]`: 末端位姿 (x, y, z, roll, pitch, yaw)
- `[6]`: 夹爪

**映射策略：** 直接截取 R2A 的 `[0:7]` 维，对应末端位姿 + 夹爪。

---

## 🛠️ 已实现的文件

### 1. 适配层代码
**文件：** `src/openpi/policies/libero_r2a_policy.py`

包含三个变换类：
- `LiberoR2AInputs`: LIBERO 观测 → R2A 模型输入
- `LiberoR2AOutputs`: R2A 模型输出 → LIBERO 动作
- `LiberoR2AWithActionMapping`: 增强版，显式索引映射

### 2. 训练配置
**文件：** `src/openpi/training/config.py`

新增两个配置：
- `acot_r2a_libero_eval`: 基础适配配置
- `acot_r2a_libero_eval_v2`: 增强版配置（推荐）

### 3. 测试脚本
**文件：** `scripts/test_libero_r2a_service.py`

验证：
- 配置加载
- 输入/输出变换
- 策略创建
- 推理执行

---

## 🚀 快速开始

### 步骤 1: 初始化子模块

```bash
cd /mnt/nas/R2A-VLA
git submodule update --init --recursive
```

### 步骤 2: 运行测试脚本

```bash
cd /mnt/nas/R2A-VLA
python scripts/test_libero_r2a_service.py
```

**预期输出：**
```
✓ Config loaded: acot_r2a_libero_eval_v2
✓ Input transform test passed!
✓ Output transform test passed!
✓ Policy created successfully!
✓ Policy inference test passed!
All tests passed! ✓
```

### 步骤 3: 启动策略服务器

```bash
cd /mnt/nas/R2A-VLA

# 激活虚拟环境
source .venv_lerobot/bin/activate

# 启动服务
uv run scripts/serve_policy.py \
    --env LIBERO \
    policy:checkpoint \
    --policy.config acot_r2a_libero_eval_v2 \
    --policy.dir ./checkpoints/acot_r2a_mymodal_temporal_noise/r2a_mymodal_v1/4000
```

**服务器日志示例：**
```
INFO: Loading model...
INFO: Creating server (host: <hostname>, ip: <ip>)
INFO: Serving policy on port 8000
```

### 步骤 4: 启动 LIBERO 客户端

**Terminal 2:**
```bash
cd /mnt/nas/R2A-VLA

# 激活虚拟环境
source .venv_lerobot/bin/activate

# 安装 LIBERO 依赖（如果未安装）
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# 运行仿真
python examples/libero/main.py \
    --task-suite-name libero_spatial \
    --num-trials-per-task 50 \
    --video-out-path ./libero_videos
```

---

## 📊 验证步骤

### 1. 最小验证（推荐先执行）

使用 simple_client 验证服务能响应：

```bash
# Terminal 1: 启动服务（同上）

# Terminal 2: 简单客户端测试
python examples/simple_client/main.py
```

### 2. 单任务验证

```bash
python examples/libero/main.py \
    --task-suite-name libero_spatial \
    --num-trials-per-task 5
```

### 3. 完整测试

```bash
python examples/libero/main.py \
    --task-suite-name libero_spatial \
    --num-trials-per-task 50
```

---

## 🔧 故障排除

### 问题 1: Checkpoint 加载失败

**错误：** `Checkpoint directory not found`

**解决：**
```bash
# 确认路径正确
ls -la /mnt/nas/R2A-VLA/checkpoints/acot_r2a_mymodal_temporal_noise/r2a_mymodal_v1/4000/
```

### 问题 2: 动作维度不匹配

**错误：** `Expected 7 action dims, got X`

**解决：** 检查 `LiberoR2AOutputs` 是否正确截取 7 维。

### 问题 3: 图像格式错误

**错误：** `Expected 4D (T,H,W,C), got X`

**解决：** 确认 `LiberoR2AInputs._create_temporal_frames` 正确创建 4 帧。

### 问题 4: 内存不足

**错误：** `OOM` 或 `CUDA out of memory`

**解决：** 减少 batch_size 或使用更小的模型配置。

---

## 📝 配置说明

### acot_r2a_libero_eval_v2 配置参数

```python
model=acot_vla_mymodal.ACOTMyModalConfig(
    coarse_action_horizon=30,    # 粗粒度动作范围
    action_horizon=30,           # 精细动作范围
    paligemma_variant="gemma_2b_lora",  # 语言模型变体
    adopt_explicit_action_reasoner=True,   # 显式动作推理
    adopt_implicit_action_reasoner=True,   # 隐式动作推理
    downsample_based_implicit_extractor=True,  # 下采样隐式提取
    num_history_frames=4,        # 时序帧数
    temporal_encoder_layers=6,   # 时序编码器层数
    adopt_noise_expert=True,     # 噪声专家
)
```

### 输入变换流程

```
LIBERO 观测
  ↓
observation/image (224, 224, 3)  ──┐
observation/wrist_image (224, 224, 3) ──┤
observation/state (8,)  ────────────────┤
prompt (str)  ─────────────────────────┤
  ↓                                    │
LiberoR2AInputs                        │
  ↓                                    │
image.base_0_rgb (4, 224, 224, 3) ←────┘
image.left_wrist_0_rgb (4, 224, 224, 3) ←─┘
image.right_wrist_0_rgb (4, 224, 224, 3) ← 复制
state (32,) ← pad
prompt (str)
  ↓
R2A 模型
  ↓
actions (30, 32)
coarse_actions (50, 32)
  ↓
LiberoR2AOutputs
  ↓
actions (30, 7) ← 截取前 7 维
coarse_actions (50, 7)
```

---

## 🎯 成功标准

迁移成功不等于成功率高，而是先满足：

1. ✅ 服务能加载 checkpoint 并完成推理
2. ✅ LIBERO 客户端能拿到合法 7 维动作
3. ✅ rollout 不因 shape/语义错误崩溃
4. ✅ 动作数值稳定，不持续 NaN 或发散

---

## 📈 后续优化

### Phase 1: 链路验证（当前阶段）
- ✅ 适配层实现
- ✅ 配置创建
- ✅ 测试脚本
- ⏳ 服务启动验证
- ⏳ LIBERO rollout 验证

### Phase 2: 性能优化
- [ ] 优化动作映射策略（基于物理语义）
- [ ] 评估时序模型降级影响
- [ ] 调整归一化统计
- [ ] 完整 benchmark 测试

### Phase 3: 分析改进
- [ ] 分析失败案例
- [ ] 微调动作映射
- [ ] 考虑微调训练

---

## 📚 参考文件

- [R2A Temporal Policy](src/openpi/policies/r2a_temporal_policy.py)
- [LIBERO Policy](src/openpi/policies/libero_policy.py)
- [ACOT-VLA MyModal Config](src/openpi/training/config.py)
- [Serve Policy](scripts/serve_policy.py)
- [LIBERO Example](examples/libero/main.py)

---

## 📞 联系

如有问题，请查看测试脚本输出日志或联系开发团队。
