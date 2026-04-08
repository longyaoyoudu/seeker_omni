# SeekerOmni

基于 MiniMind2 ChatML 风格自回归语言模型，从零实现的小型视觉-语言模型（VLM）研究框架，支持图像到文本的生成。

## 项目概览

三阶段训练流程，逐步构建图像理解能力：

| 阶段 | 描述 | 关键配置 |
|------|------|---------|
| **S0** | 文本预训练 — 建立语言基座 | 340 seq / 180k 步 |
| **SFT** | 文本微调 — ChatML 格式对齐 + 指令遵循 | 冻结词嵌入 / 6k 步 |
| **E2E** | 图像描述微调 — 注入视觉理解能力 | SigLIP2 + 视觉蒸馏 / 3k 步 |

## 模型架构

**语言模型基座** — MiniMind2 ChatML 风格 Transformer 解码器：
- 8 层，hidden=512，8 头（2 KV 头，GQA 注意力）
- Gated SiFi MLP（intermediate≈1365）、RMSNorm、RoPE（θ=1e6）
- vocab=6400，max_seq=340

**视觉编码器** — SigLIP2-base-patch16-224（768 维视觉特征）

**模态注入** — Perceiver Resampler（2 层，8 头，4×FFN）将 768 维视觉 token 压缩为 49 个模态 token，由可学习的 `img_gate` 标量控制注入强度。默认冻结视觉编码器，避免高昂的全量 LLM 微调。

**推理** — KV Cache 流式 prefill、Flash Attention（SDPA）、repetition penalty、no-repeat-ngram 防循环、temperature/top-k/top-p 采样。

## 目录结构

```
seeker_omni/
├── config.py              # YAML 配置加载器（支持 !include 组合）
├── pipeline.py             # train() / e2e() 入口
├── special_tokens.py       # ChatML + 图像 token 方案
├── model/
│   ├── lm.py              # SeekerOmniLM（forward + generate）
│   ├── block.py            # SeekerBlock（Transformer 层）
│   ├── attention.py        # GQA 自注意力 + RoPE
│   ├── rope.py             # Rotary Embedding
│   ├── mlp.py              # Gated SiFi MLP
│   ├── norm.py             # RMSNorm
│   ├── projector.py        # img_proj / img_gate 注入逻辑
│   └── resampler.py        # PerceiverResampler
├── train/
│   ├── loop.py             # 核心训练循环
│   ├── checkpoint.py        # 断点保存 / 加载与恢复
│   ├── lr.py               # 余弦学习率调度
│   ├── freezing.py          # 视觉编码器冻结工具
│   └── seed.py             # 确定性随机种子
├── steps/
│   ├── train.py            # 多阶段文本训练驱动
│   └── e2e/
│       ├── runner.py       # E2E 视觉微调运行器
│       ├── vision.py        # 视觉编码器 + 蒸馏 + 冻结工具
│       └── distill.py      # MSE 视觉特征蒸馏
├── dataset/
│   └── schema.py           # JSONL 数据格式定义
configs/
├── train.yaml              # 阶段流水线（S0 → SFT）
├── e2e.yaml                # E2E 视觉微调
├── model/base_26m.yaml    # 模型架构配置
└── stages/
    ├── s0.yaml             # S0 预训练配置
    └── sft_text.yaml       # SFT 配置
```

## 安装

```bash
pip install -e .
```

核心依赖：`torch>=2.2`、`transformers>=4.41`、`tokenizers>=0.15`、`pyyaml`、`tensorboard`、`pillow`、`tqdm`。

## 快速开始

### 1. 文本训练（S0 → SFT）

```bash
python -m seeker_omni train
```

读取 `configs/train.yaml`，通过 `auto_init` 自动串联阶段：每个阶段从上一步的 checkpoint 恢复。

### 2. E2E 视觉微调

```bash
python -m seeker_omni e2e
```

读取 `configs/e2e.yaml`。加载 SFT 阶段 checkpoint，冻结 LLM，微调视觉适配器（img_proj + img_gate），可选训练 Perceiver Resampler 和视觉编码器最后 N 层，各组件独立学习率 + MSE 特征蒸馏。

### 3. 断点续训

在阶段配置中指定 `ckpt` 路径，或依赖 `auto_init: true` 自动跨阶段关联。

### 4. 训练可视化

```bash
tensorboard --logdir outputs/tb/
```

## 训练特性

- **梯度检查点** 节省显存
- **混合精度**（FP16 / BF16）配合 `torch.amp.GradScaler`
- **流式 prefill** — 分块 KV Cache 构建，自动保护模态 token 区间不被切分
- **视觉特征蒸馏** — 冻结 Teacher 与可学习 Student SigLIP2 特征间的 MSE 损失
- **组件独立学习率** — bridge（img_proj/img_gate）、resampler、视觉编码器、LLM 各自独立
- **图文混合训练** — E2E 阶段可选混入纯文本 batch
- **确定性训练** — 固定种子打乱、设置 `torch.manual_seed` 与 Python hash seed

## 配置系统

YAML 配置支持 `!include` 组合与自定义 include 指令：

```yaml
# 示例：继承并覆盖模型配置
model:
  base: !include ../model/base_26m.yaml
  hidden_size: 768   # 覆盖
  num_layers: 12     # 覆盖
```

## 许可证

MIT
