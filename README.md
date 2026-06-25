# VAE on MNIST — 隐空间学习过程可视化

在 MNIST 上训练一个**卷积变分自编码器(2 维隐空间)**,并把训练过程录成视频:
每隔若干训练步截一帧,左边是测试样本在 2 维隐空间的投影(按数字着色),
右边是在 2 维 `z` 网格上解码出的图像流形——对应教材图 16.6(a)(b)。
训练收敛后自动合成一段 mp4(默认 10fps,即每帧 0.1s)。

![示例帧](docs/sample_frame.png)

## 环境(uv)

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # RTX 5090 (Blackwell) 需 cu128
uv pip install numpy matplotlib imageio imageio-ffmpeg scikit-learn
```

## 运行

**推荐配置(扫描实验得到的最优,隐空间最干净):**

```bash
python train.py --ch 64 --epochs 60 --batch 1024 --snap-every 18 --lr 1e-3 --out runs/conv64
```

吃满显存的大模型版(显存利用率优先,但隐空间更糊):

```bash
python train.py --ch 256 --epochs 250 --batch 10240 --snap-every 8 --lr 3e-3 --out runs/main
```

产物在 `runs/<name>/`:`vae_latent_learning.mp4`(学习过程视频)、`frames/`(每帧 PNG)、`vae_final.pt`(权重)。

仓库根目录附两段成片:
- `vae_latent_learning_conv64_best.mp4` —— **推荐**:Conv-64(0.885M),隐空间分簇清晰;
- `vae_latent_learning_conv256_big.mp4` —— 大模型 Conv-256(13.76M),显存拉满但隐空间糊。

## 参数量扫描结论(`sweep.py`)

按 **生成质量(测试 ELBO↓)+ 编码质量(隐空间 kNN 准确率↑)** 综合评估 10 档模型:

| 排名 | 模型 | 参数量 | ELBO↓ | kNN↑ | 综合分 |
|---|---|---|---|---|---|
| 🥇 1 | **Conv-64** | **0.885M** | 144.7 | **0.779** | **0.968** |
| 2 | MLP-1024 | 3.71M | 143.9 | 0.762 | 0.940 |
| 3 | Conv-32 | 0.23M | 145.4 | 0.764 | 0.885 |
| 8 | Conv-256 | 13.76M | 150.1 | 0.733 | 0.593 |
| 10 | MLP-64 | 0.11M | 156.6 | 0.637 | 0.000 |

**结论:最优参数量 ≈ 0.9M(Conv-64),甜区 0.2–1M;最大的 Conv-256(13.76M)反而排第 8** ——
2 维隐空间下容量过剩不仅无益、还轻微伤泛化。完整数据见 `docs/sweep_results.json`,
10 组隐空间对比见 `docs/latent_compare.png`。复现:`python sweep.py`。

## 主要参数

| 参数 | 含义 | 默认 |
|---|---|---|
| `--batch` | batch 大小(越大越吃显存/CUDA) | 4096 |
| `--ch` | 卷积基础通道数(越大越吃显存) | 128 |
| `--zdim` | 隐空间维度(=2 便于可视化) | 2 |
| `--snap-every` | 每多少 step 截一帧 | 10 |
| `--fps` | 视频帧率(10 = 每帧 0.1s) | 10 |
| `--epochs` / `--lr` / `--beta` | 训练轮数 / 学习率 / KL 权重 | 100 / 2e-3 / 1.0 |

## 5090 (32GB) 显存调参

显存与 CUDA 利用率主要由 `--batch` 和 `--ch` 决定。实测:
- `--batch 8192  --ch 256` → 约 18.7 GB
- `--batch 10240 --ch 256` → 约 31 GB(占满 ~95%,GPU 利用率 100%)

按需在两者之间取舍即可。

## 模型与目标

- 编码器 `q(z|x)=N(μ_φ(x), σ_φ²(x))`,解码器 `p(x|z)` 为伯努利(BCE);
- 目标 = 重构(BCE)+ `β`·KL,其中 `KL(N(μ,σ²)‖N(0,I))` 用闭式;
- 用重参数化 `z=μ+σ⊙ε` 使梯度可回传到编码器。
