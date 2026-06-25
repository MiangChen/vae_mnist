"""
MNIST 上的变分自编码器(2 维隐空间),并录制"隐空间流形"学习过程。

每训练若干步(--snap-every)截一帧:
  - 左:测试集样本在 2 维隐空间上的投影(按数字着色)        —— 对应图 16.6(a)
  - 右:在 2 维 z 网格上解码得到的图像流形                    —— 对应图 16.6(b)
训练收敛后,把所有帧合成一段 mp4(默认 0.1s/帧 = 10fps)。

超参数按 RTX 5090 (32GB) 设计:大 batch + 宽通道 + bf16-AMP + TF32,
拉高显存与 CUDA 利用率。可用 --batch / --ch 进一步加大。
"""
import argparse, os, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Noto Serif CJK SC"   # 中文标题字体
matplotlib.rcParams["axes.unicode_minus"] = False
import imageio.v2 as imageio


# ----------------------------- 模型 -----------------------------
class ConvVAE(nn.Module):
    """卷积 VAE,隐空间维度默认 2(便于可视化)。"""
    def __init__(self, ch=128, zdim=2):
        super().__init__()
        self.zdim = zdim
        C = ch
        # 编码器:28 -> 14 -> 7 -> 4
        self.enc = nn.Sequential(
            nn.Conv2d(1, C, 4, 2, 1),    nn.GroupNorm(8, C),    nn.SiLU(),   # 14
            nn.Conv2d(C, 2*C, 4, 2, 1),  nn.GroupNorm(8, 2*C),  nn.SiLU(),   # 7
            nn.Conv2d(2*C, 4*C, 3, 2, 1),nn.GroupNorm(8, 4*C),  nn.SiLU(),   # 4
        )
        self.enc_flat = 4*C*4*4
        self.fc_mu     = nn.Linear(self.enc_flat, zdim)
        self.fc_logvar = nn.Linear(self.enc_flat, zdim)
        # 解码器:4 -> 7 -> 14 -> 28
        self.fc_dec = nn.Linear(zdim, 4*C*4*4)
        self.C = C
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(4*C, 2*C, 3, 2, 1, 0), nn.GroupNorm(8, 2*C), nn.SiLU(),  # 7
            nn.ConvTranspose2d(2*C, C, 4, 2, 1, 0),   nn.GroupNorm(8, C),   nn.SiLU(),  # 14
            nn.ConvTranspose2d(C, 1, 4, 2, 1, 0),                                       # 28
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)              # 重参数化:把随机性放到与 phi 无关的 eps 上
        return mu + std * eps

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 4*self.C, 4, 4)
        return self.dec(h)                        # 返回 logits(配 BCE-with-logits)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparam(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(logits, x, mu, logvar, beta):
    # 重构:伯努利似然 = BCE(对每图求和,对 batch 求均值)
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=[1,2,3]).mean()
    # KL(q(z|x)=N(mu,sigma^2) || N(0,I)) 的闭式
    kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()
    return recon + beta * kl, recon.detach(), kl.detach()


# ----------------------------- 可视化 -----------------------------
@torch.no_grad()
def render_frame(model, vis_x, vis_y, step, epoch, recon, kl, path,
                 grid_n=20, z_lim=3.0, scatter_lim=4.0, device="cuda", amp_dt=torch.bfloat16):
    model.eval()
    # 左:隐空间散点(用固定的一批测试样本,编码取均值)
    with torch.autocast("cuda", dtype=amp_dt):
        mu, _ = model.encode(vis_x)
    mu = mu.float().cpu().numpy()
    y = vis_y.cpu().numpy()

    # 右:2 维 z 网格 -> 解码 -> 拼成大图
    gx = np.linspace(-z_lim, z_lim, grid_n)
    gy = np.linspace(z_lim, -z_lim, grid_n)          # y 轴自上而下
    zz = torch.tensor(np.array([[a, b] for b in gy for a in gx]),
                      dtype=torch.float32, device=device)
    with torch.autocast("cuda", dtype=amp_dt):
        imgs = torch.sigmoid(model.decode(zz)).float().cpu().numpy()  # (grid_n^2,1,28,28)
    canvas = np.zeros((grid_n*28, grid_n*28), dtype=np.float32)
    for i in range(grid_n):
        for j in range(grid_n):
            canvas[i*28:(i+1)*28, j*28:(j+1)*28] = imgs[i*grid_n+j, 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.6))
    sc = ax1.scatter(mu[:, 0], mu[:, 1], c=y, cmap="jet", s=4, vmin=0, vmax=9, alpha=0.7)
    ax1.set_xlim(-scatter_lim, scatter_lim); ax1.set_ylim(-scatter_lim, scatter_lim)
    ax1.set_aspect("equal"); ax1.grid(True, alpha=0.3)
    ax1.set_title("(a) 训练样本在隐空间的投影")
    cb = fig.colorbar(sc, ax=ax1, ticks=range(10)); cb.set_label("digit")

    ax2.imshow(canvas, cmap="gray"); ax2.axis("off")
    ax2.set_title("(b) 隐变量 z 在图像空间的解码")

    fig.suptitle(f"epoch {epoch:3d} | step {step:5d} | recon {recon:7.2f} | KL {kl:6.3f}",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    model.train()


# ----------------------------- 主程序 -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",   default=os.path.expanduser("~/jepa/data"), help="MNIST 根目录(含 MNIST/raw)")
    ap.add_argument("--out",    default="runs/run1")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch",  type=int, default=4096)
    ap.add_argument("--ch",     type=int, default=128, help="基础通道数(越大越吃显存)")
    ap.add_argument("--lr",     type=float, default=2e-3)
    ap.add_argument("--beta",   type=float, default=1.0)
    ap.add_argument("--zdim",   type=int, default=2)
    ap.add_argument("--snap-every", type=int, default=10, help="每多少 step 截一帧")
    ap.add_argument("--n-vis",  type=int, default=5000, help="散点图用多少测试样本")
    ap.add_argument("--fps",    type=int, default=10, help="视频帧率(10=每帧0.1s)")
    ap.add_argument("--seed",   type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = "cuda"
    amp_dt = torch.bfloat16
    os.makedirs(args.out, exist_ok=True)
    frame_dir = os.path.join(args.out, "frames"); os.makedirs(frame_dir, exist_ok=True)

    # 数据:复用缓存,不重新下载;一次性搬到 GPU 加速
    tf = transforms.ToTensor()
    train_ds = datasets.MNIST(args.data, train=True,  download=False, transform=tf)
    test_ds  = datasets.MNIST(args.data, train=False, download=False, transform=tf)
    Xtr = torch.stack([train_ds[i][0] for i in range(len(train_ds))]).to(device)       # (60000,1,28,28)
    Xte = torch.stack([test_ds[i][0]  for i in range(len(test_ds))]).to(device)
    Yte = torch.tensor([test_ds[i][1] for i in range(len(test_ds))])
    vis_idx = torch.randperm(len(Xte))[:args.n_vis]
    vis_x, vis_y = Xte[vis_idx], Yte[vis_idx]
    N = Xtr.shape[0]

    model = ConvVAE(ch=args.ch, zdim=args.zdim).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"[init] params={nparam/1e6:.2f}M  batch={args.batch}  ch={args.ch}  steps/epoch={math.ceil(N/args.batch)}")

    step = 0
    frame_paths = []
    # 训练前先截一帧(随机初始化的样子)
    f0 = os.path.join(frame_dir, f"frame_{0:06d}.png")
    render_frame(model, vis_x, vis_y, 0, 0, 0.0, 0.0, f0, device=device, amp_dt=amp_dt)
    frame_paths.append(f0)

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(N, device=device)
        model.train()
        for b in range(0, N, args.batch):
            xb = Xtr[perm[b:b+args.batch]].to(memory_format=torch.channels_last)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dt):
                logits, mu, logvar = model(xb)
                loss, recon, kl = vae_loss(logits, xb, mu, logvar, args.beta)
            loss.backward()
            opt.step()
            step += 1
            if step % args.snap_every == 0:
                fp = os.path.join(frame_dir, f"frame_{step:06d}.png")
                render_frame(model, vis_x, vis_y, step, epoch, recon.item(), kl.item(),
                             fp, device=device, amp_dt=amp_dt)
                frame_paths.append(fp)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            mem = torch.cuda.max_memory_allocated()/1e9
            print(f"[ep {epoch:3d}] step {step:5d} recon {recon.item():7.2f} KL {kl.item():6.3f} "
                  f"| {time.time()-t0:5.1f}s | peakVRAM {mem:.1f}GB | frames {len(frame_paths)}")

    # 合成视频(转 RGB 去掉 alpha;macro_block_size=16 自动把尺寸补成偶数,避免 yuv420p 报错)
    video = os.path.join(args.out, "vae_latent_learning.mp4")
    print(f"[video] writing {len(frame_paths)} frames -> {video} @ {args.fps}fps")
    with imageio.get_writer(video, fps=args.fps, codec="libx264", quality=8,
                            macro_block_size=16, pixelformat="yuv420p") as w:
        for fp in frame_paths:
            im = imageio.imread(fp)
            if im.ndim == 3 and im.shape[2] == 4:
                im = im[..., :3]                 # 去掉 alpha 通道
            w.append_data(im)
    torch.save(model.state_dict(), os.path.join(args.out, "vae_final.pt"))
    print(f"[done] {time.time()-t0:.1f}s  video={video}  ckpt={args.out}/vae_final.pt")


if __name__ == "__main__":
    main()
