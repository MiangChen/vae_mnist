"""
参数量扫描:用不同大小的 VAE(MLP / Conv,各 5 档)在 MNIST 上训到收敛,
然后综合评估"生成质量 + 编码质量",排名找出最优参数量。

评价指标:
  生成质量  = 测试集 ELBO loss = recon(BCE) + KL   (越低 = 似然越高 = 生成越好)
  编码质量  = 隐空间 kNN 分类准确率                  (越高 = 同类聚拢、异类分开 = 编码越好)
            + 轮廓系数 silhouette(辅助)
  综合得分  = 0.5 * 归一化(生成) + 0.5 * 归一化(编码)   (两者都归一化到 [0,1],1=最好)
"""
import os, json, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"] = "Noto Serif CJK SC"; matplotlib.rcParams["axes.unicode_minus"] = False


# ----------------------------- 两种结构 -----------------------------
class MLPVAE(nn.Module):
    def __init__(self, hidden=256, zdim=2):
        super().__init__()
        H = hidden
        self.enc = nn.Sequential(nn.Linear(784, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU())
        self.mu, self.lv = nn.Linear(H, zdim), nn.Linear(H, zdim)
        self.dec = nn.Sequential(nn.Linear(zdim, H), nn.SiLU(), nn.Linear(H, H), nn.SiLU(), nn.Linear(H, 784))
    def encode(self, x): h = self.enc(x.flatten(1)); return self.mu(h), self.lv(h)
    def decode(self, z): return self.dec(z).view(-1, 1, 28, 28)
    def forward(self, x):
        mu, lv = self.encode(x); z = mu + torch.exp(0.5*lv)*torch.randn_like(lv); return self.decode(z), mu, lv

class ConvVAE(nn.Module):
    def __init__(self, ch=64, zdim=2):
        super().__init__(); C = ch; self.C = C
        self.enc = nn.Sequential(
            nn.Conv2d(1, C, 4, 2, 1), nn.GroupNorm(8, C), nn.SiLU(),
            nn.Conv2d(C, 2*C, 4, 2, 1), nn.GroupNorm(8, 2*C), nn.SiLU(),
            nn.Conv2d(2*C, 4*C, 3, 2, 1), nn.GroupNorm(8, 4*C), nn.SiLU())
        self.mu, self.lv = nn.Linear(4*C*16, zdim), nn.Linear(4*C*16, zdim)
        self.fc = nn.Linear(zdim, 4*C*16)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(4*C, 2*C, 3, 2, 1, 0), nn.GroupNorm(8, 2*C), nn.SiLU(),
            nn.ConvTranspose2d(2*C, C, 4, 2, 1, 0), nn.GroupNorm(8, C), nn.SiLU(),
            nn.ConvTranspose2d(C, 1, 4, 2, 1, 0))
    def encode(self, x): h = self.enc(x).flatten(1); return self.mu(h), self.lv(h)
    def decode(self, z): return self.dec(self.fc(z).view(-1, 4*self.C, 4, 4))
    def forward(self, x):
        mu, lv = self.encode(x); z = mu + torch.exp(0.5*lv)*torch.randn_like(lv); return self.decode(z), mu, lv


def build(cfg):
    a, w = cfg["arch"], cfg["w"]
    return MLPVAE(hidden=w) if a == "mlp" else ConvVAE(ch=w)


# ----------------------------- 训练 + 评估 -----------------------------
def loss_fn(logits, x, mu, lv, beta=1.0):
    recon = F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum([1,2,3]).mean()
    kl = (-0.5*(1+lv-mu.pow(2)-lv.exp()).sum(1)).mean()
    return recon + beta*kl, recon.detach(), kl.detach()

@torch.no_grad()
def evaluate(model, Xtr, Ytr, Xte, Yte, device):
    model.eval()
    # 编码全部 train/test 取均值 mu
    def enc_all(X):
        out = []
        for b in range(0, len(X), 8192):
            mu, _ = model.encode(X[b:b+8192]); out.append(mu.float().cpu())
        return torch.cat(out).numpy()
    mu_tr, mu_te = enc_all(Xtr), enc_all(Xte)
    # 生成质量:测试集 ELBO loss
    rec_sum, kl_sum, n = 0.0, 0.0, 0
    for b in range(0, len(Xte), 8192):
        xb = Xte[b:b+8192]; logits, mu, lv = model(xb)
        _, rec, kl = loss_fn(logits, xb, mu, lv)
        bs = len(xb); rec_sum += rec.item()*bs; kl_sum += kl.item()*bs; n += bs
    recon, kl = rec_sum/n, kl_sum/n; elbo = recon + kl
    # 编码质量:kNN 准确率(train 拟合,test 评估)+ 轮廓系数
    knn = KNeighborsClassifier(n_neighbors=10).fit(mu_tr, Ytr.numpy())
    acc = knn.score(mu_te, Yte.numpy())
    idx = np.random.RandomState(0).permutation(len(mu_te))[:3000]
    sil = silhouette_score(mu_te[idx], Yte.numpy()[idx])
    model.train()
    return dict(recon=recon, kl=kl, elbo=elbo, knn_acc=acc, silhouette=sil), mu_te, Yte.numpy()


def train_one(cfg, data, args, device):
    Xtr, Ytr, Xte, Yte = data
    torch.manual_seed(0); np.random.seed(0)
    model = build(cfg).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    N = len(Xtr); t0 = time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(N, device=device)
        for b in range(0, N, args.batch):
            xb = Xtr[perm[b:b+args.batch]]
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits, mu, lv = model(xb); loss, rec, kl = loss_fn(logits, xb, mu, lv)
            loss.backward(); opt.step()
    metrics, mu_te, y_te = evaluate(model, Xtr, Ytr, Xte, Yte, device)
    metrics.update(arch=cfg["arch"], w=cfg["w"], label=cfg["label"], params=nparam,
                   params_M=round(nparam/1e6, 3), sec=round(time.time()-t0, 1))
    return metrics, mu_te, y_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/jepa/data"))
    ap.add_argument("--out", default="runs/sweep")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True

    tf = transforms.ToTensor()
    tr = datasets.MNIST(args.data, train=True, download=False, transform=tf)
    te = datasets.MNIST(args.data, train=False, download=False, transform=tf)
    Xtr = torch.stack([tr[i][0] for i in range(len(tr))]).to(device)
    Ytr = torch.tensor([tr[i][1] for i in range(len(tr))])
    Xte = torch.stack([te[i][0] for i in range(len(te))]).to(device)
    Yte = torch.tensor([te[i][1] for i in range(len(te))])
    data = (Xtr, Ytr, Xte, Yte)

    # 10 组配置:MLP 与 Conv 各 5 档,参数量从小到大
    configs = [
        dict(arch="mlp", w=64,  label="MLP-64"),
        dict(arch="mlp", w=128, label="MLP-128"),
        dict(arch="mlp", w=256, label="MLP-256"),
        dict(arch="mlp", w=512, label="MLP-512"),
        dict(arch="mlp", w=1024,label="MLP-1024"),
        dict(arch="conv",w=16,  label="Conv-16"),
        dict(arch="conv",w=32,  label="Conv-32"),
        dict(arch="conv",w=64,  label="Conv-64"),
        dict(arch="conv",w=128, label="Conv-128"),
        dict(arch="conv",w=256, label="Conv-256"),
    ]

    results, scatters = [], []
    for i, cfg in enumerate(configs):
        m, mu_te, y_te = train_one(cfg, data, args, device)
        results.append(m); scatters.append((cfg["label"], mu_te, y_te))
        print(f"[{i+1:2d}/10] {m['label']:9s} params={m['params_M']:6.3f}M  "
              f"ELBO={m['elbo']:7.2f} recon={m['recon']:7.2f} KL={m['kl']:6.2f}  "
              f"kNN={m['knn_acc']:.4f} sil={m['silhouette']:+.3f}  {m['sec']}s", flush=True)

    # 综合得分:两个指标各自 min-max 归一化到 [0,1](1=最好)
    elbos = np.array([r["elbo"] for r in results])
    accs  = np.array([r["knn_acc"] for r in results])
    gen_norm = (elbos.max() - elbos) / (elbos.max() - elbos.min() + 1e-9)   # ELBO 越低越好
    enc_norm = (accs - accs.min()) / (accs.max() - accs.min() + 1e-9)       # acc 越高越好
    for r, g, e in zip(results, gen_norm, enc_norm):
        r["gen_norm"], r["enc_norm"], r["combined"] = float(g), float(e), float(0.5*g + 0.5*e)
    results.sort(key=lambda r: r["combined"], reverse=True)
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=2, ensure_ascii=False)

    # 排名表
    print("\n" + "="*100)
    print(f"{'排名':<4}{'模型':<10}{'参数量':<10}{'ELBO↓':<9}{'recon↓':<9}{'kNN↑':<8}{'sil↑':<8}{'生成分':<8}{'编码分':<8}{'综合分':<7}")
    print("-"*100)
    for k, r in enumerate(results):
        print(f"{k+1:<5}{r['label']:<10}{r['params_M']:<10.3f}{r['elbo']:<9.2f}{r['recon']:<9.2f}"
              f"{r['knn_acc']:<8.4f}{r['silhouette']:<+8.3f}{r['gen_norm']:<8.3f}{r['enc_norm']:<8.3f}{r['combined']:<7.3f}")
    print("="*100)
    best = results[0]
    print(f"\n>>> 综合最优:{best['label']}  (参数量 {best['params_M']}M, ELBO {best['elbo']:.1f}, kNN {best['knn_acc']:.3f})")

    # 10 张隐空间散点对比图
    order = {s[0]: s for s in scatters}
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ax, cfg in zip(axes.flat, configs):
        lbl, mu, y = order[cfg["label"]]
        ax.scatter(mu[:,0], mu[:,1], c=y, cmap="jet", s=2, vmin=0, vmax=9, alpha=0.5)
        rr = next(r for r in results if r["label"]==lbl)
        ax.set_title(f"{lbl} | {rr['params_M']}M | kNN={rr['knn_acc']:.3f}", fontsize=10)
        ax.set_xlim(-4,4); ax.set_ylim(-4,4); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("不同参数量 VAE 的隐空间(测试集投影,按数字着色)", fontsize=14)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "latent_compare.png"), dpi=110); plt.close(fig)
    print(f"\n散点对比图 -> {args.out}/latent_compare.png   结果 -> {args.out}/results.json")


if __name__ == "__main__":
    main()
