import torch
from torch import nn
from torch.nn import functional as F
import math


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    def __init__(self, max_len, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()

        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(max_len).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [max_len, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [max_len, d_model // 2, 2]
        emb = emb.view(max_len, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            Swish(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class DownSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1)

    def forward(self, x):
        x = self.main(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.main = nn.Conv2d(in_ch, in_ch, 3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.main(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, in_ch)
        self.proj_q = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_k = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj_v = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)
        self.proj = nn.Conv2d(in_ch, in_ch, 1, stride=1, padding=0)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim, dropout, attn=False):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            Swish(),
            nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1),
        )
        self.temb_proj = nn.Sequential(
            Swish(),
            nn.Linear(tdim, out_ch),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
        )
        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)
        else:
            self.shortcut = nn.Identity()
        if attn:
            self.attn = AttnBlock(out_ch)
        else:
            self.attn = nn.Identity()

    def forward(self, x, temb):
        h = self.block1(x)
        h = h + self.temb_proj(temb)[:, :, None, None]
        h = self.block2(h)

        h = h + self.shortcut(x)
        h = self.attn(h)
        return h


class Classifier(nn.Module):
    def __init__(self, n_classes, ch=128, ch_mult=[1, 2, 2, 2], attn=[1], n_res_blocks=2, dropout=0.1):
        super().__init__()

        assert all([i < len(ch_mult) for i in attn]), "attn index out of bound"

        tdim = ch * 4
        self.time_embed = TimeEmbedding(max_len=1000, d_model=ch, dim=tdim)
        self.label_emb = nn.Embedding(n_classes, tdim)

        self.init_conv = nn.Conv2d(3, ch, kernel_size=3, stride=1, padding=1)
        self.downblocks = nn.ModuleList()
        cur_ch = ch
        for i, mult in enumerate(ch_mult):
            out_ch = ch * mult
            for _ in range(n_res_blocks):
                self.downblocks.append(
                    ResBlock(
                        in_ch=cur_ch,
                        out_ch=out_ch,
                        tdim=tdim,
                        dropout=dropout,
                        attn=(i in attn)
                    )
                )
                cur_ch = out_ch
            if i != len(ch_mult) - 1:
                self.downblocks.append(DownSample(cur_ch))

        self.middleblocks = nn.ModuleList([
            ResBlock(cur_ch, cur_ch, tdim, dropout, attn=True),
            ResBlock(cur_ch, cur_ch, tdim, dropout, attn=False),
        ])

        self.fin_block = nn.Sequential(
            nn.GroupNorm(32, cur_ch),
            Swish(),
            nn.Conv2d(cur_ch, n_classes, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, noisy_image, diffusion_step, label):
        temb = self.time_embed(diffusion_step)
        y = self.label_emb(label)
        temb = temb + y
        x = self.init_conv(noisy_image)
        for layer in self.downblocks:
            if isinstance(layer, DownSample):
                x = layer(x)
            else:
                x = layer(x, temb)

        for layer in self.middleblocks:
            x = layer(x, temb)

        x = self.fin_block(x)
        return x.mean(dim=(2, 3))


if __name__ == "__main__":
    model = Classifier(n_classes=10)
    noisy_image = torch.randn(4, 3, 32, 32)
    diffusion_step = torch.randint(0, 1000, size=(4,))
    label = torch.randint(0, 10, size=(4,))
    out = model(noisy_image, diffusion_step, label)
    out.shape
