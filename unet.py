# References:
    # https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/guided_diffusion/unet.py

import torch
from torch import nn
from torch.nn import functional as F
import math

from classifier import Swish, TimeEmbedding, ResBlock, Downsample


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(
        self,
        n_classes,
        channels=128,
        channel_mults=[1, 2, 2, 2],
        attns=[False, True, False, False],
        n_res_blocks=2,
    ):
        super().__init__()

        assert all([i < len(channel_mults) for i in attns]), "attns index out of bound"

        time_channels = channels * 4
        self.time_embed = nn.Sequential(
            TimeEmbedding(max_len=1000, time_channels=time_channels),
            nn.Linear(channels, time_channels),
            Swish(),
            nn.Linear(time_channels, time_channels),
        )
        self.label_emb = nn.Embedding(n_classes, time_channels)

        self.init_conv = nn.Conv2d(3, channels, 3, 1, 1)
        self.down_blocks = nn.ModuleList()
        cxs = [channels]
        cur_channels = channels
        for i, mult in enumerate(channel_mults):
            out_channels = channels * mult
            for _ in range(n_res_blocks):
                self.down_blocks.append(
                    ResBlock(
                        in_channels=cur_channels,
                        out_channels=out_channels,
                        time_channels=time_channels,
                        attn=attns[i]
                    )
                )
                cur_channels = out_channels
                cxs.append(cur_channels)
            if i != len(channel_mults) - 1:
                self.down_blocks.append(Downsample(cur_channels))
                cxs.append(cur_channels)

        self.mid_blocks = nn.ModuleList([
            ResBlock(
                in_channels=cur_channels,
                out_channels=cur_channels,
                time_channels=time_channels,
                attn=True,
            ),
            ResBlock(
                in_channels=cur_channels,
                out_channels=cur_channels,
                time_channels=time_channels,
                attn=False,
            ),
        ])

        self.up_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(channel_mults))):
            out_channels = channels * mult
            for _ in range(n_res_blocks + 1):
                self.up_blocks.append(
                    ResBlock(
                        in_channels=cxs.pop() + cur_channels,
                        out_channels=out_channels,
                        time_channels=time_channels,
                        attn=attns[i],
                    )
                )
                cur_channels = out_channels
            if i != 0:
                self.up_blocks.append(Upsample(cur_channels))
        assert len(cxs) == 0

        self.fin_block = nn.Sequential(
            nn.GroupNorm(32, cur_channels),
            Swish(),
            nn.Conv2d(cur_channels, 3, 3, 1, 1)
        )

    def forward(self, noisy_image, diffusion_step, label):
        x = self.init_conv(noisy_image)
        t = self.time_embed(diffusion_step)
        y = self.label_emb(label)

        xs = [x]
        for layer in self.down_blocks:
            if isinstance(layer, Downsample):
                x = layer(x)
            else:
                x = layer(x, t + y)
            xs.append(x)

        for layer in self.mid_blocks:
            x = layer(x, t + y)

        for layer in self.up_blocks:
            if isinstance(layer, Upsample):
                x = layer(x)
            else:
                x = torch.cat([x, xs.pop()], dim=1)
                x = layer(x, t + y)
        assert len(xs) == 0
        return self.fin_block(x)


if __name__ == "__main__":
    model = UNet(n_classes=10)

    noisy_image = torch.randn(4, 3, 32, 32)
    diffusion_step = torch.randint(0, 1000, size=(4,))
    label = torch.randint(0, 10, size=(4,))
    out = model(noisy_image, diffusion_step, label)
    out.shape
