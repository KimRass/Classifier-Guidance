# References:
    # https://huggingface.co/blog/annotated-diffusion

import torch
import matplotlib.pyplot as plt
from PIL import Image
from moviepy.video.io.bindings import mplfig_to_npimage
from pathlib import Path


def to_pil(img):
    if not isinstance(img, Image.Image):
        image = Image.fromarray(img)
        return image
    else:
        return img


def plt_to_pil(fig):
    img = mplfig_to_npimage(fig)
    image = to_pil(img)
    return image


def save_image(image, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    to_pil(image).save(str(path), quality=100)


def get_linear_beta_schdule(init_beta, fin_beta, n_timesteps):
    return torch.linspace(init_beta, fin_beta, n_timesteps + 1)


def get_cosine_beta_schedule(n_timesteps, s=0.008):
    # "we selected s such that p 0 was slightly smaller than the pixel bin size 1=127:5, which gives s = 0:008."
    timestep = torch.linspace(0, n_timesteps, n_timesteps + 1)
    # "We chose to use $\cos^{2}$ in particular."
    alpha_bar = torch.cos(((timestep / n_timesteps) + s) / (1 + s) * torch.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    beta = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    # return torch.clip(beta, 0.0001, 0.9999)
    return beta


if __name__ == "__main__":
    n_timesteps = 1000
    init_beta = 0.0001
    fin_beta = 0.02
    linear_beta = get_linear_beta_schdule(
        init_beta=init_beta, fin_beta=fin_beta, n_timesteps=n_timesteps,
    )
    cos_beta = get_cosine_beta_schedule(n_timesteps=1000)

    linear_alpha = 1 - linear_beta
    linear_alpha_bar = torch.cumprod(linear_alpha, dim=0)

    cos_alpha = 1 - cos_beta
    cos_alpha_bar = torch.cumprod(cos_alpha, dim=0)
    # linear_alpha_bar[0]
    # cos_alpha_bar[0]

    fig, axes = plt.subplots(1, 1, figsize=(5, 3))
    line2 = axes.plot(linear_alpha_bar.numpy(), label="Linear")
    line2 = axes.plot(cos_alpha_bar.numpy(), label="Cosine")
    # line2 = axes.plot((linear_alpha_bar.numpy() ** 0.5))
    # line2 = axes.plot((cos_alpha_bar.numpy() ** 0.5))
    axes.legend(fontsize=6)
    axes.tick_params(labelsize=7)
    fig.tight_layout()
    image = plt_to_pil(fig)
    # image.show()
    save_image(image, path="/Users/jongbeomkim/Desktop/workspace/Dhariwal-and-Nichol/beta_schedules.jpg")
