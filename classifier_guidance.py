# References:
    # https://github.com/openai/guided-diffusion/blob/22e0df8183507e13a7813f8d38d51b072ca1e67c/scripts/classifier_sample.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio
from tqdm import tqdm
import contextlib


class ClassifierGuidedDiffusion(nn.Module):
    def get_linear_beta_schdule(self):
        self.beta = torch.linspace(
            self.init_beta,
            self.fin_beta,
            self.n_diffusion_steps,
            device=self.device,
        )

    def __init__(
        self,
        unet,
        classifier,
        img_size,
        device,
        classifier_scale=10,
        image_channels=3,
        n_diffusion_steps=1000,
        init_beta=0.0001,
        fin_beta=0.02,
    ):
        super().__init__()

        self.unet = unet.to(device)
        self.classifier = classifier.to(device)

        self.img_size = img_size
        self.device = device
        
        self.image_channels = image_channels
        self.n_diffusion_steps = n_diffusion_steps
        self.init_beta = init_beta
        self.fin_beta = fin_beta
        self.classifier_scale = classifier_scale

        self.get_linear_beta_schdule()
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    @staticmethod
    def index(x, diffusion_step):
        return x[diffusion_step][:, None, None, None]

    def sample_noise(self, batch_size):
        return torch.randn(
            size=(batch_size, self.image_channels, self.img_size, self.img_size),
            device=self.device,
        )

    def sample_diffusion_step(self, batch_size):
        return torch.randint(
            0, self.n_diffusion_steps, size=(batch_size,), device=self.device,
        )

    def batchify_diffusion_steps(self, diffusion_step_idx, batch_size):
        return torch.full(
            size=(batch_size,),
            fill_value=diffusion_step_idx,
            dtype=torch.long,
            device=self.device,
        )

    def perform_diffusion_process(self, ori_image, diffusion_step, rand_noise=None):
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        mean = (alpha_bar_t ** 0.5) * ori_image
        var = 1 - alpha_bar_t
        if rand_noise is None:
            rand_noise = self.sample_noise(batch_size=ori_image.size(0))
        noisy_image = mean + (var ** 0.5) * rand_noise
        return noisy_image

    def forward(self, noisy_image, diffusion_step, label):
        return self.unet(noisy_image=noisy_image, diffusion_step=diffusion_step, label=label)

    def get_unet_loss(self, ori_image, label):
        rand_diffusion_step = self.sample_diffusion_step(batch_size=ori_image.size(0))
        rand_noise = self.sample_noise(batch_size=ori_image.size(0))
        noisy_image = self.perform_diffusion_process(
            ori_image=ori_image,
            diffusion_step=rand_diffusion_step,
            rand_noise=rand_noise,
        )
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16,
        ) if self.device.type == "cuda" else contextlib.nullcontext():
            pred_noise = self(
                noisy_image=noisy_image, diffusion_step=rand_diffusion_step, label=label,
            )
            return F.mse_loss(pred_noise, rand_noise, reduction="mean")

    # @torch.enable_grad()
    def get_classifier_grad(self, noisy_image, diffusion_step, label):
        # with torch.enable_grad():
        # x_in = noisy_image.detach().requires_grad_(True)
        noisy_image.requires_grad = True
        out = self.classifier(
            noisy_image=noisy_image,
            # noisy_image=x_in,
            diffusion_step=diffusion_step,
            label=label,
        )
        log_prob = F.log_softmax(out, dim=-1)
        selected = log_prob[range(log_prob.size(0)), label]
        # "$\nabla_{x_{t}}\log{p_{\phi}}(y \vert x)$"
        return torch.autograd.grad(outputs=selected.sum(), inputs=noisy_image)[0]

    # @torch.inference_mode()
    def take_denoising_step(self, noisy_image, diffusion_step_idx, label):
        diffusion_step = self.batchify_diffusion_steps(
            diffusion_step_idx=diffusion_step_idx, batch_size=noisy_image.size(0),
        )
        alpha_t = self.index(self.alpha, diffusion_step=diffusion_step)
        beta_t = self.index(self.beta, diffusion_step=diffusion_step)
        alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
        with torch.inference_mode():
            pred_noise = self(
                noisy_image=noisy_image.detach(), diffusion_step=diffusion_step, label=label,
            )
        model_mean = (1 / (alpha_t ** 0.5)) * (
            noisy_image - ((beta_t / ((1 - alpha_bar_t) ** 0.5)) * pred_noise)
        )
        model_var = beta_t

        grad = self.get_classifier_grad(
            noisy_image=noisy_image, diffusion_step=diffusion_step, label=label,
        )
        # "x_{t - 1}
        # = $\mathcal{N}(\mu + s\Sigma\nabla_{x_{t}}\log{p_{\phi}}(y \vert x), \Sigma)$"
        new_model_mean = model_mean + self.classifier_scale * model_var * grad

        if diffusion_step_idx > 0:
            rand_noise = self.sample_noise(batch_size=noisy_image.size(0))
        else:
            rand_noise = torch.zeros(
                size=(noisy_image.size(0), self.image_channels, self.img_size, self.img_size),
                device=self.device,
            )
        return new_model_mean + (model_var ** 0.5) * rand_noise

    # @torch.inference_mode()
    # def take_denoising_ddim_step(self, noisy_image, diffusion_step_idx, label):
    #     diffusion_step = self.batchify_diffusion_steps(
    #         diffusion_step_idx=diffusion_step_idx, batch_size=noisy_image.size(0),
    #     )
    #     alpha_bar_t = self.index(self.alpha_bar, diffusion_step=diffusion_step)
    #     pred_noise = self(noisy_image=noisy_image.detach(), diffusion_step=diffusion_step)
    #     grad = self.get_classifier_grad(
    #         noisy_image=noisy_image, diffusion_step=diffusion_step, label=label,
    #     )
    #     new_pred_noise = pred_noise - (1 - alpha_bar_t) ** 0.5 * grad
    #     # "$x_{t - 1}
    #     # = \sqrt{\bar{\alpha}_{t - 1}}\Bigg(\frac{x_{t} - \sqrt{1 - \bar{\alpha}_{t}}\hat{\epsilon}}{\sqrt{\bar{\alpha}_{t}}}\Bigg)
    #     # + \sqrt{1 - \bar{\alpha}_{t - 1}}\hat{\epsilon}$"
    #     return (prev_alpha_bar_t ** 0.5) * (
    #         (noisy_image -  ((1 - alpha_bar_t) ** 0.5) * new_pred_noise) / ((alpha_bar_t) ** 0.5)
    #     ) + (1 - prev_alpha_bar_t) * new_pred_noise

    def perform_denoising_process(self, noisy_image, start_diffusion_step_idx, label, n_frames=None):
        if n_frames is not None:
            frames = list()

        x = noisy_image
        pbar = tqdm(range(start_diffusion_step_idx, -1, -1), leave=False)
        for diffusion_step_idx in pbar:
            pbar.set_description("Denoising...")

            x = self.take_denoising_step(x, diffusion_step_idx=diffusion_step_idx, label=label)

            if n_frames is not None and (
                diffusion_step_idx % (self.n_diffusion_steps // n_frames) == 0
            ):
                frames.append(self._get_frame(x))
        return frames if n_frames is not None else x

    def sample(self, batch_size, label):
        rand_noise = self.sample_noise(batch_size=batch_size)
        return self.perform_denoising_process(
            noisy_image=rand_noise,
            start_diffusion_step_idx=self.n_diffusion_steps - 1,
            label=label,
            n_frames=None,
        )
