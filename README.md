# 1. Pre-trained Models
- [classifier_guidance-cifar10.pth](https://drive.google.com/file/d/1MaCkJPspB-U-2H_Sug1CdReQXilHppfP/view?usp=sharing)

# 2. Samples
| `classifier_scale=20` |
|:-:|
| <img src="https://github.com/KimRass/KimRass/assets/67457712/bd878804-13b8-4b18-a345-c6da1f6b6321" width="250"> |

# 3. Theoretical Background
$$x_{t - 1} \leftarrow \text{sample from } \mathcal{N}(\mu + s\Sigma\nabla_{x_{t}}\log{p_{\phi}}(y \vert x), \Sigma)$$
$$\hat{\epsilon} \leftarrow \epsilon_{\theta}(x_{t}) - \sqrt{1 - \bar{\alpha}_{t}}\nabla_{x_{t}}\log{p_{\phi}}(y \vert x)$$
$$x_{t - 1} \leftarrow \sqrt{\bar{\alpha}_{t - 1}}\Bigg(\frac{x_{t} - \sqrt{1 - \bar{\alpha}_{t}}\hat{\epsilon}}{\sqrt{\bar{\alpha}_{t}}}\Bigg) + \sqrt{1 - \bar{\alpha}_{t - 1}}\hat{\epsilon}$$

# 4. To Do
- [ ] AdaGN
- [ ] BiGGAN Upsample/Downsample
- [ ] Improved DDPM sampling
- [ ] Conditional/Unconditional models
- [ ] Super-resolution model
- [ ] Interpolation

# 5. References
- [1] https://github.com/openai/guided-diffusion
- [2] https://github.com/openai/improved-diffusion
