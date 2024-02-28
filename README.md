# 1. Theoretical Background
$$x_{t - 1} \leftarrow \text{sample from } \mathcal{N}(\mu + s\Sigma\nabla_{x_{t}}\log{p_{\phi}}(y \vert x), \Sigma)$$
$$\hat{\epsilon} \leftarrow \epsilon_{\theta}(x_{t}) - \sqrt{1 - \bar{\alpha}_{t}}\nabla_{x_{t}}\log{p_{\phi}}(y \vert x)$$
$$x_{t - 1} \leftarrow \sqrt{\bar{\alpha}_{t - 1}}\Bigg(\frac{x_{t} - \sqrt{1 - \bar{\alpha}_{t}}\hat{\epsilon}}{\sqrt{\bar{\alpha}_{t}}}\Bigg) + \sqrt{1 - \bar{\alpha}_{t - 1}}\hat{\epsilon}$$

# 2. References
- https://github.com/openai/guided-diffusion
- https://github.com/openai/improved-diffusion
