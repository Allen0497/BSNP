# BSNP: Bias-Spectrum Neural Processes for Parametric PDEs

Official PyTorch code for **Bias-Spectrum Neural Processes (BSNP)**, a
unified meta-learning framework that combines **weak structural priors**
(ConvCNP-style translation-equivariant encoder) with **strong physical priors**
(PDE-residual regularization via stochastic collocation).

## Method Overview

BSNP solves parametric PDE families in a meta-learning setting: given a sparse,
noisy set of context observations $C = \{(x_i, y_i)\}_{i=1}^{N_c}$ and PDE
parameters $\lambda$, predict the solution $u(x^*)$ at arbitrary query
locations $x^*$ with calibrated uncertainty.

Key design:

1. **ConvCNP encoder (weak bias, Paper Eq. 7-11):** kernel-interpolates
   irregular context onto a regular grid via the RBF kernel
   $\kappa_\rho(s, x) = \exp\left(-\|s - x\|^2 / (2\rho^2)\right)$, applies a
   translation-equivariant CNN, and global-pools to obtain a variational
   latent posterior $q_\theta(z \mid C, \lambda) = \mathcal{N}(\mu_\theta, \mathrm{diag}(\sigma_\theta^2))$.

2. **Decoder (Paper Eq. 12-13):** conditions grid features
   $\{g_m\}_{m=1}^{M}$ on a sampled latent $z$, then interpolates to query
   points via normalized kernel weights
   $\alpha_m(x^*) = \kappa_\rho(s_m, x^*) / \sum_{m'} \kappa_\rho(s_{m'}, x^*)$
   to produce a heteroscedastic Gaussian predictive distribution
   $p(y^* \mid x^*, C, \lambda, z) = \mathcal{N}\!\left(\mu(x^*),\, \sigma^2(x^*)\right)$.

3. **Mean-field physics loss (Paper Section 4.3):** PDE residuals are
   evaluated on the mean field $\bar{u}(x) = \mu_\theta(x; C, \lambda, z_{\mathrm{phys}})$
   where $z_{\mathrm{phys}} \sim q_\theta(z \mid C, \lambda)$ is sampled from
   the **context-only** posterior (avoids target leakage). Residuals are
   computed via PyTorch autograd and scored with stochastic Monte Carlo
   collocation (Paper Eq. 16):
   $\widehat{\mathcal{J}}_{\mathrm{phys}}(\theta; X_r) = \frac{1}{N_r} \sum_{k=1}^{N_r} \left\| \mathcal{G}_\lambda[\bar{u}](x_k^r) \right\|_2^2,
   \quad x_k^r \overset{\mathrm{i.i.d.}}{\sim} p_r$.

4. **Total objective (Paper Eq. 21):**
   $\mathcal{L}(\theta) = \mathcal{L}_{\mathrm{data}}(\theta) + \beta \cdot \widehat{\mathcal{J}}_{\mathrm{phys}}(\theta; X_r) + \beta_0 \cdot \widehat{\mathcal{J}}_{\partial}(\theta; X_\partial)$,
   where $\mathcal{L}_{\mathrm{data}}$ is the standard NP data ELBO (Paper Eq. 19):
   $\mathcal{L}_{\mathrm{data}}(\theta) = \mathbb{E}_{q_\theta(z \mid C \cup (T, Y_T), \lambda)} \!\left[ \log p_\theta(Y_T \mid T, C, \lambda, z) \right] - \mathrm{KL}\!\left( q_\theta(z \mid C \cup (T, Y_T), \lambda) \,\|\, q_\theta(z \mid C, \lambda) \right)$.


## Project Structure

```
BSNP/
|-- models/
|   |-- encoder.py      # ConvCNP encoder: grid deposition + CNN + pooling
|   |-- decoder.py      # kernel interpolation + heteroscedastic heads
|   |-- bsnp.py         # full model: dual posterior + reparameterized sampling
|   \-- losses.py       # ELBO, physics residual, total loss
|-- data/
|   |-- poisson.py      # 1D nonlinear Poisson dataset + FD solver
|   |-- burgers.py      # Burgers dataset (uses stable solver below)
|   |-- burgers_solver.py  # integrating-factor spectral Burgers solver
|   \-- collate.py      # variable-length context batching (padding + mask)
|-- utils/
|   |-- pde_residuals.py   # autograd PDE residual operators
|   |-- metrics.py      # MNSE / NLL / ECP
|   \-- visualization.py   # paper-style plots
|-- scripts/
|   |-- train.py        # main training loop (DDP + AMP + grad accumulation)
|   |-- evaluate.py     # test-set evaluation from checkpoint
|   \-- visualize.py    # generate figures from checkpoint
|-- configs/
|   |-- poisson.yaml    # best config for 1D nonlinear Poisson
|   \-- burgers.yaml    # best config for Burgers
|-- requirements.txt
\-- README.md
```


## Environment Setup

### Hardware & Software (paper appendix C.1)

- **GPU:** NVIDIA RTX 4090 (16 GB per GPU), up to 8 GPUs
- **CPU:** Intel(R) Xeon(R) Silver 4416+ (40 cores)
- **OS:** Ubuntu 22.04.5 LTS
- **CUDA:** 11.8
- **cuDNN:** 8.6
- **PyTorch:** 2.0.0
- **NumPy:** 1.26.3
- **Scikit-learn:** 1.5.0
- **Pandas:** 2.2.2

### Install

```bash
cd BSNP

# (Recommended) create an isolated environment
conda create -n bsnp python=3.11 -y
conda activate bsnp

# Install PyTorch matching your CUDA version from pytorch.org, e.g.:
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```


## Training

### Single-GPU

```bash
# 1D nonlinear Poisson (best config: ~10 minutes on RTX 4090)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/poisson.yaml

# Burgers (best config: ~50 minutes on RTX 4090, 60k steps)
CUDA_VISIBLE_DEVICES=0 python scripts/train.py --config configs/burgers.yaml
```

On the first run, the solver will pre-generate ground-truth solutions and
cache them under `data/cache/` for reuse. Subsequent runs load instantly.

### Multi-GPU (DistributedDataParallel)

```bash
torchrun --nproc_per_node=4 scripts/train.py --config configs/poisson.yaml
```

Note: DDP initialization may be slow on some environments. The single-GPU
path with `CUDA_VISIBLE_DEVICES=N` is a reliable fallback.

### Training Features

- **Mixed precision (AMP)** via `torch.amp.autocast` + `GradScaler`
- **Gradient accumulation** via `grad_accum_steps` in config
- **Persistent DataLoader workers** (`num_workers=4`, `pin_memory=True`)
- **cuDNN benchmark mode** (`torch.backends.cudnn.benchmark = True`)
- **Variable-length context** via padding + boolean mask in the collate fn

### Resume from Checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train.py \
    --config configs/poisson.yaml \
    --resume experiments/poisson_1d_ckpt_10000.pt
```


## Evaluation

```bash
python scripts/evaluate.py \
    --config configs/poisson.yaml \
    --ckpt   experiments/poisson_1d_final.pt
```

Reports mean +/- std over the test set for MNSE, NLL, and ECP(90%).


## Visualization

```bash
python scripts/visualize.py \
    --config configs/poisson.yaml \
    --ckpt   experiments/poisson_1d_final.pt \
    --outdir experiments/figures
```

Produces:

- `fig2_predictions.png` -- Paper Figure 2 style grid of predictions (Poisson only)
- `training_curves.png`  -- loss / physics residual / validation MNSE over steps
- `noise_robustness.png` -- Paper Table 2 style MNSE vs noise level sweep


## Customizing Configs

All hyperparameters live in `configs/*.yaml`. The most impactful ones:

| Key                        | Effect                                           |
|----------------------------|--------------------------------------------------|
| `model.grid_size`          | ConvCNP grid resolution (M for 1D, [H,W] for 2D) |
| `model.latent_dim`         | bottleneck dimension of `q(z | C, lambda)`       |
| `model.kernel_lengthscale` | RBF lengthscale rho (larger for higher-D grids)  |
| `data.n_context_max`       | upper bound on context set size                  |
| `training.beta`            | weight of interior PDE residual                  |
| `training.beta_0`          | weight of boundary/initial-condition residual    |
| `training.n_steps`         | number of gradient steps                         |


## Adding a New PDE

1. Add a solver + `Dataset` subclass under `data/`.
2. Add PDE and boundary residual functions in `utils/pde_residuals.py`.
3. Wire them into `scripts/train.py::compute_physics`.
4. Write a config in `configs/` referencing `pde.type`.


## Notes

- `torch.compile` is currently **not** wrapped around the model because it
  conflicts with autograd-based physics residuals in recent PyTorch versions.
  It can be re-enabled once the underlying issue is resolved upstream.
- The Burgers parameter range uses `nu >= 0.005` instead of the paper's
  `nu >= 0.001` because the spectral solver becomes unstable for very low
  viscosities (near-shock regime).


## Citation

Hui Li, Huafeng Liu, Chenguang Li, Tianxiao Zhang, Yajun Yang, and Liping Jing. Bias-Spectrum Neural Processes for Parametric PDEs: Architecture Priors Meet PDE Constraints. In Proceedings of the 43rd International Conference on Machine Learning (ICML 2026).

```
BibTeX：
  @inproceedings{li2026bsnp,
   title     = {Bias-Spectrum Neural Processes for Parametric {PDEs Architecture Priors Meet {PDE} Constraints},
   author    = {Li, Hui and Liu, Huafeng and Li, Chenguang and Zhang, Tianxiao and Yang, Yajun and Jing, Liping},
   booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
   series    = {Proceedings of Machine Learning Research},
   publisher = {PMLR},
   year      = {2026},
 }
```
