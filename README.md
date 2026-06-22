# Surrogate-Assisted CMA-ES on COCO/BBOB

Benchmark study of neural and classical **surrogate models** under three
**evolution-control** strategies in surrogate-assisted CMA-ES, evaluated on the
COCO/BBOB suite and compared against established CMA-ES baselines.

This follows the standard experimental design described in `docs/`
(see [docs/source-materials-index.md](docs/source-materials-index.md) and
[the experiment protocol](the experiment protocol)).

## Experiment matrix

| Axis | Values |
|---|---|
| Surrogate models (6) | `gp`, `bnn_mc_dropout`, `pfn_bnn`, `pfn_hebo`, `gmm_np`, `pfn_transformer` |
| Evolution controls (3) | `lmm`, `dts`, `lq` (Pitra et al., GECCO 2021) |
| Base optimizer | IPOP-CMA-ES (50 restarts, IncPopSize=2, σ₀=8/3, λ=8+⌊6·logD⌋, x₀∼U[−4,4]ᴰ) |
| Functions (Stage 1) | noiseless BBOB `f1–f24` |
| Functions (Stage 1b) | noisy `f101–f130` |
| Dimensions | 2, 3, 5, 10, 20 |
| Instances | 1–15 |
| Budget | 250 × dimension |
| Checkpoints | 50 FE/D and 250 FE/D |
| Metric | Δµf over target interval [1e-13, 1e7]; median convergence (log₁₀ Δf) |
| Stats | pairwise % wins; Wilcoxon signed-rank + Holm correction |
| Baselines | CMA-ES-2019, DTS-CMA-ES, LMM-CMA-ES, LQ-CMA-ES (COCO archives) |
| Stage 2 | `fsim` buoy benchmark (GECCO 2016 WEC, 24 settings) — separate stage |

18 internal variants = 6 models × 3 evolution controls.

## Status

Built in phases (see the task list). Stage 1 = noiseless `f1–f24`, end-to-end and verified, first.

- [x] **P0** Foundations — dependencies, repo layout, transformer weight cache
- [x] **P1** IPOP-CMA-ES alignment to protocol settings
- [x] **P2** COCO runner (noiseless `f1–f24`)
- [x] **P3** Metrics (Δµf, 12 groups) & statistics (Wilcoxon + Holm)
- [x] **P4** Baseline integration (COCO archives)
- [ ] **P5** Smoke-verify the full chain on a small matrix
- [ ] **P6** Full-run package for GPU / long unattended run
- [ ] **P8** Project report (Stage 1)

## Running

```bash
# 1. Run the experiment (full Stage 1, or a smoke subset)
python scripts/run_experiment.py --config configs/stage1_noiseless.yaml

# 2. Analyze: Delta-mu-f tables, stats, ablation, convergence plots
#    --baselines folds in CMA-ES-2019 / DTS / LMM / LQ; --cocopp adds the ECDF report
python scripts/analyze.py --baselines --cocopp
```

## Setup

```bash
python -m pip install -r requirements.txt
```

> `coco-experiment` is a C extension. On this machine it built a `cp314` wheel
> from source (a working C/MSVC toolchain is present). On a machine without a
> compiler, install a Python version with prebuilt COCO wheels.

The transformer surrogate (`pfn_transformer`) pretrains once on a mixed prior and
caches weights to `models/models/weights/pfn_bnn_transformer.pt`; later runs load
them with zero training cost. To (re)generate explicitly:

```bash
python models/models/pfn_transformer.py --steps 10000   # --force to overwrite
```

## Layout

```
docs/        GECCO 2016 fsim source
models/      the surrogate-model package (gp, bnn_mc_dropout, pfn_*, gmm_np) + CMA optimizer
src/sacma/   experiment framework (IPOP wrapper, COCO runner, metrics, stats, plots)
configs/     experiment configuration files
results/     COCO-format output + processed tables/plots (gitignored)
reports/     generated project report
scripts/     entry points
```
