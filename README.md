# Kernel Regularized Wasserstein F-Localization Confidence Intervals for Lipschitz Functionals

This repository implements **kernel regularized Wasserstein F-localization confidence intervals** 
for empirical Bayes estimands (focused on **posterior mean**, i.e. Lipschitz targets) and adds a 
modular **Wasserstein F-localization**. The original F-localization method was proposed by 
**Nikos Ignatiadis** (UChicago) and **Stefan Wager** (Stanford) in their paper titled 
`Confidence Intervals for Nonparametric Empirical Bayes Analysis`. Our new adaptive method 
will serve as an extension to the original **F-localiztion** method where the authors employed
a distribution-free DKW bound for the radius of the localization sets. In our 
**Regularized Wasserstein F-localization** method, we replaced the original KS metric with 
a kernel-regularized **1-Wasserstein** distance. Due to regularity concern around the 
**Modulus of Continuity**, we only implemented the confidence interval estimation for 
Lipschitz estimand.

## Radius Options
In addtional to the `DKW-F-Localization` and `Gauss-F-Localization` radii, we implemented 
three additional radii computation methods:
  - `:dkw` finite-sample radius via DKW (for bounded support)
  - `:bootstrap` bootstrap-calibrated radius
  - `:clt` Wasserstein CLT-calibrated radius for 1D Wasserstein-1

For regularized Wasserstein-1 we also:
- Added **smooth/regularized Wasserstein-1** localization (Goldfeld et al. 2024):
  - set `[localization] regularization = "smooth"` and provide `smooth_sigma`
  - currently supports `radius_method = "clt"` via `wasserstein_smooth_clt`

We removed the AMARI method and do not include non-smooth estimands (e.g., indicator-type functionals)
in the example experiment configs.

## Quick start (local computer no cluster)

From the repository root:

```bash
julia --project=. -e 'import Pkg; Pkg.instantiate(); Pkg.precompile()'
```

Run a **single** config:

```bash
julia --project=. scripts/run_experiment.jl --config configs/gaussian_sim_wasserstein_clt.toml --outdir results
```

Run **all** configs in `./configs` and then automatically generate plots:

```bash
julia --project=. scripts/pipeline_local.jl --outdir results
```

If you want to skip plotting during the run:

```bash
julia --project=. scripts/pipeline_local.jl --outdir results --no-plot
```

Generate plots **only** (after results exist):

```bash
julia --project=. scripts/plot_results.jl --outdir results
```

Outputs:

- CI CSVs: `results/*_ci.csv`
- Plots: `results/plots/ci_bands.png` and `results/plots/ci_length.png`

## Posterior Mean Simulation

This repository includes a self-contained simulation + plotting pipeline for the
Gaussian empirical Bayes **posterior mean** experiment used in 
Ignatiadis & Wager (2021). We used same data generating process with the same 
dataset size and number of replications.

Our simulation setup will produce 4 figures:

- Spiky CI bands
- NegSpiky CI bands
- Spiky pointwise coverage
- NegSpiky pointwise coverage

Run both priors locally:

```bash
julia --project=. scripts/paper_fig4_postmean_pipeline.jl --outdir results/paper_fig4 --nreps 400
```

By default, the Figure-4 scripts use a smaller sample size `n = 500` per Monte Carlo replicate
to make local iteration faster. To match the paper's simulation setting (`n = 5000`), add `--n 5000`:

```bash
julia --project=. scripts/paper_fig4_postmean_pipeline.jl --outdir results/paper_fig4 --n 5000 --nreps 400
```

Run one prior only:

```bash
julia --project=. scripts/paper_fig4_postmean.jl --prior spiky --outdir results/paper_fig4 --nreps 400
```

Outputs:

- CSV summaries:
  - `results/paper_fig4/postmean_ci_Spiky.csv`
  - `results/paper_fig4/postmean_coverage_Spiky.csv`
  - (and the analogous `NegSpiky` files)
- Plots:
  - `results/paper_fig4/plots/postmean_ci_bands_spiky.png`
  - `results/paper_fig4/plots/postmean_ci_bands_negspiky.png`
  - `results/paper_fig4/plots/postmean_coverage_spiky.png`
  - `results/paper_fig4/plots/postmean_coverage_negspiky.png`

## SLURM

Edit `scripts/slurm/pipeline_slurm.sbatch` to match your cluster (module loads, partitions, etc.), then:

```bash
sbatch scripts/slurm/pipeline_slurm.sbatch
```

Posterior mean simulation pipeline:

```bash
sbatch scripts/slurm/pipeline_fig4_postmean.sbatch
```

This SLURM script runs the same sequential pipeline as `scripts/pipeline_local.jl` and then generates the same plots.

In addition, `scripts/slurm/pipeline_slurm.sbatch` runs the **prostate z-scores real-data experiment**
(Figure 2-style) via `scripts/paper_fig2_prostate.jl`.

## Real data: Prostate z-scores (Figure 2-style)

To reproduce the prostate real-data figures (empirical CDF with DKW band + posterior-mean
CI bands from multiple F-localization radii), run:

```bash
julia --project=. scripts/paper_fig2_prostate.jl --outdir results/paper_fig2_prostate
```

The script looks for `data/prostate/prostz.txt`. If missing, it will try to auto-download it.
If your cluster disallows outbound internet, download `prostz.txt` manually and place it at
`data/prostate/prostz.txt` (see `data/prostate/README.md`).

### SLURM: parallel Figure-4 posterior mean via job arrays

For large `nreps` (e.g. 4000) the Figure-4 posterior-mean simulation can be
slow if run as one sequential SLURM job. The repository includes a **job-array**
workflow that parallelizes across the z-grid points.

This pipeline submits:

1. A **precompute** job that simulates the datasets and caches *all*
   data-dependent localization quantities (empirical CDF/KDE + radii) to disk.
2. A **job array** with one task per *(prior, z0-grid-point)* to solve the LPs.
3. A final **aggregate** job that merges partial outputs and produces the same
   four plots as the sequential pipeline.

Submit from the login node:

```bash
bash scripts/slurm/submit_fig4_postmean_parallel.sh --outdir results/paper_fig4 --nreps 4000
```

The final outputs are written to the **same** locations as the sequential run:

- `results/paper_fig4/postmean_ci_Spiky.csv`, `postmean_coverage_Spiky.csv`
- `results/paper_fig4/postmean_ci_NegSpiky.csv`, `postmean_coverage_NegSpiky.csv`
- plots under `results/paper_fig4/plots/`

Intermediate files are written under:

- `results/paper_fig4/precompute/` (cached stats)
- `results/paper_fig4/partial/` (one CSV per z0 task)

## Gurobi notes

- You need a working Gurobi installation and license on both local machine and cluster.
- If your cluster requires a module, load it in the `.sbatch` file (see template).
- This code uses a **single shared Gurobi environment per Julia process** 
(recommended pattern for Gurobi.jl).

## Using Your Own 1D Dataset

You can run the methods on any **1D numeric dataset** by providing a CSV and a column name.

1. Copy the template config:

   - `configs/templates/csv_wasserstein_example.toml`

2. Edit:

   - `[data] path` and `[data] column`
   - the `mu_grid` range in `[grid]`
   - the `t_grid` range in `[localization]`

3. Run:

```bash
julia --project=. scripts/run_experiment.jl --config path/to/your_config.toml --outdir results
```

Notes:

- The Wasserstein DKW radius assumes **bounded support**. For real-valued unbounded data, 
  prefer `radius_method = "bootstrap"` or `"clt"`.
- For regularized/smooth Wasserstein (Goldfeld et al. 2024), set:
  - `regularization = "smooth"`
  - `radius_method = "clt"`
  - `smooth_sigma = ...`

## Acknowledgements

This project includes code adapted from:
- UpstreamProject by Upstream Authors: https://github.com/nignatiadis/Empirikos.jl
  - Used from commit: <ecae57c>
  - License: MIT
  - Changes: refactored original, added methods for computing

See THIRD_PARTY_NOTICES.md for detailed attributions and license texts.

## License

Unless otherwise noted, original code in this repository is licensed under <Your License>.
Third-party code remains under its original license as listed in THIRD_PARTY_NOTICES.md.