# NonparBayesCI (rewrite)

This is a clean-room rewrite of the F-localization confidence interval code for nonparametric Empirical Bayes,
with the following updates:

- LP problems (DKW / Gauss / Wasserstein constraints) are solved with **Gurobi** via JuMP.
- Conic problems (χ² localization) continue to use a conic solver (default: **Clarabel**).
- Added **Wasserstein F-localization** with radius options:
  - `:dkw` finite-sample radius via DKW (bounded support)
  - `:bootstrap` bootstrap-calibrated radius
  - `:clt` CLT-calibrated radius for 1D Wasserstein-1 (Gaussian-process simulation)

- Added **smooth/regularized Wasserstein-1** localization (Goldfeld et al. 2024):
  - set `[localization] regularization = "smooth"` and provide `smooth_sigma`
  - currently supports `radius_method = "clt"` via `wasserstein_smooth_clt`

We removed the AMARI method and do not include non-smooth estimands (e.g., indicator-type functionals)
in the example experiment configs.

## Quick start (local)

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

## Posterior mean simulation (paper Figure 4-style)

This repository includes a self-contained simulation + plotting pipeline for the
Gaussian empirical Bayes **posterior mean** experiment used in Ignatiadis & Wager (2021).

It produces **four** figures:

- Spiky CI bands
- NegSpiky CI bands
- Spiky pointwise coverage
- NegSpiky pointwise coverage

Run both priors locally:

```bash
julia --project=. scripts/paper_fig4_postmean_pipeline.jl --outdir results/paper_fig4 --nreps 4000
```

By default, the Figure-4 scripts use a smaller sample size `n = 500` per Monte Carlo replicate
to make local iteration faster. To match the paper's simulation setting (`n = 5000`), add `--n 5000`:

```bash
julia --project=. scripts/paper_fig4_postmean_pipeline.jl --outdir results/paper_fig4 --n 5000 --nreps 4000
```

Run one prior only:

```bash
julia --project=. scripts/paper_fig4_postmean.jl --prior spiky --outdir results/paper_fig4 --nreps 4000
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

Edit the SLURM scripts under `scripts/slurm/` to match your cluster
(module loads, partitions, Gurobi license paths, etc.).

### Paper replication (recommended default)

To run the *paper-style* experiments (Figure 4 posterior mean + prostate real data), submit:

```bash
sbatch scripts/slurm/pipeline_slurm.sbatch
```

This uses the **paper settings** for Fig-4 by default: `n=5000`, `nreps=400`, `z=-3:0.2:3`.

### Figure 4 only (single sequential job)

```bash
sbatch scripts/slurm/pipeline_fig4_postmean.sbatch
```

This also defaults to the paper settings, and you may override by passing flags to `sbatch`, e.g.

```bash
sbatch scripts/slurm/pipeline_fig4_postmean.sbatch --n 2000 --nreps 200
```

### Smoke tests (small TOML configs)

The demo configs under `./configs` are intentionally small sanity checks
and **do not** correspond to the paper's Figure-4 simulation (they use `z0_list = [-2, -1, 0, 1, 2]`).
To run them on SLURM:

```bash
sbatch scripts/slurm/pipeline_smoke_slurm.sbatch
```

## Real data: Prostate z-scores (Figure 2-style)

To reproduce the prostate real-data figures (empirical CDF with DKW band + posterior-mean
CI bands from multiple F-localization radii), run:

```bash
julia --project=. scripts/paper_fig2_prostate.jl --outdir results/paper_fig2_prostate
```

The script looks for `data/prostate/prostz.txt`. If missing, it will try to auto-download it.
If your cluster disallows outbound internet, download `prostz.txt` manually and place it at
`data/prostate/prostz.txt` (see `data/prostate/README.md`).

On SLURM you can run it via:

```bash
sbatch scripts/slurm/fig2_prostate.sbatch
```

### SLURM (recommended): parallel Figure-4 posterior mean via job arrays

For large `nreps` (e.g. 4000) the Figure-4 posterior-mean simulation can be
slow if run as one sequential SLURM job. The repository includes a **job-array**
workflow that parallelizes across:

- the z-grid points, and
- **replicate blocks** (e.g. 100 Monte Carlo replicates per task)

This pipeline submits:

1. A **precompute** job that simulates the datasets and caches *all*
   data-dependent localization quantities (empirical CDF/KDE + radii) to disk.
2. A **job array** with one task per *(prior, z0-grid-point, replicate-block)*
   to solve the LPs.
3. A final **aggregate** job that merges partial outputs and produces the same
   four plots as the sequential pipeline.

Submit from the login node:

```bash
bash scripts/slurm/submit_fig4_postmean_parallel.sh --outdir results/paper_fig4 --nreps 400
```

By default the submit helper uses:

- `--block_size 100` (≈ 100 replicates per array task)
- `--max_concurrent 6` (≈ up to 6 array tasks at once, i.e. no more than ~6 nodes)

You can override these:

```bash
bash scripts/slurm/submit_fig4_postmean_parallel.sh --outdir results/paper_fig4 --nreps 400 --block_size 50 --max_concurrent 6
```

If you prefer `sbatch` for submission, you can also submit the included
submitter job:

```bash
sbatch scripts/slurm/fig4_postmean_submit_parallel.sbatch
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
- This code uses a **single shared Gurobi environment per Julia process** (recommended pattern for Gurobi.jl).

## Using your own 1D dataset (CSV)

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

- The Wasserstein DKW radius assumes **bounded support**. For real-valued unbounded data, prefer `radius_method = "bootstrap"` or `"clt"`.
- For regularized/smooth Wasserstein (Goldfeld et al. 2024), set:
  - `regularization = "smooth"`
  - `radius_method = "clt"`
  - `smooth_sigma = ...`
