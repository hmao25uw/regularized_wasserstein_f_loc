module Experiments

using Random
using Printf
using TOML
using DelimitedFiles
using CSV
using DataFrames

# Solvers (for building SolverConfig)
using Gurobi
using HiGHS
using Clarabel

using ..Types: SolverConfig
using ..Localization: DKWLocalization, GaussLocalization, WassersteinLocalization, Chi2Localization,
                      prepare_localization_stats
using ..Likelihoods: gaussian_likelihood, binomial_likelihood, poisson_likelihood
using ..Methods: EBProblem, f_localization_ci

export run_config

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

function _ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

function _parse_vector(x)
    # TOML will parse [1,2,3] as Vector{Any}; coerce to Float64
    return Float64.(x)
end

function _linspace(a::Real, b::Real, m::Integer)
    return collect(range(Float64(a), Float64(b), length=Int(m)))
end

# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

function _simulate_gaussian(cfg::Dict, rng::AbstractRNG)
    n = Int(cfg["n"])
    σ = Float64(cfg["sigma"])
    prior = cfg["prior"]

    if prior == "two_point"
        μ1 = Float64(cfg["mu1"])
        μ2 = Float64(cfg["mu2"])
        w1 = Float64(cfg["w1"])
        μ = rand(rng) < w1 ? μ1 : μ2
        # vectorize
        μs = [rand(rng) < w1 ? μ1 : μ2 for _ in 1:n]
        z = μs .+ σ .* randn(rng, n)
        return z, σ
    elseif prior == "normal"
        μ_mean = Float64(cfg["mu_mean"])
        μ_sd = Float64(cfg["mu_sd"])
        μs = μ_mean .+ μ_sd .* randn(rng, n)
        z = μs .+ σ .* randn(rng, n)
        return z, σ
    else
        error("Unknown gaussian prior=$(prior). Supported: two_point, normal")
    end
end

function _load_csv_column(path::AbstractString, colname::AbstractString)
    df = CSV.read(path, DataFrame)
    hasproperty(df, Symbol(colname)) || error("CSV file does not have a column named $(colname)")
    col = df[:, Symbol(colname)]

    # Allow missing values; drop them by default.
    n_total = length(col)
    vals = collect(skipmissing(col))
    n_drop = n_total - length(vals)
    if n_drop > 0
        @warn "Dropped $(n_drop) missing values from CSV column $(colname)."
    end
    isempty(vals) && error("CSV column $(colname) is empty (after dropping missing values).")

    try
        return Float64.(vals)
    catch
        error("Failed to convert CSV column $(colname) to Float64. Ensure it contains numeric values.")
    end
end

function _load_txt_vector(path::AbstractString)
    # Accept whitespace- or newline-delimited numeric vectors.
    raw = readdlm(path)
    v = vec(raw)
    isempty(v) && error("Text file $(path) appears to be empty.")
    try
        return Float64.(v)
    catch
        error("Failed to parse $(path) as a numeric vector. Ensure it contains only numbers.")
    end
end

# ---------------------------------------------------------------------------
# Localization builder
# ---------------------------------------------------------------------------

function _build_localization(cfg::Dict, z::Vector{Float64})
    method = cfg["method"]
    alpha = Float64(cfg["alpha"])

    if method == "dkw"
        t_grid = _linspace(cfg["t_min"], cfg["t_max"], cfg["t_points"])
        return DKWLocalization(alpha=alpha, t_grid=t_grid)

    elseif method == "wasserstein"
        t_grid = _linspace(cfg["t_min"], cfg["t_max"], cfg["t_points"])
        rmethod = Symbol(cfg["radius_method"])
        B = Int(get(cfg, "B", 2000))

        # Optional smooth/regularized Wasserstein (Goldfeld et al. 2024)
        reg = haskey(cfg, "regularization") ? Symbol(cfg["regularization"]) : :none
        smooth_sigma = haskey(cfg, "smooth_sigma") ? Float64(cfg["smooth_sigma"]) : 0.0
        kernel = haskey(cfg, "kernel") ? Symbol(cfg["kernel"]) : :uniform
        quad_points = Int(get(cfg, "quad_points", 33))

        support = if haskey(cfg, "support_min") && haskey(cfg, "support_max")
            (Float64(cfg["support_min"]), Float64(cfg["support_max"]))
        else
            nothing
        end
        return WassersteinLocalization(alpha=alpha,
                                      t_grid=t_grid,
                                      radius_method=rmethod,
                                      support=support,
                                      B=B,
                                      regularization=reg,
                                      smooth_sigma=smooth_sigma,
                                      kernel=kernel,
                                      quad_points=quad_points)

    elseif method == "gauss"
        x_grid = _linspace(cfg["x_min"], cfg["x_max"], cfg["x_points"])
        bandwidth = Float64(cfg["bandwidth"])
        B = Int(get(cfg, "B", 2000))
        return GaussLocalization(alpha=alpha, x_grid=x_grid, bandwidth=bandwidth, B=B)

    elseif method == "chi2"
        # Discrete support must be provided
        support = _parse_vector(cfg["support"])
        return Chi2Localization(alpha=alpha, z_support=support)

    else
        error("Unknown localization method=$(method).")
    end
end

# ---------------------------------------------------------------------------
# SolverConfig builder
# ---------------------------------------------------------------------------

function _build_solver(cfg::Dict)
    lp = get(cfg, "lp", "gurobi")
    conic = get(cfg, "conic", "clarabel")

    lp_opt = (lp == "gurobi") ? Gurobi.Optimizer :
             (lp == "highs")  ? HiGHS.Optimizer :
             error("Unknown LP solver $(lp). Use 'gurobi' or 'highs'.")

    conic_opt = (conic == "clarabel") ? Clarabel.Optimizer :
                error("Unknown conic solver $(conic). Use 'clarabel' (or extend).")

    silent = Bool(get(cfg, "silent", true))
    time_limit = Float64(get(cfg, "time_limit_sec", Inf))

    gurobi_threads = Int(get(cfg, "gurobi_threads", 1))
    gurobi_method = haskey(cfg, "gurobi_method") ? Int(cfg["gurobi_method"]) : nothing
    gurobi_numeric_focus = haskey(cfg, "gurobi_numeric_focus") ? Int(cfg["gurobi_numeric_focus"]) : nothing
    gurobi_seed = haskey(cfg, "gurobi_seed") ? Int(cfg["gurobi_seed"]) : 1

    return SolverConfig(
        lp_optimizer=lp_opt,
        conic_optimizer=conic_opt,
        silent=silent,
        time_limit_sec=time_limit,
        gurobi_threads=gurobi_threads,
        gurobi_method=gurobi_method,
        gurobi_numeric_focus=gurobi_numeric_focus,
        gurobi_seed=gurobi_seed,
    )
end

# ---------------------------------------------------------------------------
# Main: run a config file
# ---------------------------------------------------------------------------

"""
Run a single experiment config (TOML).

Writes a CSV with columns: z0, lower, upper

Returns the (z0, lower, upper) matrix.
"""
function run_config(config_path::AbstractString; outdir::AbstractString="results")
    cfg = TOML.parsefile(config_path)
    seed = Int(get(cfg, "seed", 12345))
    rng = MersenneTwister(seed)

    _ensure_dir(outdir)

    # ------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------
    data_cfg = cfg["data"]
    source = data_cfg["source"]

    z = if source == "simulation_gaussian"
        z_sim, σ = _simulate_gaussian(data_cfg, rng)
        z_sim
    elseif source == "csv"
        path = data_cfg["path"]
        col = data_cfg["column"]
        _load_csv_column(path, col)
    elseif source == "txt"
        path = data_cfg["path"]
        _load_txt_vector(path)
    else
        error("Unknown data.source=$(source).")
    end

    isempty(z) && error("Loaded data is empty. Check your data source settings.")

    # ------------------------------------------------------------
    # Likelihood & estimand
    # ------------------------------------------------------------
    model_cfg = cfg["model"]
    lik = model_cfg["likelihood"]

    likelihood = if lik == "gaussian"
        σ = Float64(model_cfg["sigma"])
        gaussian_likelihood(σ)
    elseif lik == "binomial"
        N = Int(model_cfg["N"])
        binomial_likelihood(N)
    elseif lik == "poisson"
        poisson_likelihood()
    else
        error("Unknown likelihood=$(lik).")
    end

    # Smooth estimand only (posterior mean by default)
    estimand_cfg = cfg["estimand"]
    h_name = estimand_cfg["h"]
    h = if h_name == "identity"
        μ -> μ
    elseif h_name == "square"
        μ -> μ^2
    else
        error("Unknown estimand.h=$(h_name). Allowed: identity, square")
    end

    # μ grid
    grid_cfg = cfg["grid"]
    mu_grid = _linspace(grid_cfg["mu_min"], grid_cfg["mu_max"], grid_cfg["mu_points"])

    problem = EBProblem(mu_grid; pdf=likelihood.pdf, cdf=likelihood.cdf, h=h)

    # ------------------------------------------------------------
    # Localization
    # ------------------------------------------------------------
    loc = _build_localization(cfg["localization"], z)

    # Precompute data-dependent localization stats ONCE so all z0 evaluations
    # share the same localization set (important for bootstrap/CLT-calibrated
    # radii).
    stats = prepare_localization_stats(loc, z; rng=rng)

    # ------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------
    solver = _build_solver(cfg["solver"])

    # ------------------------------------------------------------
    # Evaluate CI at requested z0 points
    # ------------------------------------------------------------
    z0_list = _parse_vector(cfg["estimand"]["z0_list"])
    out = Matrix{Float64}(undef, length(z0_list), 3)

    for (i, z0) in enumerate(z0_list)
        lower, upper = f_localization_ci(z, z0, problem, loc;
                                         solver=solver,
                                         rng=rng,
                                         stats_override=stats)
        out[i, 1] = z0
        out[i, 2] = lower
        out[i, 3] = upper
        @printf("z0 = %8.3f  CI = [% .6f, % .6f]\n", z0, lower, upper)
    end

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    fname = replace(basename(config_path), ".toml" => "") * "_ci.csv"
    outpath = joinpath(outdir, fname)
    writedlm(outpath, out, ',')
    @info "Wrote results to $(outpath)"

    return out
end

end # module
