#!/usr/bin/env julia
#
# Posterior mean simulation (Gaussian empirical Bayes) in the style of
# Ignatiadis & Wager (2021), Figure 4.
#
# This script:
#   1) simulates n iid observations Z_i = μ_i + ε_i,  ε_i ~ N(0, 1)
#      under either the Spiky or NegSpiky prior from equation (36).
#   2) constructs F-localization CIs for the posterior mean θ_G(z0) on a grid
#      z0 ∈ [-3, 3] with step 0.2 (customizable).
#   3) averages CI endpoints over Monte Carlo replicates.
#   4) computes pointwise coverage of the *true* posterior mean.
#   5) saves CSV summaries and two plots (CI band + coverage) for that prior.
#
# Usage examples:
#
#   julia --project=. scripts/paper_fig4_postmean.jl \
#       --prior spiky --outdir results/paper_fig4 --nreps 4000
#
#   julia --project=. scripts/paper_fig4_postmean.jl \
#       --prior negspiky --outdir results/paper_fig4 --nreps 4000
#

using Random
using Statistics
using Printf
using Distributions

using CSV
using DataFrames

# Make Plots work on headless machines (e.g., SLURM compute nodes)
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots

using NonparBayesCI


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

function parse_args(args)
    d = Dict{String,Any}(
        "prior" => "spiky",
        "outdir" => "results/paper_fig4",
        # NOTE:
        #   - The original Ignatiadis & Wager (2021) Figure-4 simulations use n = 5000.
        #   - We default to a smaller n for faster local iteration.
        #     To match the paper more closely, pass `--n 5000`.
        "n" => 500,
        "nreps" => 4000,          # 10× the paper's 400 replicates
        "alpha" => 0.05,
        "seed" => 12345,
        "zmin" => -3.0,
        "zmax" => 3.0,
        "zstep" => 0.2,
        # Optimization / calibration settings
        "lp_time_limit" => 600.0,
        "gurobi_threads" => 1,
        "boot_B" => 500,
        "clt_B" => 2000,
        # Discretization grids
        "mu_min" => -6.0,
        "mu_max" => 6.0,
        "mu_points" => 241,
        "t_min" => -6.0,
        "t_max" => 6.0,
        "t_points" => 301,
        "x_min" => -6.0,
        "x_max" => 6.0,
        "x_points" => 301,
        "gauss_bw" => 0.3,
        # Smooth Wasserstein (Goldfeld et al. 2024) parameters
        "smooth_sigma" => 0.25,
        "quad_points" => 33,
    )

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--prior"
            d["prior"] = lowercase(String(args[i+1])); i += 2
        elseif a == "--outdir"
            d["outdir"] = String(args[i+1]); i += 2
        elseif a == "--n"
            d["n"] = parse(Int, args[i+1]); i += 2
        elseif a == "--nreps"
            d["nreps"] = parse(Int, args[i+1]); i += 2
        elseif a == "--alpha"
            d["alpha"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--seed"
            d["seed"] = parse(Int, args[i+1]); i += 2
        elseif a == "--zmin"
            d["zmin"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--zmax"
            d["zmax"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--zstep"
            d["zstep"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--boot_B"
            d["boot_B"] = parse(Int, args[i+1]); i += 2
        elseif a == "--clt_B"
            d["clt_B"] = parse(Int, args[i+1]); i += 2
        elseif a == "--lp_time_limit"
            d["lp_time_limit"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--gurobi_threads"
            d["gurobi_threads"] = parse(Int, args[i+1]); i += 2
        else
            error("Unknown argument: $a")
        end
    end
    return d
end

function ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

function make_grid(a::Real, b::Real, step::Real)
    a = Float64(a); b = Float64(b); step = Float64(step)
    step > 0 || error("step must be > 0")
    npts = Int(round((b - a) / step)) + 1
    return [a + (i-1) * step for i in 1:npts]
end


# -----------------------------------------------------------------------------
# Priors (Ignatiadis & Wager 2021, eq. (36))
# -----------------------------------------------------------------------------

"""Return (weights, means, sds) for the simulation priors."""
function prior_spec(name::AbstractString)
    name_lc = lowercase(String(name))
    if name_lc in ("spiky", "gspiky")
        # G_spiky = 0.4 N(0, 0.25^2) + 0.2 N(0, 0.5^2) + 0.2 N(0, 1^2) + 0.2 N(0, 2^2)
        w = [0.4, 0.2, 0.2, 0.2]
        m = [0.0, 0.0, 0.0, 0.0]
        s = [0.25, 0.5, 1.0, 2.0]
        return (w=w, m=m, s=s, label="Spiky")
    elseif name_lc in ("negspiky", "negspiky", "gnegspiky")
        # G_negspiky = 0.8 N(-0.25, 0.25^2) + 0.2 N(0, 1^2)
        w = [0.8, 0.2]
        m = [-0.25, 0.0]
        s = [0.25, 1.0]
        return (w=w, m=m, s=s, label="NegSpiky")
    else
        error("Unknown prior=$(name). Use 'spiky' or 'negspiky'.")
    end
end

"""Sample μ ~ mixture of normals described by spec."""
function sample_mu(spec, n::Int, rng::AbstractRNG)
    w = Float64.(spec.w)
    m = Float64.(spec.m)
    s = Float64.(spec.s)
    length(w) == length(m) == length(s) || error("Mixture spec lengths mismatch")
    c = cumsum(w)
    abs(c[end] - 1.0) > 1e-8 && error("Mixture weights must sum to 1")

    μ = Vector{Float64}(undef, n)
    @inbounds for i in 1:n
        u = rand(rng)
        k = searchsortedfirst(c, u)
        μ[i] = m[k] + s[k] * randn(rng)
    end
    return μ
end

"""Simulate Z = μ + ε with ε ~ N(0, σ^2)."""
function simulate_z(spec, n::Int, σ::Real, rng::AbstractRNG)
    μ = sample_mu(spec, n, rng)
    return μ .+ Float64(σ) .* randn(rng, n)
end

"""Oracle posterior mean θ(z) under the mixture-of-normals prior and N(μ, σ^2) likelihood."""
function oracle_postmean(z::Real, spec, σ::Real)
    z = Float64(z)
    σ2 = Float64(σ)^2
    w = Float64.(spec.w)
    m = Float64.(spec.m)
    s = Float64.(spec.s)

    K = length(w)
    numer = 0.0
    denom = 0.0
    @inbounds for k in 1:K
        τ2 = s[k]^2
        v = τ2 + σ2
        # marginal density of Z under component k
        dens = w[k] * pdf(Normal(m[k], sqrt(v)), z)
        # posterior mean of μ within component k
        mpost = m[k] + (τ2 / v) * (z - m[k])
        numer += dens * mpost
        denom += dens
    end
    return numer / denom
end


# -----------------------------------------------------------------------------
# Matrices for caching (avoid rebuilding model likelihood matrices repeatedly)
# -----------------------------------------------------------------------------

function cdf_mat(cdf_fun::Function, t_grid::Vector{Float64}, mu_grid::Vector{Float64})
    m = length(t_grid)
    p = length(mu_grid)
    out = Matrix{Float64}(undef, m, p)
    @inbounds for i in 1:m
        t = t_grid[i]
        for j in 1:p
            out[i, j] = Float64(cdf_fun(t, mu_grid[j]))
        end
    end
    return out
end

function pdf_mat(pdf_fun::Function, x_grid::Vector{Float64}, mu_grid::Vector{Float64})
    m = length(x_grid)
    p = length(mu_grid)
    out = Matrix{Float64}(undef, m, p)
    @inbounds for i in 1:m
        x = x_grid[i]
        for j in 1:p
            out[i, j] = Float64(pdf_fun(x, mu_grid[j]))
        end
    end
    return out
end


# -----------------------------------------------------------------------------
# Main simulation routine
# -----------------------------------------------------------------------------

function run_one_prior(; prior::String,
                        outdir::String,
                        n::Int,
                        nreps::Int,
                        alpha::Float64,
                        seed::Int,
                        zmin::Float64,
                        zmax::Float64,
                        zstep::Float64,
                        boot_B::Int,
                        clt_B::Int,
                        lp_time_limit::Float64,
                        gurobi_threads::Int,
                        mu_min::Float64,
                        mu_max::Float64,
                        mu_points::Int,
                        t_min::Float64,
                        t_max::Float64,
                        t_points::Int,
                        x_min::Float64,
                        x_max::Float64,
                        x_points::Int,
                        gauss_bw::Float64,
                        smooth_sigma::Float64,
                        quad_points::Int)

    spec = prior_spec(prior)
    σ = 1.0

    # Grids
    z0_grid = make_grid(zmin, zmax, zstep)
    mu_grid = collect(range(mu_min, mu_max, length=mu_points))
    t_grid = collect(range(t_min, t_max, length=t_points))
    x_grid = collect(range(x_min, x_max, length=x_points))

    # EB setup
    lik = NonparBayesCI.gaussian_likelihood(σ)
    problem = NonparBayesCI.EBProblem(mu_grid; pdf=lik.pdf, cdf=lik.cdf, h=μ -> μ)

    # Solver (LPs use Gurobi by default)
    solver = NonparBayesCI.SolverConfig(
        silent=true,
        time_limit_sec=lp_time_limit,
        gurobi_threads=gurobi_threads,
        gurobi_seed=1,
    )

    # Localization methods to compare
    methods = [
        (name="DKW-F",          loc=NonparBayesCI.DKWLocalization(alpha=alpha, t_grid=t_grid),
                                linestyle=:dot,   kind=:orig),
        (name="Gauss-F",        loc=NonparBayesCI.GaussLocalization(alpha=alpha, x_grid=x_grid, bandwidth=gauss_bw, B=boot_B),
                                linestyle=:dot,   kind=:orig),
        (name="W₁ (DKW)",       loc=NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_grid, radius_method=:dkw,
                                                                          support=(t_min, t_max), B=clt_B),
                                linestyle=:solid, kind=:new),
        (name="W₁ (boot)",      loc=NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_grid, radius_method=:bootstrap, B=boot_B),
                                linestyle=:solid, kind=:new),
        (name="W₁ (CLT)",       loc=NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_grid, radius_method=:clt, B=clt_B),
                                linestyle=:solid, kind=:new),
        (name="Smooth-W₁ (CLT)",loc=NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_grid, radius_method=:clt, B=clt_B,
                                                                          regularization=:smooth,
                                                                          smooth_sigma=smooth_sigma,
                                                                          kernel=:uniform,
                                                                          quad_points=quad_points),
                                linestyle=:solid, kind=:new),
    ]

    # Precompute model matrices for each method (huge speedup vs rebuilding every solve)
    caches = Dict{String,NamedTuple}()
    # IMPORTANT: For DKW-F, the constraint rows must align with the *exact*
    # ordering of `loc.t_grid` used to compute `Fhat`.
    caches["DKW-F"] = (Ccdf=cdf_mat(lik.cdf, Float64.(t_grid), mu_grid),)
    caches["W₁ (DKW)"] = (Ccdf=cdf_mat(lik.cdf, sort(t_grid), mu_grid),)
    caches["W₁ (boot)"] = (Ccdf=cdf_mat(lik.cdf, sort(t_grid), mu_grid),)
    caches["W₁ (CLT)"] = (Ccdf=cdf_mat(lik.cdf, sort(t_grid), mu_grid),)
    caches["Gauss-F"] = (PdfMat=pdf_mat(lik.pdf, x_grid, mu_grid),)
    caches["Smooth-W₁ (CLT)"] = (Ccdf=NonparBayesCI.smooth_cdf_mat(lik.cdf, sort(t_grid), mu_grid;
                                                                   sigma=smooth_sigma,
                                                                   kernel=:uniform,
                                                                   quad_points=quad_points),)

    # Output
    ensure_dir(outdir)
    plots_dir = ensure_dir(joinpath(outdir, "plots"))

    # Accumulators
    M = length(methods)
    K = length(z0_grid)
    lower_sum = zeros(Float64, M, K)
    upper_sum = zeros(Float64, M, K)
    cover_cnt = zeros(Int, M, K)
    valid_cnt = zeros(Int, M, K)
    fail_cnt  = zeros(Int, M, K)

    # Truth (deterministic)
    theta_true = [oracle_postmean(z0, spec, σ) for z0 in z0_grid]

    rng = MersenneTwister(seed)
    tick = max(1, Int(floor(nreps / 20)))

    @info "Running posterior mean simulation" prior=spec.label n=n nreps=nreps alpha=alpha zgrid="[$zmin:$zstep:$zmax]"

    for rep in 1:nreps
        z = simulate_z(spec, n, σ, rng)

        for (midx, meth) in enumerate(methods)
            loc = meth.loc
            cache = caches[meth.name]
            stats = NonparBayesCI.prepare_localization_stats(loc, z; rng=rng)

            @inbounds for (k, z0) in enumerate(z0_grid)
                out = try
                    NonparBayesCI.f_localization_ci(z, z0, problem, loc;
                                                    solver=solver,
                                                    rng=rng,
                                                    stats_override=stats,
                                                    mat_cache=cache)
                catch
                    nothing
                end
                if out === nothing
                    fail_cnt[midx, k] += 1
                    continue
                end
                lo, hi = out
                if !(isfinite(lo) && isfinite(hi))
                    fail_cnt[midx, k] += 1
                    continue
                end
                lower_sum[midx, k] += lo
                upper_sum[midx, k] += hi
                valid_cnt[midx, k] += 1
                θ = theta_true[k]
                if (lo - 1e-8) <= θ <= (hi + 1e-8)
                    cover_cnt[midx, k] += 1
                end
            end
        end

        if rep % tick == 0 || rep == 1 || rep == nreps
            @info("progress", rep=rep, nreps=nreps)
        end
    end

    lower_mean = fill(NaN, M, K)
    upper_mean = fill(NaN, M, K)
    coverage = fill(NaN, M, K)
    for midx in 1:M, k in 1:K
        if valid_cnt[midx, k] > 0
            lower_mean[midx, k] = lower_sum[midx, k] / valid_cnt[midx, k]
            upper_mean[midx, k] = upper_sum[midx, k] / valid_cnt[midx, k]
            coverage[midx, k] = cover_cnt[midx, k] / valid_cnt[midx, k]
        end
    end
    if any(fail_cnt .> 0)
        @warn "Some CI solves failed and were skipped in averages" total_failed=sum(fail_cnt) total_solved=sum(valid_cnt)
    end

    # ------------------------------------------------------------------
    # Save CSVs
    # ------------------------------------------------------------------
    df_ci = DataFrame(
        prior = String[],
        method = String[],
        z0 = Float64[],
        lower = Float64[],
        upper = Float64[],
        length = Float64[],
        theta_true = Float64[],
        n_valid = Int[],
        n_failed = Int[],
    )

    df_cov = DataFrame(
        prior = String[],
        method = String[],
        z0 = Float64[],
        coverage = Float64[],
        theta_true = Float64[],
        n_valid = Int[],
        n_failed = Int[],
    )

    for (midx, meth) in enumerate(methods)
        for (k, z0) in enumerate(z0_grid)
            lo = lower_mean[midx, k]
            hi = upper_mean[midx, k]
            len = hi - lo
            θ = theta_true[k]
            push!(df_ci, (spec.label, meth.name, z0, lo, hi, len, θ, valid_cnt[midx, k], fail_cnt[midx, k]))
            push!(df_cov, (spec.label, meth.name, z0, coverage[midx, k], θ, valid_cnt[midx, k], fail_cnt[midx, k]))
        end
    end

    ci_path = joinpath(outdir, "postmean_ci_$(spec.label).csv")
    cov_path = joinpath(outdir, "postmean_coverage_$(spec.label).csv")
    CSV.write(ci_path, df_ci)
    CSV.write(cov_path, df_cov)
    @info "Wrote CSV summaries" ci_path=ci_path cov_path=cov_path

    # ------------------------------------------------------------------
    # Plot 1: CI bands
    # ------------------------------------------------------------------
    p_band = plot(
        xlabel = "z",
        ylabel = "posterior mean θ(z)",
        title = "Posterior mean CI bands ($(spec.label)), n=$(n), reps=$(nreps)",
        legend = :topleft,
    )

    plot!(p_band, z0_grid, theta_true; label="True posterior mean", color=:black, lw=2)

    for (midx, meth) in enumerate(methods)
        center = (lower_mean[midx, :] .+ upper_mean[midx, :]) ./ 2
        ribbon = (upper_mean[midx, :] .- lower_mean[midx, :]) ./ 2
        plot!(p_band, z0_grid, center;
              ribbon=ribbon,
              label=meth.name,
              linestyle=meth.linestyle,
              lw=2,
              marker=:none)
    end

    band_path = joinpath(plots_dir, "postmean_ci_bands_$(lowercase(spec.label)).png")
    savefig(p_band, band_path)

    # ------------------------------------------------------------------
    # Plot 2: Coverage
    # ------------------------------------------------------------------
    p_cov = plot(
        xlabel = "z",
        ylabel = "coverage probability",
        title = "Pointwise coverage ($(spec.label)), nominal=$(1 - alpha)",
        legend = :bottomright,
        ylim = (0.0, 1.05),
    )
    hline!(p_cov, [1 - alpha]; label="Nominal $(1 - alpha)", color=:black, linestyle=:dash, lw=2)

    for (midx, meth) in enumerate(methods)
        plot!(p_cov, z0_grid, vec(coverage[midx, :]);
              label=meth.name,
              linestyle=meth.linestyle,
              lw=2,
              marker=:circle)
    end

    covfig_path = joinpath(plots_dir, "postmean_coverage_$(lowercase(spec.label)).png")
    savefig(p_cov, covfig_path)

    @info "Wrote plots" band_path=band_path covfig_path=covfig_path
    return nothing
end


function main(args=ARGS)
    d = parse_args(args)

    run_one_prior(
        prior = d["prior"],
        outdir = d["outdir"],
        n = d["n"],
        nreps = d["nreps"],
        alpha = d["alpha"],
        seed = d["seed"],
        zmin = d["zmin"],
        zmax = d["zmax"],
        zstep = d["zstep"],
        boot_B = d["boot_B"],
        clt_B = d["clt_B"],
        lp_time_limit = d["lp_time_limit"],
        gurobi_threads = d["gurobi_threads"],
        mu_min = d["mu_min"],
        mu_max = d["mu_max"],
        mu_points = d["mu_points"],
        t_min = d["t_min"],
        t_max = d["t_max"],
        t_points = d["t_points"],
        x_min = d["x_min"],
        x_max = d["x_max"],
        x_points = d["x_points"],
        gauss_bw = d["gauss_bw"],
        smooth_sigma = d["smooth_sigma"],
        quad_points = d["quad_points"],
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
