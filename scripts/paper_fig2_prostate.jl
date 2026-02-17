#!/usr/bin/env julia
#
# Real-data experiment: Prostate z-scores (Ignatiadis & Wager, Section 5.2 / Figure 2)
#
# What this script does
#   1) Loads the 6033 prostate z-scores (downloads `prostz.txt` if missing).
#   2) Plots the empirical CDF with a DKW band.
#   3) Computes 95% F-localization confidence bands for the *posterior mean*
#      θ(z) = E[μ | Z=z] under the Gaussian location-mixture class
#         G ∈ LN(τ^2, K)
#      used in the paper (default τ = 0.25, K = [-3,3] discretized at step 0.05).
#
# IMPORTANT implementation detail
#   In the LN(τ^2,K) class, μ is modeled as μ = U + τ·ε with ε~N(0,1) and U~Π on K,
#   and Z|μ ~ N(μ, σ^2) (σ=1 here).
#
#   Integrating out μ gives: Z|U=u ~ N(u, σ_Z^2) with σ_Z^2 = σ^2 + τ^2.
#
#   The posterior mean of μ satisfies the identity
#      E[μ | Z=z] = (τ^2/(τ^2+σ^2)) z + (σ^2/(τ^2+σ^2)) E[U | Z=z].
#
#   Therefore we compute CIs for E[U|Z=z] with our standard EBProblem (h(u)=u)
#   and then apply the above affine transformation to obtain CIs for E[μ|Z=z].
#
# Outputs (under --outdir)
#   - data/prostate_ci.csv              (CI endpoints for each method on the z-grid)
#   - plots/prostate_empirical_cdf.png  (empirical CDF + DKW band)
#   - plots/prostate_postmean_ci.png    (posterior-mean CI bands)
#   - plots/prostate_fig2_like.png      (2-panel combined figure)
#

using Random
using Statistics
using Printf
using Distributions
using Downloads

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
        "outdir" => "results/paper_fig2_prostate",
        "alpha" => 0.05,
        "seed" => 12345,

        # z-grid for the posterior-mean curve
        "zmin" => -3.0,
        "zmax" => 3.0,
        "zstep" => 0.2,

        # LN(τ^2,K) class parameters (paper uses τ=0.25 and K=[-3,3] step 0.05)
        "tau" => 0.25,
        "sigma" => 1.0,
        "u_min" => -3.0,
        "u_max" => 3.0,
        "u_step" => 0.05,

        # Localization grids
        "t_min" => -6.0,
        "t_max" => 6.0,
        "t_points" => 301,
        "x_min" => -6.0,
        "x_max" => 6.0,
        "x_points" => 301,

        # Calibration budgets
        "boot_B" => 500,
        "clt_B" => 2000,

        # Gauss KDE bandwidth
        "gauss_bw" => 0.3,

        # Smooth Wasserstein parameters
        "smooth_sigma" => 0.25,
        "quad_points" => 33,

        # LP settings
        "lp_time_limit" => 600.0,
        "gurobi_threads" => 1,
        "gurobi_seed" => 1,

        # Data path (optional)
        "data_path" => "",
        "download" => true,
    )

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--outdir"
            d["outdir"] = String(args[i+1]); i += 2
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
        elseif a == "--tau"
            d["tau"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--sigma"
            d["sigma"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--u_min"
            d["u_min"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--u_max"
            d["u_max"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--u_step"
            d["u_step"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--t_min"
            d["t_min"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--t_max"
            d["t_max"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--t_points"
            d["t_points"] = parse(Int, args[i+1]); i += 2
        elseif a == "--x_min"
            d["x_min"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--x_max"
            d["x_max"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--x_points"
            d["x_points"] = parse(Int, args[i+1]); i += 2
        elseif a == "--boot_B"
            d["boot_B"] = parse(Int, args[i+1]); i += 2
        elseif a == "--clt_B"
            d["clt_B"] = parse(Int, args[i+1]); i += 2
        elseif a == "--gauss_bw"
            d["gauss_bw"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--smooth_sigma"
            d["smooth_sigma"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--quad_points"
            d["quad_points"] = parse(Int, args[i+1]); i += 2
        elseif a == "--lp_time_limit"
            d["lp_time_limit"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--gurobi_threads"
            d["gurobi_threads"] = parse(Int, args[i+1]); i += 2
        elseif a == "--gurobi_seed"
            d["gurobi_seed"] = parse(Int, args[i+1]); i += 2
        elseif a == "--data_path"
            d["data_path"] = String(args[i+1]); i += 2
        elseif a == "--no_download"
            d["download"] = false; i += 1
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
# Data
# -----------------------------------------------------------------------------

"""Parse a file containing a single numeric column (robust to commas/whitespace)."""
function read_numeric_vector(path::AbstractString)::Vector{Float64}
    txt = read(path, String)
    out = Float64[]
    for tok in split(txt, r"[,\s]+")
        isempty(tok) && continue
        push!(out, parse(Float64, tok))
    end
    return out
end

"""Ensure `prostz.txt` exists locally; if missing, optionally download it."""
function ensure_prostz(; path::AbstractString, download::Bool=true)
    if isfile(path)
        return path
    end
    download || error("Missing prostate data file at '$path' and --no_download was set.\n" *
                      "Download prostz.txt and place it at data/prostate/prostz.txt (see data/prostate/README.md).")

    ensure_dir(dirname(path))

    # Primary source (CASI / Hastie)
    url1 = "https://hastie.su.domains/CASI_files/DATA/prostz.txt"
    # Backup mirror used in some course notes
    url2 = "https://faculty.washington.edu/kenrice/sisgbayes/prostz.txt"

    @info "Downloading prostate z-scores (prostz.txt)" dest=path url=url1
    try
        Downloads.download(url1, path)
    catch err
        @warn "Primary download failed; trying backup mirror" url=url2 err=err
        Downloads.download(url2, path)
    end

    return path
end


# -----------------------------------------------------------------------------
# Model matrices (cache)
# -----------------------------------------------------------------------------

function cdf_mat(cdf_fun::Function, t_grid::Vector{Float64}, u_grid::Vector{Float64})
    m = length(t_grid)
    p = length(u_grid)
    out = Matrix{Float64}(undef, m, p)
    @inbounds for i in 1:m
        t = t_grid[i]
        for j in 1:p
            out[i, j] = Float64(cdf_fun(t, u_grid[j]))
        end
    end
    return out
end

function pdf_mat(pdf_fun::Function, x_grid::Vector{Float64}, u_grid::Vector{Float64})
    m = length(x_grid)
    p = length(u_grid)
    out = Matrix{Float64}(undef, m, p)
    @inbounds for i in 1:m
        x = x_grid[i]
        for j in 1:p
            out[i, j] = Float64(pdf_fun(x, u_grid[j]))
        end
    end
    return out
end


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

function main(args::Vector{String})
    par = parse_args(args)

    outdir = String(par["outdir"])
    ensure_dir(outdir)
    plots_dir = ensure_dir(joinpath(outdir, "plots"))

    alpha = Float64(par["alpha"])
    seed  = Int(par["seed"])

    # Data location
    data_path = String(par["data_path"])
    if isempty(data_path)
        data_path = joinpath(@__DIR__, "..", "data", "prostate", "prostz.txt")
        data_path = normpath(data_path)
    end
    ensure_prostz(path=data_path, download=Bool(par["download"]))

    z = read_numeric_vector(data_path)
    n = length(z)
    if n != 6033
        @warn "Expected 6033 prostate z-scores; got $n. Proceeding anyway." file=data_path
    end

    @info "Loaded prostate z-scores" n=n file=data_path min=minimum(z) max=maximum(z) mean=mean(z) sd=std(z)

    # ------------------------------------------------------------------
    # Plot: empirical CDF + DKW band
    # ------------------------------------------------------------------

    t_plot = collect(range(minimum(z) - 0.5, maximum(z) + 0.5, length=500))
    Fhat_plot = NonparBayesCI.empirical_cdf(z, t_plot)
    eps = NonparBayesCI.dkw_epsilon(alpha, n)
    lower_band = clamp.(Fhat_plot .- eps, 0.0, 1.0)
    upper_band = clamp.(Fhat_plot .+ eps, 0.0, 1.0)

    p_cdf = plot(t_plot, Fhat_plot;
                 label="Empirical CDF",
                 xlabel="t",
                 ylabel="F̂(t)",
                 title="Prostate z-scores: empirical CDF with DKW band (1-α=$(1-alpha))")
    plot!(t_plot, lower_band; label="DKW band", linestyle=:dash)
    plot!(t_plot, upper_band; label="", linestyle=:dash)
    savefig(p_cdf, joinpath(plots_dir, "prostate_empirical_cdf.png"))

    # ------------------------------------------------------------------
    # Posterior-mean CI under LN(τ^2,K)
    # ------------------------------------------------------------------

    z0_grid = make_grid(par["zmin"], par["zmax"], par["zstep"])

    τ = Float64(par["tau"])
    σ = Float64(par["sigma"])
    denom = τ^2 + σ^2
    a_of_z(z0) = (τ^2 / denom) * Float64(z0)
    b = σ^2 / denom

    u_grid = collect(Float64(par["u_min"]):Float64(par["u_step"]):Float64(par["u_max"]))
    σZ = sqrt(σ^2 + τ^2)
    lik = NonparBayesCI.gaussian_likelihood(σZ)
    problem_u = NonparBayesCI.EBProblem(u_grid; pdf=lik.pdf, cdf=lik.cdf, h=u -> u)

    t_grid = collect(range(Float64(par["t_min"]), Float64(par["t_max"]), length=Int(par["t_points"])))
    x_grid = collect(range(Float64(par["x_min"]), Float64(par["x_max"]), length=Int(par["x_points"])))

    boot_B = Int(par["boot_B"])
    clt_B  = Int(par["clt_B"])
    gauss_bw = Float64(par["gauss_bw"])
    smooth_sigma = Float64(par["smooth_sigma"])
    quad_points = Int(par["quad_points"])

    solver = NonparBayesCI.SolverConfig(
        silent=true,
        time_limit_sec=Float64(par["lp_time_limit"]),
        gurobi_threads=Int(par["gurobi_threads"]),
        gurobi_seed=Int(par["gurobi_seed"]),
    )

    methods = [
        (name="DKW-F-Loc",       loc=NonparBayesCI.DKWLocalization(alpha=alpha, t_grid=t_grid),
                                 linestyle=:dot,   kind=:orig),
        (name="Gauss-F-Loc",     loc=NonparBayesCI.GaussLocalization(alpha=alpha, x_grid=x_grid, bandwidth=gauss_bw, B=boot_B),
                                 linestyle=:dot,   kind=:orig),
        (name="Wasserstein_DKW", loc=NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_grid, radius_method=:dkw,
                                                                           support=(Float64(par["t_min"]), Float64(par["t_max"])),
                                                                           B=clt_B),
                                 linestyle=:solid, kind=:new),
        (name="Wasserstein_bootstrap", loc=NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_grid, radius_method=:bootstrap, B=boot_B),
                                 linestyle=:solid, kind=:new),
        (name="Wasserstein_clt", loc=NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_grid, radius_method=:clt, B=clt_B),
                                 linestyle=:solid, kind=:new),
        (name="Smoothed_Wasserstein_clt", loc=NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_grid, radius_method=:clt, B=clt_B,
                                                                           regularization=:smooth,
                                                                           smooth_sigma=smooth_sigma,
                                                                           kernel=:uniform,
                                                                           quad_points=quad_points),
                                 linestyle=:solid, kind=:new),
    ]

    # Cache model matrices per method (big speedup)
    caches = Dict{String,NamedTuple}()
    caches["DKW-F-Loc"] = (Ccdf=cdf_mat(lik.cdf, Float64.(t_grid), u_grid),)
    caches["Wasserstein_DKW"] = (Ccdf=cdf_mat(lik.cdf, sort(t_grid), u_grid),)
    caches["Wasserstein_bootstrap"] = (Ccdf=cdf_mat(lik.cdf, sort(t_grid), u_grid),)
    caches["Wasserstein_clt"] = (Ccdf=cdf_mat(lik.cdf, sort(t_grid), u_grid),)
    caches["Gauss-F-Loc"] = (PdfMat=pdf_mat(lik.pdf, x_grid, u_grid),)
    caches["Smoothed_Wasserstein_clt"] = (Ccdf=NonparBayesCI.smooth_cdf_mat(lik.cdf, sort(t_grid), u_grid;
                                                                    sigma=smooth_sigma,
                                                                    kernel=:uniform,
                                                                    quad_points=quad_points),)

    rng = MersenneTwister(seed)
    M = length(methods)
    K = length(z0_grid)
    lower = zeros(Float64, M, K)
    upper = zeros(Float64, M, K)

    @info "Computing prostate posterior-mean CI bands" alpha=alpha tau=τ sigma=σ sigmaZ=σZ zgrid="[$(par["zmin"]):$(par["zstep"]):$(par["zmax"])]" n=n

    for (midx, m) in enumerate(methods)
        name = m.name
        loc  = m.loc

        # Compute stats once so lower/upper share the same localization set.
        stats = NonparBayesCI.prepare_localization_stats(loc, z; rng=rng)

        for (k, z0) in enumerate(z0_grid)
            lo_u, hi_u = NonparBayesCI.f_localization_ci(z, z0, problem_u, loc;
                                                        solver=solver,
                                                        rng=rng,
                                                        stats_override=stats,
                                                        mat_cache=caches[name])
            lo_mu = a_of_z(z0) + b * lo_u
            hi_mu = a_of_z(z0) + b * hi_u
            lower[midx, k] = lo_mu
            upper[midx, k] = hi_mu
        end
    end

    # Save CSV
    df = DataFrame(z0 = z0_grid)
    for (midx, m) in enumerate(methods)
        nm = replace(m.name, "₁" => "1")
        nm = replace(nm, r"[^A-Za-z0-9]+" => "_")
        nm = replace(nm, r"^_+|_+$" => "")
        df[!, Symbol("lower_" * nm)] = vec(lower[midx, :])
        df[!, Symbol("upper_" * nm)] = vec(upper[midx, :])
    end
    CSV.write(joinpath(outdir, "prostate_ci.csv"), df)

    # Plot posterior-mean CI bands
    p_ci = plot(; xlabel="z", ylabel="E[μ | Z=z]", title="Prostate: posterior-mean CIs (1-α=$(1-alpha))")
    for (midx, m) in enumerate(methods)
        mid = (lower[midx, :] .+ upper[midx, :]) ./ 2
        ribbon_low  = mid .- lower[midx, :]
        ribbon_high = upper[midx, :] .- mid
        plot!(p_ci, z0_grid, mid;
              ribbon=(ribbon_low, ribbon_high),
              label=m.name,
              linestyle=m.linestyle)
    end
    savefig(p_ci, joinpath(plots_dir, "prostate_postmean_ci.png"))

    # Combined 2-panel figure (Figure-2-like)
    p_combo = plot(p_cdf, p_ci; layout=(2,1), size=(900,900))
    savefig(p_combo, joinpath(plots_dir, "prostate_fig2_like.png"))

    @info "Done." outdir=outdir
    return nothing
end


if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
