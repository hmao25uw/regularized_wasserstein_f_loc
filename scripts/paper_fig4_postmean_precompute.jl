#!/usr/bin/env julia
#
# Precompute (and cache to disk) all **data-dependent** quantities needed for the
# Figure-4 style posterior-mean simulation:
#   - empirical CDF / KDE grids
#   - all F-localization radii (DKW / Gauss bootstrap / W1 DKW / W1 bootstrap / W1 CLT / Smooth-W1 CLT)
#
# The goal is to enable a SLURM job-array workflow where the expensive LP solves
# for each z0-grid point can be run in parallel *without recomputing radii*.
#
# This script does NOT solve any LPs.
#
# Output (under --outdir):
#   precompute/postmean_meta.jls
#   precompute/postmean_stats_spiky.jls
#   precompute/postmean_stats_negspiky.jls
#
# Usage (local or SLURM):
#   julia --project=. scripts/paper_fig4_postmean_precompute.jl --outdir results/paper_fig4 --nreps 4000
#

using Random
using Statistics
using Printf
using Distributions
using Serialization

using NonparBayesCI


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

function parse_args(args)
    d = Dict{String,Any}(
        "prior" => "both",           # spiky | negspiky | both
        "outdir" => "results/paper_fig4",
        # NOTE:
        #   - The original Ignatiadis & Wager (2021) Figure-4 simulations use n = 5000.
        #   - We default to a smaller n for faster local iteration.
        #     To match the paper more closely, pass `--n 5000`.
        "n" => 500,
        "nreps" => 4000,
        "alpha" => 0.05,
        "seed" => 12345,
        "zmin" => -3.0,
        "zmax" => 3.0,
        "zstep" => 0.2,
        # Calibration settings
        "boot_B" => 500,
        "clt_B" => 2000,
        # SLURM parallelization:
        # We split the nreps replicates into blocks and let each job-array task
        # process one (prior, z0, block) triple.
        #
        # This value is only used to write metadata to disk (it does NOT affect
        # the computed radii / stats).
        "block_size" => 100,
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
        # (Ignored here) LP / solver knobs, accepted so the SLURM submit
        # helper can pass the same arg list to all stages.
        "lp_time_limit" => 600.0,
        "gurobi_threads" => 1,
        "gurobi_seed" => 1,
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
        elseif a == "--block_size"
            d["block_size"] = parse(Int, args[i+1]); i += 2
        elseif a == "--mu_min"
            d["mu_min"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--mu_max"
            d["mu_max"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--mu_points"
            d["mu_points"] = parse(Int, args[i+1]); i += 2
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
        w = [0.4, 0.2, 0.2, 0.2]
        m = [0.0, 0.0, 0.0, 0.0]
        s = [0.25, 0.5, 1.0, 2.0]
        return (w=w, m=m, s=s, label="Spiky", key="spiky")
    elseif name_lc in ("negspiky", "gnegspiky")
        w = [0.8, 0.2]
        m = [-0.25, 0.0]
        s = [0.25, 1.0]
        return (w=w, m=m, s=s, label="NegSpiky", key="negspiky")
    else
        error("Unknown prior=$(name). Use 'spiky' or 'negspiky'.")
    end
end

"""Sample μ ~ mixture of normals described by spec."""
function sample_mu(spec, n::Int, rng::AbstractRNG)
    w = Float64.(spec.w)
    m = Float64.(spec.m)
    s = Float64.(spec.s)
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
        dens = w[k] * pdf(Normal(m[k], sqrt(v)), z)
        mpost = m[k] + (τ2 / v) * (z - m[k])
        numer += dens * mpost
        denom += dens
    end
    return numer / denom
end


# -----------------------------------------------------------------------------
# Precompute
# -----------------------------------------------------------------------------

function precompute_one_prior(spec;
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

    σ = 1.0

    # Grids
    z0_grid = make_grid(zmin, zmax, zstep)
    mu_grid = collect(range(mu_min, mu_max, length=mu_points))
    t_grid = collect(range(t_min, t_max, length=t_points))
    x_grid = collect(range(x_min, x_max, length=x_points))

    # Localization specs (exactly as in scripts/paper_fig4_postmean.jl)
    loc_dkw = NonparBayesCI.DKWLocalization(alpha=alpha, t_grid=t_grid)
    loc_gauss = NonparBayesCI.GaussLocalization(alpha=alpha, x_grid=x_grid, bandwidth=gauss_bw, B=boot_B)
    loc_w_dkw = NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_grid, radius_method=:dkw,
                                                      support=(t_min, t_max), B=clt_B)
    loc_w_boot = NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_grid, radius_method=:bootstrap, B=boot_B)
    loc_w_clt = NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_grid, radius_method=:clt, B=clt_B)
    loc_smooth = NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_grid, radius_method=:clt, B=clt_B,
                                                       regularization=:smooth,
                                                       smooth_sigma=smooth_sigma,
                                                       kernel=:uniform,
                                                       quad_points=quad_points)

    # Use the package's preparation routines to avoid any mismatch.
    mt = length(Float64.(t_grid))
    mx = length(Float64.(x_grid))

    # DKW-F
    Fhat_dkw = Matrix{Float64}(undef, nreps, mt)
    eps_dkw = Vector{Float64}(undef, nreps)

    # Gauss-F
    fhat_gauss = Matrix{Float64}(undef, nreps, mx)
    c_gauss = Vector{Float64}(undef, nreps)

    # W1 (no regularization) — Fhat shared across the three radius methods
    Fhat_w = Matrix{Float64}(undef, nreps, mt)
    rho_w_dkw = Vector{Float64}(undef, nreps)
    rho_w_boot = Vector{Float64}(undef, nreps)
    rho_w_clt = Vector{Float64}(undef, nreps)

    # Smooth W1 (CLT)
    Fhat_smooth = Matrix{Float64}(undef, nreps, mt)
    rho_smooth = Vector{Float64}(undef, nreps)

    rng = MersenneTwister(seed)
    tick = max(1, Int(floor(nreps / 20)))

    @info "Precomputing localization stats" prior=spec.label n=n nreps=nreps alpha=alpha

    # We intentionally follow the SAME RNG consumption order as the sequential
    # script: simulate z, then prepare stats in the same method ordering.
    for rep in 1:nreps
        z = simulate_z(spec, n, σ, rng)

        st = NonparBayesCI.prepare_localization_stats(loc_dkw, z; rng=rng)
        @inbounds Fhat_dkw[rep, :] .= st.Fhat
        eps_dkw[rep] = st.eps

        st = NonparBayesCI.prepare_localization_stats(loc_gauss, z; rng=rng)
        @inbounds fhat_gauss[rep, :] .= st.fhat
        c_gauss[rep] = st.c

        st = NonparBayesCI.prepare_localization_stats(loc_w_dkw, z; rng=rng)
        @inbounds Fhat_w[rep, :] .= st.Fhat
        rho_w_dkw[rep] = st.rho

        st = NonparBayesCI.prepare_localization_stats(loc_w_boot, z; rng=rng)
        rho_w_boot[rep] = st.rho

        st = NonparBayesCI.prepare_localization_stats(loc_w_clt, z; rng=rng)
        rho_w_clt[rep] = st.rho

        st = NonparBayesCI.prepare_localization_stats(loc_smooth, z; rng=rng)
        @inbounds Fhat_smooth[rep, :] .= st.Fhat
        rho_smooth[rep] = st.rho

        if rep % tick == 0 || rep == 1 || rep == nreps
            @info("precompute progress", prior=spec.label, rep=rep, nreps=nreps)
        end
    end

    z0_grid = Float64.(z0_grid)
    theta_true = [oracle_postmean(z0, spec, σ) for z0 in z0_grid]

    # Constant grids/spacings (match how constraints discretize the integral).
    t_sorted = sort(Float64.(t_grid))
    Δt = diff(t_sorted)

    payload = (
        prior_key = spec.key,
        prior_label = spec.label,
        n = n,
        nreps = nreps,
        alpha = alpha,
        seed = seed,
        z0_grid = z0_grid,
        theta_true = Float64.(theta_true),
        mu_grid = Float64.(mu_grid),
        # grids for localization
        t_grid_dkw = Float64.(t_grid),
        t_grid_sorted = t_sorted,
        Δt = Float64.(Δt),
        x_grid = Float64.(x_grid),
        gauss_bw = gauss_bw,
        boot_B = boot_B,
        clt_B = clt_B,
        # smooth params
        smooth_sigma = smooth_sigma,
        quad_points = quad_points,
        kernel = :uniform,
        # stats
        dkw = (Fhat=Fhat_dkw, eps=eps_dkw),
        gauss = (fhat=fhat_gauss, c=c_gauss),
        w1 = (Fhat=Fhat_w, rho_dkw=rho_w_dkw, rho_boot=rho_w_boot, rho_clt=rho_w_clt),
        smooth_w1 = (Fhat=Fhat_smooth, rho_clt=rho_smooth),
        method_names = [
            "DKW-F",
            "Gauss-F",
            "W₁ (DKW)",
            "W₁ (boot)",
            "W₁ (CLT)",
            "Smooth-W₁ (CLT)",
        ],
    )

    pre_dir = ensure_dir(joinpath(outdir, "precompute"))
    outpath = joinpath(pre_dir, "postmean_stats_$(spec.key).jls")
    open(outpath, "w") do io
        serialize(io, payload)
    end
    @info "Wrote precompute cache" path=outpath
    return payload
end


function main(args=ARGS)
    d = parse_args(args)
    outdir = d["outdir"]
    ensure_dir(outdir)

    prior = d["prior"]
    priors = if prior == "both"
        ["spiky", "negspiky"]
    else
        [prior]
    end

    results = Vector{Any}()
    for pr in priors
        spec = prior_spec(pr)
        push!(results, precompute_one_prior(spec;
            outdir = outdir,
            n = d["n"],
            nreps = d["nreps"],
            alpha = d["alpha"],
            seed = d["seed"],
            zmin = d["zmin"],
            zmax = d["zmax"],
            zstep = d["zstep"],
            boot_B = d["boot_B"],
            clt_B = d["clt_B"],
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
        ))
    end

    # Meta file for job-array mapping and aggregation.
    # We take z0_grid/method_names from the first payload (they are identical
    # across priors by construction).
    first_payload = results[1]
    meta = (
        priors = [prior_spec(p).key for p in priors],
        z0_grid = first_payload.z0_grid,
        method_names = first_payload.method_names,
        alpha = first_payload.alpha,
        n = first_payload.n,
        nreps = first_payload.nreps,
        block_size = Int(d["block_size"]),
        nblocks = Int(cld(Int(first_payload.nreps), Int(d["block_size"]))),
    )
    pre_dir = ensure_dir(joinpath(outdir, "precompute"))
    meta_path = joinpath(pre_dir, "postmean_meta.jls")
    open(meta_path, "w") do io
        serialize(io, meta)
    end
    @info "Wrote meta" path=meta_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
