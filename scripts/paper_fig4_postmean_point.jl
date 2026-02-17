#!/usr/bin/env julia
#
# Compute posterior-mean F-localization CIs for **one** grid point z0.
#
# This script is designed for SLURM job arrays:
#   - A precompute stage caches all data-dependent localization stats.
#   - Each array task solves LPs for one (prior, z0) pair.
#   - A final aggregation stage combines partial results and plots.
#
# Inputs:
#   Requires precompute files created by `paper_fig4_postmean_precompute.jl`.
#
# Output (under --outdir):
#   partial/postmean_point_<prior>_zidx<k>.csv
#
# Usage (manual):
#   julia --project=. scripts/paper_fig4_postmean_point.jl --outdir results/paper_fig4 --prior spiky --zindex 1
#
# Usage (SLURM array):
#   (read SLURM_ARRAY_TASK_ID automatically)
#   julia --project=. scripts/paper_fig4_postmean_point.jl --outdir results/paper_fig4
#

using Random
using Statistics
using Printf
using Distributions
using Serialization

using CSV
using DataFrames

using NonparBayesCI


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

function parse_args(args)
    d = Dict{String,Any}(
        "outdir" => "results/paper_fig4",
        "taskid" => nothing,         # optional; otherwise reads SLURM_ARRAY_TASK_ID
        "prior" => nothing,          # optional override for manual runs
        "zindex" => nothing,         # 1-based index into z0_grid
        # (Optional sanity checks; values are read from the precompute cache)
        "n" => nothing,
        "nreps" => nothing,
        "alpha" => nothing,
        "seed" => nothing,
        "zmin" => nothing,
        "zmax" => nothing,
        "zstep" => nothing,
        "boot_B" => nothing,
        "clt_B" => nothing,
        "gauss_bw" => nothing,
        "smooth_sigma" => nothing,
        "quad_points" => nothing,
        # Optimization settings (must match precompute defaults unless you know what you're doing)
        "lp_time_limit" => 600.0,
        "gurobi_threads" => 1,
        "gurobi_seed" => 1,
    )

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--outdir"
            d["outdir"] = String(args[i+1]); i += 2
        elseif a == "--taskid"
            d["taskid"] = parse(Int, args[i+1]); i += 2
        elseif a == "--prior"
            d["prior"] = lowercase(String(args[i+1])); i += 2
        elseif a == "--zindex"
            d["zindex"] = parse(Int, args[i+1]); i += 2
        elseif a == "--lp_time_limit"
            d["lp_time_limit"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--gurobi_threads"
            d["gurobi_threads"] = parse(Int, args[i+1]); i += 2
        elseif a == "--gurobi_seed"
            d["gurobi_seed"] = parse(Int, args[i+1]); i += 2
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
        elseif a == "--gauss_bw"
            d["gauss_bw"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--smooth_sigma"
            d["smooth_sigma"] = parse(Float64, args[i+1]); i += 2
        elseif a == "--quad_points"
            d["quad_points"] = parse(Int, args[i+1]); i += 2
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


# -----------------------------------------------------------------------------
# Helper: build caches (likelihood matrices) once per task
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
# Main
# -----------------------------------------------------------------------------

function main(args=ARGS)
    d = parse_args(args)
    outdir = d["outdir"]
    pre_dir = joinpath(outdir, "precompute")
    meta_path = joinpath(pre_dir, "postmean_meta.jls")
    isfile(meta_path) || error("Missing meta file: $meta_path. Run paper_fig4_postmean_precompute.jl first.")

    meta = open(meta_path, "r") do io
        deserialize(io)
    end

    priors = Vector{String}(meta.priors)
    z0_grid = Vector{Float64}(meta.z0_grid)
    K = length(z0_grid)
    K > 0 || error("meta.z0_grid is empty")

    # Resolve which (prior, zindex) this task should run.
    prior = d["prior"]
    zindex = d["zindex"]

    if prior !== nothing || zindex !== nothing
        # Manual mode: require both.
        prior === nothing && error("Manual mode requires --prior")
        zindex === nothing && error("Manual mode requires --zindex")
        prior in priors || error("Unknown prior=$prior. Expected one of $(priors)")
        (1 <= zindex <= K) || error("zindex must be in 1:$K")
    else
        # SLURM array mode: use taskid or SLURM_ARRAY_TASK_ID.
        taskid = d["taskid"]
        if taskid === nothing
            haskey(ENV, "SLURM_ARRAY_TASK_ID") || error("No --taskid and SLURM_ARRAY_TASK_ID is not set")
            taskid = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
        end
        taskid >= 1 || error("taskid must be >= 1")
        ntasks = length(priors) * K
        taskid <= ntasks || error("taskid=$taskid exceeds ntasks=$ntasks")

        idx0 = taskid - 1
        prior_idx = div(idx0, K) + 1
        zindex = mod(idx0, K) + 1
        prior = priors[prior_idx]
    end

    z0 = z0_grid[zindex]

    # Load precomputed stats for this prior.
    cache_path = joinpath(pre_dir, "postmean_stats_$(prior).jls")
    isfile(cache_path) || error("Missing precompute cache: $cache_path")
    pre = open(cache_path, "r") do io
        deserialize(io)
    end

    # Optional sanity checks (useful when passing a shared arg list via the
    # SLURM submit helper).
    if d["n"] !== nothing && Int(d["n"]) != Int(pre.n)
        error("Arg mismatch: --n=$(d[\"n\"]) but precompute used n=$(pre.n)")
    end
    if d["nreps"] !== nothing && Int(d["nreps"]) != Int(pre.nreps)
        error("Arg mismatch: --nreps=$(d[\"nreps\"]) but precompute used nreps=$(pre.nreps)")
    end
    if d["alpha"] !== nothing && abs(Float64(d["alpha"]) - Float64(pre.alpha)) > 1e-12
        error("Arg mismatch: --alpha=$(d[\"alpha\"]) but precompute used alpha=$(pre.alpha)")
    end
    if d["seed"] !== nothing && Int(d["seed"]) != Int(pre.seed)
        error("Arg mismatch: --seed=$(d[\"seed\"]) but precompute used seed=$(pre.seed)")
    end
    if d["boot_B"] !== nothing && Int(d["boot_B"]) != Int(pre.boot_B)
        error("Arg mismatch: --boot_B=$(d[\"boot_B\"]) but precompute used boot_B=$(pre.boot_B)")
    end
    if d["clt_B"] !== nothing && Int(d["clt_B"]) != Int(pre.clt_B)
        error("Arg mismatch: --clt_B=$(d[\"clt_B\"]) but precompute used clt_B=$(pre.clt_B)")
    end
    if d["gauss_bw"] !== nothing && abs(Float64(d["gauss_bw"]) - Float64(pre.gauss_bw)) > 1e-12
        error("Arg mismatch: --gauss_bw=$(d[\"gauss_bw\"]) but precompute used gauss_bw=$(pre.gauss_bw)")
    end
    if d["smooth_sigma"] !== nothing && abs(Float64(d["smooth_sigma"]) - Float64(pre.smooth_sigma)) > 1e-12
        error("Arg mismatch: --smooth_sigma=$(d[\"smooth_sigma\"]) but precompute used smooth_sigma=$(pre.smooth_sigma)")
    end
    if d["quad_points"] !== nothing && Int(d["quad_points"]) != Int(pre.quad_points)
        error("Arg mismatch: --quad_points=$(d[\"quad_points\"]) but precompute used quad_points=$(pre.quad_points)")
    end

    @info "Point job starting" prior=prior zindex=zindex z0=z0 nreps=pre.nreps

    # EB setup (same as sequential script)
    σ = 1.0
    lik = NonparBayesCI.gaussian_likelihood(σ)
    mu_grid = Vector{Float64}(pre.mu_grid)
    problem = NonparBayesCI.EBProblem(mu_grid; pdf=lik.pdf, cdf=lik.cdf, h=μ -> μ)

    # Solver config
    solver = NonparBayesCI.SolverConfig(
        silent=true,
        time_limit_sec=Float64(d["lp_time_limit"]),
        gurobi_threads=Int(d["gurobi_threads"]),
        gurobi_seed=Int(d["gurobi_seed"]),
    )

    # Reconstruct localization methods (same ordering/names as sequential).
    t_grid = Vector{Float64}(pre.t_grid_dkw)
    t_sorted = Vector{Float64}(pre.t_grid_sorted)
    Δt = Vector{Float64}(pre.Δt)
    x_grid = Vector{Float64}(pre.x_grid)

    alpha = Float64(pre.alpha)

    methods = [
        (name="DKW-F",           loc=NonparBayesCI.DKWLocalization(alpha=alpha, t_grid=t_grid),
                                 linestyle=:dot,   kind=:orig),
        (name="Gauss-F",         loc=NonparBayesCI.GaussLocalization(alpha=alpha, x_grid=x_grid, bandwidth=Float64(pre.gauss_bw), B=Int(pre.boot_B)),
                                 linestyle=:dot,   kind=:orig),
        (name="W₁ (DKW)",        loc=NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_sorted, radius_method=:dkw,
                                                                           support=(Float64(first(t_grid)), Float64(last(t_grid))), B=Int(pre.clt_B)),
                                 linestyle=:solid, kind=:new),
        (name="W₁ (boot)",       loc=NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_sorted, radius_method=:bootstrap, B=Int(pre.boot_B)),
                                 linestyle=:solid, kind=:new),
        (name="W₁ (CLT)",        loc=NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_sorted, radius_method=:clt, B=Int(pre.clt_B)),
                                 linestyle=:solid, kind=:new),
        (name="Smooth-W₁ (CLT)", loc=NonparBayesCI.WassersteinLocalization(alpha=alpha, t_grid=t_sorted, radius_method=:clt, B=Int(pre.clt_B),
                                                                           regularization=:smooth,
                                                                           smooth_sigma=Float64(pre.smooth_sigma),
                                                                           kernel=Symbol(pre.kernel),
                                                                           quad_points=Int(pre.quad_points)),
                                 linestyle=:solid, kind=:new),
    ]

    # Likelihood matrix caches
    caches = Dict{String,NamedTuple}()
    caches["DKW-F"] = (Ccdf=cdf_mat(lik.cdf, t_grid, mu_grid),)
    caches["W₁ (DKW)"] = (Ccdf=cdf_mat(lik.cdf, t_sorted, mu_grid),)
    caches["W₁ (boot)"] = (Ccdf=cdf_mat(lik.cdf, t_sorted, mu_grid),)
    caches["W₁ (CLT)"] = (Ccdf=cdf_mat(lik.cdf, t_sorted, mu_grid),)
    caches["Gauss-F"] = (PdfMat=pdf_mat(lik.pdf, x_grid, mu_grid),)
    caches["Smooth-W₁ (CLT)"] = (Ccdf=NonparBayesCI.smooth_cdf_mat(lik.cdf, t_sorted, mu_grid;
                                                                    sigma=Float64(pre.smooth_sigma),
                                                                    kernel=Symbol(pre.kernel),
                                                                    quad_points=Int(pre.quad_points)),)

    # Accumulators over replicates
    #
    # IMPORTANT:
    # We *skip* failed solves when computing averages/coverage. This mirrors the
    # authors' plotting code, which drops repetitions where the solver returned
    # an Exception/NaNs (see their `simulation_plots.jl`).
    M = length(methods)
    lower_sum = zeros(Float64, M)
    upper_sum = zeros(Float64, M)
    cover_cnt = zeros(Int, M)
    valid_cnt = zeros(Int, M)
    fail_cnt = zeros(Int, M)

    θ_true = Float64(pre.theta_true[zindex])
    nreps = Int(pre.nreps)
    z_dummy = Float64[]

    # Pull out precomputed matrices for speed
    Fhat_dkw = pre.dkw.Fhat
    eps_dkw = pre.dkw.eps
    fhat_gauss = pre.gauss.fhat
    c_gauss = pre.gauss.c
    Fhat_w = pre.w1.Fhat
    rho_w_dkw = pre.w1.rho_dkw
    rho_w_boot = pre.w1.rho_boot
    rho_w_clt = pre.w1.rho_clt
    Fhat_smooth = pre.smooth_w1.Fhat
    rho_smooth = pre.smooth_w1.rho_clt

    # One safe solve (returns `nothing` on failure)
    function solve_one(meth, stats)::Union{Nothing,Tuple{Float64,Float64}}
        try
            lo, hi = NonparBayesCI.f_localization_ci(z_dummy, z0, problem, meth.loc;
                                                    solver=solver,
                                                    stats_override=stats,
                                                    mat_cache=caches[meth.name])
            if !(isfinite(lo) && isfinite(hi))
                return nothing
            end
            return (Float64(lo), Float64(hi))
        catch
            return nothing
        end
    end

    tick = max(1, Int(floor(nreps / 10)))
    for rep in 1:nreps
        # Build per-method stats for this replicate from cached arrays
        stats_list = NamedTuple[
            (t_grid=t_grid,    Fhat=view(Fhat_dkw, rep, :),    eps=eps_dkw[rep]),
            (x_grid=x_grid,    fhat=view(fhat_gauss, rep, :),  c=c_gauss[rep]),
            (t_grid=t_sorted,  Fhat=view(Fhat_w, rep, :),      rho=rho_w_dkw[rep],  Δt=Δt),
            (t_grid=t_sorted,  Fhat=view(Fhat_w, rep, :),      rho=rho_w_boot[rep], Δt=Δt),
            (t_grid=t_sorted,  Fhat=view(Fhat_w, rep, :),      rho=rho_w_clt[rep],  Δt=Δt),
            (t_grid=t_sorted,  Fhat=view(Fhat_smooth, rep, :), rho=rho_smooth[rep], Δt=Δt),
        ]

        for (midx, meth) in enumerate(methods)
            out = solve_one(meth, stats_list[midx])
            if out === nothing
                fail_cnt[midx] += 1
                continue
            end
            lo, hi = out
            lower_sum[midx] += lo
            upper_sum[midx] += hi
            valid_cnt[midx] += 1
            if (lo - 1e-8) <= θ_true <= (hi + 1e-8)
                cover_cnt[midx] += 1
            end
        end

        if rep % tick == 0 || rep == 1 || rep == nreps
            @info("point progress", prior=prior, z0=z0, rep=rep, nreps=nreps)
        end
    end

    lower_mean = fill(NaN, M)
    upper_mean = fill(NaN, M)
    coverage = fill(NaN, M)
    for midx in 1:M
        if valid_cnt[midx] > 0
            lower_mean[midx] = lower_sum[midx] / valid_cnt[midx]
            upper_mean[midx] = upper_sum[midx] / valid_cnt[midx]
            coverage[midx] = cover_cnt[midx] / valid_cnt[midx]
        end
    end

    if any(fail_cnt .> 0)
        @warn "Some CI solves failed and were skipped in averages" prior=prior z0=z0 fail_cnt=fail_cnt valid_cnt=valid_cnt
    end

    df = DataFrame(
        prior = fill(String(pre.prior_label), M),
        prior_key = fill(String(pre.prior_key), M),
        method = [m.name for m in methods],
        z0 = fill(z0, M),
        lower = vec(lower_mean),
        upper = vec(upper_mean),
        coverage = vec(coverage),
        theta_true = fill(θ_true, M),
        n_valid = valid_cnt,
        n_failed = fail_cnt,
    )

    partial_dir = ensure_dir(joinpath(outdir, "partial"))
    outpath = joinpath(partial_dir, @sprintf("postmean_point_%s_zidx%03d.csv", String(pre.prior_key), zindex))
    CSV.write(outpath, df)
    @info "Wrote partial" path=outpath
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
