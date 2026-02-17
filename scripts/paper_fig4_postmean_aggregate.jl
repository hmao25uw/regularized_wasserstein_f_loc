#!/usr/bin/env julia
#
# Aggregate the per-z0 partial results produced by `paper_fig4_postmean_point.jl`
# into the same CSV outputs and figures as the sequential script
# `paper_fig4_postmean.jl`.
#
# Output (under --outdir):
#   postmean_ci_Spiky.csv
#   postmean_coverage_Spiky.csv
#   postmean_ci_NegSpiky.csv
#   postmean_coverage_NegSpiky.csv
#   plots/postmean_ci_bands_spiky.png
#   plots/postmean_ci_bands_negspiky.png
#   plots/postmean_coverage_spiky.png
#   plots/postmean_coverage_negspiky.png
#
# Usage:
#   julia --project=. scripts/paper_fig4_postmean_aggregate.jl --outdir results/paper_fig4

using Printf
using Serialization

using CSV
using DataFrames

# Make Plots work on headless machines (e.g., SLURM compute nodes)
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

function parse_args(args)
    d = Dict{String,Any}(
        "outdir" => "results/paper_fig4",
        # Accept (but ignore) the same args as precompute/point scripts so a
        # shared arg list can be passed to all pipeline stages.
        "prior" => nothing,
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
        "lp_time_limit" => nothing,
        "gurobi_threads" => nothing,
        "gurobi_seed" => nothing,
    )
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--outdir"
            d["outdir"] = String(args[i+1]); i += 2
        elseif a in ("--prior", "--n", "--nreps", "--alpha", "--seed",
                     "--zmin", "--zmax", "--zstep", "--boot_B", "--clt_B",
                     "--gauss_bw", "--smooth_sigma", "--quad_points",
                     "--lp_time_limit", "--gurobi_threads", "--gurobi_seed")
            # Ignore (advance by 2 for flags that expect a value)
            i += 2
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
# Plotting helpers (match scripts/paper_fig4_postmean.jl)
# -----------------------------------------------------------------------------

function method_style(method::AbstractString)
    # Dotted for original Ignatiadis & Wager methods; solid for our new ones.
    if method in ("DKW-F", "Gauss-F")
        return :dot
    else
        return :solid
    end
end


function aggregate_one_prior(outdir::String, prior_key::String)
    pre_dir = joinpath(outdir, "precompute")
    stats_path = joinpath(pre_dir, "postmean_stats_$(prior_key).jls")
    isfile(stats_path) || error("Missing precompute cache: $stats_path")
    pre = open(stats_path, "r") do io
        deserialize(io)
    end

    z0_grid = Vector{Float64}(pre.z0_grid)
    K = length(z0_grid)
    methods = Vector{String}(pre.method_names)

    partial_dir = joinpath(outdir, "partial")
    isdir(partial_dir) || error("Missing partial directory: $partial_dir")

    # Read partials (one file per z-index)
    df_all = DataFrame()
    for k in 1:K
        path = joinpath(partial_dir, @sprintf("postmean_point_%s_zidx%03d.csv", prior_key, k))
        isfile(path) || error("Missing partial file: $path")
        dfk = CSV.read(path, DataFrame)
        df_all = isempty(df_all) ? dfk : vcat(df_all, dfk)
    end

    # Ensure sorting by z0 then method
    sort!(df_all, [:z0, :method])

    # Construct the same CSV outputs as the sequential script.
    df_ci = DataFrame(
        prior = String[],
        method = String[],
        z0 = Float64[],
        lower = Float64[],
        upper = Float64[],
        length = Float64[],
        theta_true = Float64[],
    )

    df_cov = DataFrame(
        prior = String[],
        method = String[],
        z0 = Float64[],
        coverage = Float64[],
        theta_true = Float64[],
    )

    prior_label = String(pre.prior_label)
    for row in eachrow(df_all)
        lo = Float64(row.lower)
        hi = Float64(row.upper)
        z0 = Float64(row.z0)
        θ = Float64(row.theta_true)
        push!(df_ci, (prior_label, String(row.method), z0, lo, hi, hi - lo, θ))
        push!(df_cov, (prior_label, String(row.method), z0, Float64(row.coverage), θ))
    end

    ci_path = joinpath(outdir, "postmean_ci_$(prior_label).csv")
    cov_path = joinpath(outdir, "postmean_coverage_$(prior_label).csv")
    CSV.write(ci_path, df_ci)
    CSV.write(cov_path, df_cov)
    @info "Wrote aggregated CSVs" ci_path=ci_path cov_path=cov_path

    plots_dir = ensure_dir(joinpath(outdir, "plots"))

    # Plot 1: CI bands
    theta_true = Vector{Float64}(pre.theta_true)

    p_band = plot(
        xlabel = "z",
        ylabel = "posterior mean θ(z)",
        title = "Posterior mean CI bands ($(prior_label)), n=$(pre.n), reps=$(pre.nreps)",
        legend = :topleft,
    )
    plot!(p_band, z0_grid, theta_true; label="True posterior mean", color=:black, lw=2)

    for meth in methods
        dfi = filter(:method => ==(meth), df_all)
        sort!(dfi, :z0)
        center = (dfi.lower .+ dfi.upper) ./ 2
        ribbon = (dfi.upper .- dfi.lower) ./ 2
        plot!(p_band, dfi.z0, center;
              ribbon=ribbon,
              label=meth,
              linestyle=method_style(meth),
              lw=2,
              marker=:none)
    end

    band_path = joinpath(plots_dir, "postmean_ci_bands_$(prior_key).png")
    savefig(p_band, band_path)

    # Plot 2: Coverage
    p_cov = plot(
        xlabel = "z",
        ylabel = "coverage probability",
        title = "Pointwise coverage ($(prior_label)), nominal=$(1 - pre.alpha)",
        legend = :bottomright,
        ylim = (0.0, 1.05),
    )
    hline!(p_cov, [1 - pre.alpha]; label="Nominal $(1 - pre.alpha)", color=:black, linestyle=:dash, lw=2)

    for meth in methods
        dfi = filter(:method => ==(meth), df_all)
        sort!(dfi, :z0)
        plot!(p_cov, dfi.z0, dfi.coverage;
              label=meth,
              linestyle=method_style(meth),
              lw=2,
              marker=:circle)
    end

    covfig_path = joinpath(plots_dir, "postmean_coverage_$(prior_key).png")
    savefig(p_cov, covfig_path)

    @info "Wrote plots" band_path=band_path covfig_path=covfig_path
    return nothing
end


function main(args=ARGS)
    d = parse_args(args)
    outdir = d["outdir"]

    pre_dir = joinpath(outdir, "precompute")
    meta_path = joinpath(pre_dir, "postmean_meta.jls")
    isfile(meta_path) || error("Missing meta file: $meta_path. Run precompute first.")
    meta = open(meta_path, "r") do io
        deserialize(io)
    end

    priors = Vector{String}(meta.priors)
    for pr in priors
        aggregate_one_prior(outdir, pr)
    end
    @info "Aggregation complete" outdir=outdir
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
