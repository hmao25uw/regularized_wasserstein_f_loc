#!/usr/bin/env julia
# Plot CI results produced by scripts/run_experiment.jl or scripts/pipeline_local.jl.
#
# Usage:
#   julia --project=. scripts/plot_results.jl --outdir results
#   julia --project=. scripts/plot_results.jl --outdir results --configs_dir configs

using TOML
using CSV
using DataFrames

# Make Plots work on headless machines (e.g., SLURM nodes).
ENV["GKSwstype"] = get(ENV, "GKSwstype", "100")
using Plots


function parse_args(args)
    d = Dict{String,String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--outdir"
            d["outdir"] = args[i+1]
            i += 2
        elseif a == "--configs_dir"
            d["configs_dir"] = args[i+1]
            i += 2
        elseif startswith(a, "--") && occursin("=", a)
            k, v = split(a[3:end], "=", limit=2)
            d[k] = v
            i += 1
        else
            error("Unknown argument: $a")
        end
    end
    return d
end

function _ensure_dir(path::AbstractString)
    isdir(path) || mkpath(path)
    return path
end

function _label_for_config(cfg::Dict)::String
    loc = cfg["localization"]
    method = loc["method"]

    if method == "dkw"
        return "DKW-F"
    elseif method == "gauss"
        return "Gauss-F"
    elseif method == "chi2"
        return "χ²-F"
    elseif method == "wasserstein"
        rmethod = haskey(loc, "radius_method") ? loc["radius_method"] : "dkw"
        reg = haskey(loc, "regularization") ? loc["regularization"] : "none"
        if reg == "smooth"
            return "Smooth-W₁ (CLT)"
        else
            # none
            if rmethod == "bootstrap"
                return "W₁ (boot)"
            elseif rmethod == "dkw"
                return "W₁ (DKW)"
            elseif rmethod == "clt"
                return "W₁ (CLT)"
            else
                return "W₁ ($(rmethod))"
            end
        end
    else
        return string(method)
    end
end

function _result_path(outdir::AbstractString, cfg_path::AbstractString)
    stem = replace(basename(cfg_path), ".toml" => "")
    return joinpath(outdir, stem * "_ci.csv")
end

function _load_ci_csv(path::AbstractString)::DataFrame
    df = CSV.read(path, DataFrame; header=false)
    if ncol(df) < 3
        error("Expected at least 3 columns (z0, lower, upper) in $(path)")
    end
    rename!(df, Dict(names(df)[1] => :z0, names(df)[2] => :lower, names(df)[3] => :upper))
    df.z0 = Float64.(df.z0)
    df.lower = Float64.(df.lower)
    df.upper = Float64.(df.upper)
    sort!(df, :z0)
    return df
end

function plot_all(outdir::AbstractString; configs_dir::AbstractString="configs")
    cfgs = sort(filter(f -> endswith(f, ".toml"), readdir(configs_dir; join=true)))
    isempty(cfgs) && error("No .toml configs found in $(configs_dir)")

    labels = String[]
    dfs = DataFrame[]

    for cfg_path in cfgs
        cfg = TOML.parsefile(cfg_path)
        label = _label_for_config(cfg)
        res_path = _result_path(outdir, cfg_path)
        if !isfile(res_path)
            @warn "Skipping (no results found): $(res_path)"
            continue
        end
        push!(labels, label)
        push!(dfs, _load_ci_csv(res_path))
    end

    isempty(dfs) && error("No result CSVs found in $(outdir). Run experiments first.")

    plots_dir = _ensure_dir(joinpath(outdir, "plots"))

    # ------------------------------------------------------------
    # Plot 1: CI length vs z0
    # ------------------------------------------------------------
    p_len = plot(
        xlabel = "z₀",
        ylabel = "CI length (upper − lower)",
        legend = :topright,
        title = "F-localization CI length",
    )

    for (df, lab) in zip(dfs, labels)
        len = df.upper .- df.lower
        plot!(p_len, df.z0, len; label=lab, marker=:circle)
    end

    savefig(p_len, joinpath(plots_dir, "ci_length.png"))

    # ------------------------------------------------------------
    # Plot 2: CI bands (ribbons) vs z0
    # ------------------------------------------------------------
    p_band = plot(
        xlabel = "z₀",
        ylabel = "CI",
        legend = :topright,
        title = "F-localization confidence intervals",
    )

    for (df, lab) in zip(dfs, labels)
        center = (df.upper .+ df.lower) ./ 2
        ribbon = (df.upper .- df.lower) ./ 2
        plot!(p_band, df.z0, center; ribbon=ribbon, label=lab, marker=:circle)
    end

    savefig(p_band, joinpath(plots_dir, "ci_bands.png"))

    @info "Wrote plots to $(plots_dir)"
    return nothing
end

function main(args=ARGS)
    d = parse_args(args)
    outdir = get(d, "outdir", "results")
    configs_dir = get(d, "configs_dir", joinpath(@__DIR__, "..", "configs"))
    plot_all(outdir; configs_dir=configs_dir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
