#!/usr/bin/env julia
# Usage:
#   julia --project=. scripts/run_experiment.jl --config configs/gaussian_sim_wasserstein.toml --outdir results

using NonparBayesCI

function parse_args(args)
    d = Dict{String,String}()
    i = 1
    while i <= length(args)
        a = args[i]
        if startswith(a, "--") && occursin("=", a)
            k, v = split(a[3:end], "=", limit=2)
            d[k] = v
            i += 1
        elseif a == "--config"
            d["config"] = args[i+1]
            i += 2
        elseif a == "--outdir"
            d["outdir"] = args[i+1]
            i += 2
        else
            error("Unknown argument: $a")
        end
    end
    return d
end

function main(args=ARGS)
    d = parse_args(args)
    haskey(d, "config") || error("Missing --config <path>")
    outdir = get(d, "outdir", "results")
    NonparBayesCI.run_config(d["config"]; outdir=outdir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
