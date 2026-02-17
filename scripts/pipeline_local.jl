#!/usr/bin/env julia
# Run all TOML configs in ./configs sequentially (local machine).
#
# Usage:
#   julia --project=. scripts/pipeline_local.jl --outdir results
#   julia --project=. scripts/pipeline_local.jl --outdir results --no-plot

using NonparBayesCI

function parse_args(args)
    d = Dict{String,Any}()
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--outdir"
            d["outdir"] = args[i+1]
            i += 2
        elseif a == "--no-plot"
            d["no_plot"] = true
            i += 1
        else
            error("Unknown argument: $a")
        end
    end
    return d
end

function main(args=ARGS)
    d = parse_args(args)
    outdir = get(d, "outdir", "results")
    no_plot = get(d, "no_plot", false)

    cfg_dir = joinpath(@__DIR__, "..", "configs")
    cfgs = sort(filter(f -> endswith(f, ".toml"), readdir(cfg_dir; join=true)))

    isempty(cfgs) && error("No .toml configs found in $cfg_dir")

    for cfg in cfgs
        println("============================================================")
        println("Running config: $cfg")
        NonparBayesCI.run_config(cfg; outdir=outdir)
    end

    if !no_plot
        println("============================================================")
        println("Plotting results into: $(joinpath(outdir, "plots"))")
        # Run plotting in a fresh Julia process so plotting deps are isolated.
        # (This also avoids loading Plots during the optimization runs.)
        cmd = `julia --project=. $(joinpath(@__DIR__, "plot_results.jl")) --outdir $(outdir)`
        run(cmd)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
