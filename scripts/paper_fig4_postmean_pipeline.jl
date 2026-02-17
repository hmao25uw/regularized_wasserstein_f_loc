#!/usr/bin/env julia
#
# Convenience pipeline to generate *all four* figures requested for the
# posterior-mean simulation:
#   - Spiky CI bands
#   - NegSpiky CI bands
#   - Spiky coverage
#   - NegSpiky coverage
#
# It simply runs `scripts/paper_fig4_postmean.jl` twice (once per prior).
#
# Usage:
#   julia --project=. scripts/paper_fig4_postmean_pipeline.jl --outdir results/paper_fig4 --nreps 4000
#

using Printf

function parse_args(args)
    d = Dict{String,Any}(
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
        "boot_B" => 500,
        "clt_B" => 2000,
        "lp_time_limit" => 600.0,
        "gurobi_threads" => 1,
    )

    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--outdir"
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

function main(args=ARGS)
    d = parse_args(args)

    script = joinpath(@__DIR__, "paper_fig4_postmean.jl")
    outdir = d["outdir"]

    common = String[
        "--outdir", outdir,
        "--n", string(d["n"]),
        "--nreps", string(d["nreps"]),
        "--alpha", string(d["alpha"]),
        "--seed", string(d["seed"]),
        "--zmin", string(d["zmin"]),
        "--zmax", string(d["zmax"]),
        "--zstep", string(d["zstep"]),
        "--boot_B", string(d["boot_B"]),
        "--clt_B", string(d["clt_B"]),
        "--lp_time_limit", string(d["lp_time_limit"]),
        "--gurobi_threads", string(d["gurobi_threads"]),
    ]

    for prior in ("spiky", "negspiky")
        @printf("\n============================================================\n")
        @printf("Running posterior-mean simulation for prior: %s\n", prior)
        # NOTE (Windows/PowerShell robustness):
        # Interpolating `$(common...)` can concatenate the array entries into a
        # *single* argument (e.g. "--outdirresults/...--n500--nreps4000...")
        # which breaks `parse_args` in `paper_fig4_postmean.jl`.
        #
        # Interpolating the *vector* `$common` correctly splices it into
        # separate CLI arguments across platforms.
        cmd = `julia --project=. $script --prior $prior $common`
        run(cmd)
    end

    @printf("\nDone. Look in: %s/plots\n", outdir)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
