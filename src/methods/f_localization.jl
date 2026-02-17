module Methods

using Random
using LinearAlgebra
using JuMP
import MathOptInterface as MOI

using ..Types: SolverConfig, PriorConstraints
using ..Optimization: lp_model, conic_model, optimize_or_throw!
using ..Localization: AbstractLocalization,
    DKWLocalization, GaussLocalization, WassersteinLocalization, Chi2Localization,
    prepare_localization_stats, add_constraints!,
    smooth_cdf_mat

export EBProblem, f_localization_bound, f_localization_ci

"""
Empirical Bayes point-functional problem:

    θ_G(z0) = E_G[h(μ) | Z = z0]
            = ∫ h(μ) p(z0|μ) dG(μ) / ∫ p(z0|μ) dG(μ)

We approximate G by a discrete distribution on `mu_grid`.
"""
struct EBProblem{Fpdf,Fcdf,Fh}
    mu_grid::Vector{Float64}
    pdf::Fpdf
    cdf::Fcdf
    h::Fh
    prior_constraints::PriorConstraints
end

function EBProblem(mu_grid::AbstractVector{<:Real};
                   pdf::Function,
                   cdf::Function,
                   h::Function,
                   prior_constraints::PriorConstraints=PriorConstraints())
    return EBProblem(Float64.(mu_grid), pdf, cdf, h, prior_constraints)
end

# ---------------------------------------------------------------------------
# Internal helpers: precompute likelihood vectors/matrices
# ---------------------------------------------------------------------------

function _pdf_vec(pdf::Function, z0::Real, mu_grid::Vector{Float64})::Vector{Float64}
    p = length(mu_grid)
    out = Vector{Float64}(undef, p)
    for j in 1:p
        out[j] = Float64(pdf(z0, mu_grid[j]))
    end
    return out
end

function _pdf_mat(pdf::Function, x_grid::Vector{Float64}, mu_grid::Vector{Float64})::Matrix{Float64}
    m = length(x_grid)
    p = length(mu_grid)
    out = Matrix{Float64}(undef, m, p)
    for k in 1:m
        x = x_grid[k]
        for j in 1:p
            out[k, j] = Float64(pdf(x, mu_grid[j]))
        end
    end
    return out
end

function _cdf_mat(cdf::Function, t_grid::Vector{Float64}, mu_grid::Vector{Float64})::Matrix{Float64}
    m = length(t_grid)
    p = length(mu_grid)
    out = Matrix{Float64}(undef, m, p)
    for k in 1:m
        t = t_grid[k]
        for j in 1:p
            out[k, j] = Float64(cdf(t, mu_grid[j]))
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Prior constraints A*w<=b, Aeq*w==beq  =>  A*g<=b*ζ, Aeq*g==beq*ζ
# ---------------------------------------------------------------------------

function _add_prior_constraints!(model::JuMP.Model,
                                 g,
                                 ζ,
                                 pc::PriorConstraints)
    p = length(g)

    if size(pc.A, 1) > 0
        size(pc.A, 2) == p || error("PriorConstraints.A has $(size(pc.A,2)) columns but g has length $p.")
        length(pc.b) == size(pc.A, 1) || error("PriorConstraints.b length mismatch with A rows.")
        for i in 1:size(pc.A, 1)
            @constraint(model, sum(pc.A[i, j] * g[j] for j in 1:p) <= pc.b[i] * ζ)
        end
    end

    if size(pc.Aeq, 1) > 0
        size(pc.Aeq, 2) == p || error("PriorConstraints.Aeq has $(size(pc.Aeq,2)) columns but g has length $p.")
        length(pc.beq) == size(pc.Aeq, 1) || error("PriorConstraints.beq length mismatch with Aeq rows.")
        for i in 1:size(pc.Aeq, 1)
            @constraint(model, sum(pc.Aeq[i, j] * g[j] for j in 1:p) == pc.beq[i] * ζ)
        end
    end

    return nothing
end

# ---------------------------------------------------------------------------
# One-sided bound via Charnes-Cooper
# ---------------------------------------------------------------------------

"""
Compute one side of the F-localization bound at a single z0.

Arguments
- z: sample of observations Z_1,...,Z_n
- z0: point where we want the EB estimand θ(z0)
- problem: EBProblem containing mu_grid, (pdf,cdf), h
- loc: localization specification (DKW, Gauss, Wasserstein, Chi2)

Keyword args
- solver: SolverConfig (LP uses cfg.lp_optimizer, conic uses cfg.conic_optimizer)
- direction: :upper or :lower
- rng: RNG used only for bootstrap-calibrated radii

Returns: (bound_value, w_star, zeta_star)
"""
function f_localization_bound(z::AbstractVector{<:Real},
                              z0::Real,
                              problem::EBProblem,
                              loc::AbstractLocalization;
                              solver::SolverConfig = SolverConfig(),
                              direction::Symbol = :upper,
                              rng::AbstractRNG = Random.default_rng(),
                              stats_override::Union{Nothing,NamedTuple}=nothing,
                              mat_cache::Union{Nothing,NamedTuple}=nothing)

    direction in (:upper, :lower) || error("direction must be :upper or :lower")

    mu = problem.mu_grid
    p = length(mu)

    # Likelihood at z0
    pz = _pdf_vec(problem.pdf, z0, mu)
    if all(pz .<= 0)
        error("All pdf(z0|μ_j) are <= 0 on the mu_grid. Check mu_grid range or likelihood definition.")
    end

    # Objective coefficients
    q = Vector{Float64}(undef, p)
    for j in 1:p
        q[j] = Float64(problem.h(mu[j])) * pz[j]
    end

    # Prepare localization stats (may use bootstrap/CLT simulation).
    # IMPORTANT: For stochastic radius calibration (bootstrap / CLT simulation),
    # we must reuse the *same* radius for both lower/upper bounds at a given
    # dataset. Provide `stats_override` to enforce this.
    stats = stats_override === nothing ?
        prepare_localization_stats(loc, z; rng=rng) :
        stats_override

    # Choose LP vs conic model
    model = (loc isa Chi2Localization) ? conic_model(solver) : lp_model(solver)

    # Charnes-Cooper variables
    @variable(model, ζ >= 0)
    @variable(model, g[1:p] >= 0)

    # simplex scaling: sum(w)=1  <=>  sum(g)=ζ
    @constraint(model, sum(g) == ζ)

    # normalization at z0: sum_j p(z0|μ_j) w_j = 1/ζ  <=>  sum_j p(z0|μ_j) g_j = 1
    @constraint(model, sum(pz[j] * g[j] for j in 1:p) == 1)

    # Optional linear constraints on w
    _add_prior_constraints!(model, g, ζ, problem.prior_constraints)

    # Localization constraints
    if loc isa DKWLocalization
        Ccdf = (mat_cache !== nothing && hasproperty(mat_cache, :Ccdf)) ?
            mat_cache.Ccdf :
            _cdf_mat(problem.cdf, stats.t_grid, mu)
        add_constraints!(model, loc, g, ζ, Ccdf, stats)

    elseif loc isa WassersteinLocalization
        Ccdf = if (mat_cache !== nothing && hasproperty(mat_cache, :Ccdf))
            mat_cache.Ccdf
        elseif loc.regularization == :none
            _cdf_mat(problem.cdf, stats.t_grid, mu)
        elseif loc.regularization == :smooth
            smooth_cdf_mat(problem.cdf, stats.t_grid, mu;
                           sigma=loc.smooth_sigma,
                           kernel=loc.kernel,
                           quad_points=loc.quad_points)
        else
            error("Unknown Wasserstein regularization=$(loc.regularization). Use :none or :smooth.")
        end
        add_constraints!(model, loc, g, ζ, Ccdf, stats)

    elseif loc isa GaussLocalization
        PdfMat = (mat_cache !== nothing && hasproperty(mat_cache, :PdfMat)) ?
            mat_cache.PdfMat :
            _pdf_mat(problem.pdf, stats.x_grid, mu)
        add_constraints!(model, loc, g, ζ, PdfMat, stats)

    elseif loc isa Chi2Localization
        # On a discrete support: P(Z = support_k | μ_j)
        Ppmf = (mat_cache !== nothing && hasproperty(mat_cache, :Ppmf)) ?
            mat_cache.Ppmf :
            _pdf_mat(problem.pdf, stats.support, mu)
        add_constraints!(model, loc, g, ζ, Ppmf, stats)

    else
        error("Unsupported localization type: $(typeof(loc))")
    end

    # Objective
    if direction == :upper
        @objective(model, Max, sum(q[j] * g[j] for j in 1:p))
    else
        @objective(model, Min, sum(q[j] * g[j] for j in 1:p))
    end

    optimize_or_throw!(model)

    bound = objective_value(model)
    ζ_val = value(ζ)
    g_val = value.(g)
    w_val = g_val ./ ζ_val

    return (bound, w_val, ζ_val)
end

"""
Two-sided F-localization CI at z0.
Returns (lower, upper).
"""
function f_localization_ci(z::AbstractVector{<:Real},
                           z0::Real,
                           problem::EBProblem,
                           loc::AbstractLocalization;
                           solver::SolverConfig = SolverConfig(),
                           rng::AbstractRNG = Random.default_rng(),
                           stats_override::Union{Nothing,NamedTuple}=nothing,
                           mat_cache::Union{Nothing,NamedTuple}=nothing)
    # Compute (or reuse) localization stats once so both one-sided solves
    # share the exact same localization set.
    stats = stats_override === nothing ?
        prepare_localization_stats(loc, z; rng=rng) :
        stats_override

    lower, _, _ = f_localization_bound(z, z0, problem, loc;
                                       solver=solver,
                                       direction=:lower,
                                       rng=rng,
                                       stats_override=stats,
                                       mat_cache=mat_cache)
    upper, _, _ = f_localization_bound(z, z0, problem, loc;
                                       solver=solver,
                                       direction=:upper,
                                       rng=rng,
                                       stats_override=stats,
                                       mat_cache=mat_cache)
    return (lower, upper)
end

end # module
