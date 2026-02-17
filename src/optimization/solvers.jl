module Optimization

using JuMP
import MathOptInterface as MOI

using ..Types: SolverConfig

# Solvers
using Gurobi
using HiGHS
using Clarabel

# ------------------------------------------------------------
# Gurobi environment handling
# ------------------------------------------------------------
#
# Gurobi.jl recommends reusing a single Gurobi.Env() per process.
# This avoids repeated license checks / setup overhead and is
# important for stable behavior on clusters.
#
const _GUROBI_ENV = Ref{Union{Nothing, Gurobi.Env}}(nothing)

function _gurobi_env()::Gurobi.Env
    if _GUROBI_ENV[] === nothing
        _GUROBI_ENV[] = Gurobi.Env()
    end
    return _GUROBI_ENV[]::Gurobi.Env
end

# ------------------------------------------------------------
# Model factories
# ------------------------------------------------------------

"""
Create a JuMP Model for *LP* problems, configured per `cfg`.

If `cfg.lp_optimizer == Gurobi.Optimizer`, we create the model using a closure:
`Model(() -> Gurobi.Optimizer(env))` to reuse the shared environment.
"""
function lp_model(cfg::SolverConfig)::JuMP.Model
    model = if cfg.lp_optimizer === Gurobi.Optimizer
        Model(() -> Gurobi.Optimizer(_gurobi_env()))
    elseif cfg.lp_optimizer === HiGHS.Optimizer
        Model(HiGHS.Optimizer)
    elseif cfg.lp_optimizer isa Function
        Model(cfg.lp_optimizer)
    else
        Model(cfg.lp_optimizer)
    end
    _apply_common_attributes!(model, cfg; problem_kind=:lp)
    return model
end

"""
Create a JuMP Model for conic problems (e.g., chi-square localization),
configured per `cfg`.
"""
function conic_model(cfg::SolverConfig)::JuMP.Model
    model = if cfg.conic_optimizer === Clarabel.Optimizer
        Model(Clarabel.Optimizer)
    elseif cfg.conic_optimizer isa Function
        Model(cfg.conic_optimizer)
    else
        Model(cfg.conic_optimizer)
    end
    _apply_common_attributes!(model, cfg; problem_kind=:conic)
    return model
end

# ------------------------------------------------------------
# Attributes & solve wrapper
# ------------------------------------------------------------

function _apply_common_attributes!(model::JuMP.Model, cfg::SolverConfig; problem_kind::Symbol)
    if cfg.silent
        set_silent(model)
    end
    if isfinite(cfg.time_limit_sec)
        set_time_limit_sec(model, cfg.time_limit_sec)
    end

    # LP-solver-specific attributes
    if problem_kind == :lp && cfg.lp_optimizer === Gurobi.Optimizer
        # Silence Gurobi (JuMP's set_silent may not always set OutputFlag)
        if cfg.silent
            set_optimizer_attribute(model, "OutputFlag", 0)
        end
        if cfg.gurobi_threads > 0
            set_optimizer_attribute(model, "Threads", cfg.gurobi_threads)
        end
        if cfg.gurobi_method !== nothing
            set_optimizer_attribute(model, "Method", cfg.gurobi_method)
        end
        if cfg.gurobi_numeric_focus !== nothing
            set_optimizer_attribute(model, "NumericFocus", cfg.gurobi_numeric_focus)
        end
        if cfg.gurobi_seed !== nothing
            set_optimizer_attribute(model, "Seed", cfg.gurobi_seed)
        end
    end
    return model
end

"""
Solve the model and throw if the solve status is not acceptable.

Set `allow_time_limit=true` to accept a TIME_LIMIT termination if a feasible
solution exists (useful on SLURM when you want an incumbent).
"""
function optimize_or_throw!(model::JuMP.Model; allow_time_limit::Bool=false)
    optimize!(model)

    term = termination_status(model)
    primal = primal_status(model)

    if term == MOI.OPTIMAL
        return
    end

    if allow_time_limit &&
       term == MOI.TIME_LIMIT &&
       (primal == MOI.FEASIBLE_POINT || primal == MOI.NEARLY_FEASIBLE_POINT)
        @warn "TIME_LIMIT reached; proceeding with incumbent feasible solution."
        return
    end

    msg = """
Optimization failed.

termination_status: $(term)
primal_status:      $(primal)
dual_status:        $(dual_status(model))
raw_status:         $(raw_status(model))
objective_bound:    $(objective_bound(model))
"""
    error(msg)
end

end # module
