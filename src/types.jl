module Types

using Clarabel
using Gurobi
using HiGHS

"""
SolverConfig controls which solvers are used.

- LP problems (DKW / Gauss / Wasserstein constraints) use `lp_optimizer`.
- Conic problems (chi-square localization) use `conic_optimizer`.

### Gurobi notes
When `lp_optimizer == Gurobi.Optimizer`, the code will create a *single shared* `Gurobi.Env()`
per Julia process and pass it into `Gurobi.Optimizer(env)` (recommended for Gurobi.jl).

### Defaults
This rewrite defaults LPs to **Gurobi** and conic problems to **Clarabel**.
"""
Base.@kwdef struct SolverConfig
    # JuMP optimizer constructors (or factory functions). Examples:
    #   - Gurobi.Optimizer
    #   - HiGHS.Optimizer
    #   - () -> Gurobi.Optimizer(env)
    lp_optimizer::Any = Gurobi.Optimizer
    conic_optimizer::Any = Clarabel.Optimizer

    # Common run controls
    silent::Bool = true
    time_limit_sec::Float64 = Inf

    # Gurobi-only tuning (ignored unless lp_optimizer == Gurobi.Optimizer)
    gurobi_threads::Int = 1
    gurobi_method::Union{Nothing,Int} = nothing
    gurobi_numeric_focus::Union{Nothing,Int} = nothing
    gurobi_seed::Union{Nothing,Int} = 1
end

"""
Linear constraints on prior weights `w` (simplex variables).

Interpreted as:

- A * w <= b          (inequalities)
- Aeq * w == beq      (equalities)

The simplex constraints `w >= 0` and `sum(w) == 1` are always added by the F-localization solver.

Under the Charnes-Cooper transformation, the solver enforces these as:
- A * g <= b * ζ
- Aeq * g == beq * ζ
where `g = ζ * w`.
"""
struct PriorConstraints
    A::Matrix{Float64}
    b::Vector{Float64}
    Aeq::Matrix{Float64}
    beq::Vector{Float64}
end

function PriorConstraints(; A::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0),
                           b::AbstractVector{<:Real}=Float64[],
                           Aeq::AbstractMatrix{<:Real}=Matrix{Float64}(undef, 0, 0),
                           beq::AbstractVector{<:Real}=Float64[])
    return PriorConstraints(Matrix{Float64}(A), Float64.(b), Matrix{Float64}(Aeq), Float64.(beq))
end

end # module
