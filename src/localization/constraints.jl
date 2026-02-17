# JuMP constraint builders for each localization type ------------------------

"""
Add DKW constraints (LP) to the JuMP model.

Inputs:
- model: JuMP.Model
- loc: DKWLocalization
- g, ζ: Charnes-Cooper variables (g = ζ*w)
- Ccdf: matrix with entries Ccdf[k,j] = P(Z <= t_k | μ_j)
- stats: NamedTuple from prepare_localization_stats (contains t_grid, Fhat, eps)
"""
function add_constraints!(model::JuMP.Model,
                          loc::DKWLocalization,
                          g,
                          ζ,
                          Ccdf::AbstractMatrix{<:Real},
                          stats::NamedTuple)
    eps = stats.eps
    Fhat = stats.Fhat
    m, p = size(Ccdf)
    length(Fhat) == m || error("DKW: length(Fhat) != size(Ccdf,1)")
    length(g) == p || error("DKW: length(g) != size(Ccdf,2)")

    for k in 1:m
        expr = sum(Ccdf[k, j] * g[j] for j in 1:p) - ζ * Fhat[k]
        @constraint(model, expr <= eps * ζ)
        @constraint(model, -expr <= eps * ζ)
    end
    return nothing
end

"""
Add Gauss-F constraints (LP) to the JuMP model.

Inputs:
- PdfMat[k,j] = p(x_k | μ_j)
- stats provides fhat[k] and radius c.
"""
function add_constraints!(model::JuMP.Model,
                          loc::GaussLocalization,
                          g,
                          ζ,
                          PdfMat::AbstractMatrix{<:Real},
                          stats::NamedTuple)
    c = stats.c
    fhat = stats.fhat
    m, p = size(PdfMat)
    length(fhat) == m || error("Gauss: length(fhat) != size(PdfMat,1)")
    length(g) == p || error("Gauss: length(g) != size(PdfMat,2)")

    for k in 1:m
        expr = sum(PdfMat[k, j] * g[j] for j in 1:p) - ζ * fhat[k]
        @constraint(model, expr <= c * ζ)
        @constraint(model, -expr <= c * ζ)
    end
    return nothing
end

"""
Add Wasserstein constraints (LP) using the 1D identity:

    W1(F, Fhat) = ∫ |F(t) - Fhat(t)| dt

Approximated on an ordered grid t_1 < ... < t_m as:
    sum_{k=1}^{m-1} |F(t_k) - Fhat(t_k)| * (t_{k+1} - t_k) <= rho

We linearize the absolute value with u_k >= ±(F(t_k) - Fhat(t_k)).
"""
function add_constraints!(model::JuMP.Model,
                          loc::WassersteinLocalization,
                          g,
                          ζ,
                          Ccdf::AbstractMatrix{<:Real},
                          stats::NamedTuple)
    rho = stats.rho
    Fhat = stats.Fhat
    Δt = stats.Δt

    m, p = size(Ccdf)
    length(Fhat) == m || error("Wasserstein: length(Fhat) != size(Ccdf,1)")
    length(Δt) == m - 1 || error("Wasserstein: length(Δt) != size(Ccdf,1)-1")
    length(g) == p || error("Wasserstein: length(g) != size(Ccdf,2)")

    @variable(model, u[1:m-1] >= 0)

    for k in 1:(m-1)
        expr = sum(Ccdf[k, j] * g[j] for j in 1:p) - ζ * Fhat[k]
        @constraint(model, u[k] >= expr)
        @constraint(model, u[k] >= -expr)
    end

    @constraint(model, sum(u[k] * Δt[k] for k in 1:(m-1)) <= rho * ζ)
    return u
end

"""
Add chi-square constraints as a Second-Order Cone (SOC):

    || D^{-1/2} (p - p_hat) ||_2 <= sqrt(r)

Under Charnes-Cooper with g=ζw and P = ζ p:
    || D^{-1/2} (P - ζ p_hat) ||_2 <= sqrt(r) * ζ

Inputs:
- Ppmf[k,j] = P(Z = support_k | μ_j)   (a pmf matrix on discrete support)
- stats provides support, p_hat, r
"""
function add_constraints!(model::JuMP.Model,
                          loc::Chi2Localization,
                          g,
                          ζ,
                          Ppmf::AbstractMatrix{<:Real},
                          stats::NamedTuple)
    p_hat = stats.p_hat
    r = stats.r

    m, p = size(Ppmf)
    length(p_hat) == m || error("Chi2: length(p_hat) != size(Ppmf,1)")
    length(g) == p || error("Chi2: length(g) != size(Ppmf,2)")
    any(p_hat .<= 0) && error("Chi2: p_hat has zeros; cannot scale by sqrt(p_hat).")

    # y_k = (P_k - ζ p_hat_k)/sqrt(p_hat_k)
    @expression(model, y[k=1:m], (sum(Ppmf[k, j] * g[j] for j in 1:p) - ζ * p_hat[k]) / sqrt(p_hat[k]))

    t = sqrt(r) * ζ
    @constraint(model, [t; y] in SecondOrderCone())
    return nothing
end
