# Smooth / regularized Wasserstein-1 helpers ---------------------------------

using LinearAlgebra

"""
    kernel_cdf(u; sigma, kernel=:uniform)

CDF of the smoothing kernel `η_σ`.

Supported kernels:

- `:uniform`:  Uniform(-σ, σ)   (compact support)
- `:gaussian`: Normal(0, σ²)    (Gaussian kernel)

This is used for the *smooth Wasserstein* distance from Goldfeld et al. (2024):

    W^σ_1(μ, ν) := W_1(μ * η_σ, ν * η_σ).

For `:uniform`, the kernel CDF is

    H_σ(u) = 0                 if u ≤ -σ
           = (u + σ) / (2σ)    if -σ < u < σ
           = 1                 if u ≥ σ.
"""
@inline function kernel_cdf(u::Real; sigma::Real, kernel::Symbol=:uniform)::Float64
    σ = Float64(sigma)
    σ > 0 || error("sigma must be > 0; got sigma=$(sigma)")
    x = Float64(u)

    if kernel == :uniform
        if x <= -σ
            return 0.0
        elseif x >= σ
            return 1.0
        else
            return (x + σ) / (2σ)
        end
    elseif kernel == :gaussian
        # Gaussian kernel η_σ = Normal(0, σ²)
        # H_σ(u) = Φ(u/σ)
        return cdf(Normal(), x / σ)
    else
        error("Unsupported kernel=$(kernel). Supported: :uniform, :gaussian")
    end
end

"""
    smooth_empirical_cdf(z, t_grid; sigma, kernel=:uniform)

Compute the CDF values of the *convolved* empirical measure `μ̂_n * η_σ` on `t_grid`.

For each `t`,

    (μ̂_n * η_σ)((-∞, t]) = (1/n) Σ_i H_σ(t - z_i),

where `H_σ` is the kernel CDF.
"""
function smooth_empirical_cdf(z::AbstractVector{<:Real},
                              t_grid::AbstractVector{<:Real};
                              sigma::Real,
                              kernel::Symbol=:uniform)::Vector{Float64}
    n = length(z)
    n > 0 || error("z must be non-empty")

    σ = Float64(sigma)
    σ > 0 || error("sigma must be > 0; got sigma=$(sigma)")

    zf = Float64.(z)
    tf = Float64.(t_grid)
    m = length(tf)
    out = Vector{Float64}(undef, m)

    if kernel == :uniform
        @inbounds for k in 1:m
            t = tf[k]
            s = 0.0
            for i in 1:n
                u = t - zf[i]
                if u <= -σ
                    # add 0
                elseif u >= σ
                    s += 1.0
                else
                    s += (u + σ) / (2σ)
                end
            end
            out[k] = s / n
        end
    elseif kernel == :gaussian
        # H_σ(t - z_i) = Φ((t - z_i)/σ)
        @inbounds for k in 1:m
            t = tf[k]
            s = 0.0
            for i in 1:n
                s += cdf(Normal(), (t - zf[i]) / σ)
            end
            out[k] = s / n
        end
    else
        @inbounds for k in 1:m
            t = tf[k]
            s = 0.0
            for i in 1:n
                s += kernel_cdf(t - zf[i]; sigma=σ, kernel=kernel)
            end
            out[k] = s / n
        end
    end

    return out
end

"""
    smooth_cdf(cdf, t, μ; sigma, kernel=:uniform, quad_points=33)

Compute the convolved CDF `(F_{Z|μ} * η_σ)(t)` given a base CDF function
`cdf(t, μ) = P(Z ≤ t | μ)`.

For the uniform kernel, we use

    (F * η_σ)(t) = (1/(2σ)) ∫_{t-σ}^{t+σ} F(u) du,

approximated by the trapezoidal rule with `quad_points` evaluation points.

For the Gaussian kernel, we use Gauss–Hermite quadrature with `quad_points`
nodes.
"""
function smooth_cdf(cdf::Function,
                    t::Real,
                    μ::Real;
                    sigma::Real,
                    kernel::Symbol=:uniform,
                    quad_points::Integer=33)::Float64

    σ = Float64(sigma)
    σ > 0 || error("sigma must be > 0; got sigma=$(sigma)")
    qp = Int(quad_points)
    qp >= 2 || error("quad_points must be >= 2; got quad_points=$(quad_points)")

    if kernel == :uniform
        a = Float64(t) - σ
        b = Float64(t) + σ
        # Trapezoidal rule for the average value of F on [a, b].
        xs = range(a, b, length=qp)
        Δ = (b - a) / (qp - 1)
        acc = 0.0
        @inbounds for (i, x) in enumerate(xs)
            w = (i == 1 || i == qp) ? 0.5 : 1.0
            acc += w * Float64(cdf(x, μ))
        end
        integral = Δ * acc
        return integral / (b - a)
    elseif kernel == :gaussian
        # Gaussian kernel η_σ = Normal(0, σ²)
        # (F * η_σ)(t) = E[ F(t - U) ] with U ~ Normal(0, σ²).
        #
        # Use Gauss–Hermite quadrature:
        #   Let U = σ * √2 * X where X has density proportional to exp(-x^2).
        #   Then E[F(t-U)] = (1/√π) ∫ F(t - σ√2 x) e^{-x^2} dx
        #                ≈ Σ w_i F(t - σ√2 x_i),
        # where Σ w_i = 1.
        x, w = _gh_expectation_rule(qp)
        scale = σ * sqrt(2.0)
        tt = Float64(t)
        μf = Float64(μ)
        acc = 0.0
        @inbounds for i in 1:qp
            acc += w[i] * Float64(cdf(tt - scale * x[i], μf))
        end
        return acc
    else
        error("Unsupported kernel=$(kernel). Supported: :uniform, :gaussian")
    end
end

"""
    smooth_cdf_mat(cdf, t_grid, mu_grid; sigma, kernel=:uniform, quad_points=33)

Compute the matrix of convolved CDF values:

    C[k, j] = (F_{Z|μ_j} * η_σ)(t_grid[k]).

This is used to build LP constraints for smooth Wasserstein localization.
"""
function smooth_cdf_mat(cdf::Function,
                        t_grid::AbstractVector{<:Real},
                        mu_grid::AbstractVector{<:Real};
                        sigma::Real,
                        kernel::Symbol=:uniform,
                        quad_points::Integer=33)::Matrix{Float64}

    tf = Float64.(t_grid)
    mf = Float64.(mu_grid)
    m = length(tf)
    p = length(mf)

    out = Matrix{Float64}(undef, m, p)
    @inbounds for k in 1:m
        t = tf[k]
        for j in 1:p
            out[k, j] = smooth_cdf(cdf, t, mf[j];
                                   sigma=sigma,
                                   kernel=kernel,
                                   quad_points=quad_points)
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# Internal: Gauss–Hermite quadrature rule (expectation form)
# ---------------------------------------------------------------------------

"""Cache for Gauss–Hermite nodes/weights (expectation-normalized)."""
const _GH_CACHE = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()

"""
    _gh_expectation_rule(n)

Return `(x, w)` for Gauss–Hermite quadrature in **expectation form**:

    (1/√π) ∫_{-∞}^{∞} f(x) e^{-x^2} dx  ≈  Σ_{i=1}^n w[i] f(x[i]),

where `sum(w) == 1`.

We compute nodes/weights via the Golub–Welsch eigenvalue method.
"""
function _gh_expectation_rule(n::Int)
    n >= 1 || error("Gauss–Hermite order n must be >= 1; got n=$(n)")
    if haskey(_GH_CACHE, n)
        return _GH_CACHE[n]
    end

    if n == 1
        x = [0.0]
        w = [1.0]
        _GH_CACHE[n] = (x, w)
        return x, w
    end

    # Jacobi matrix for Hermite polynomials with weight exp(-x^2)
    β = sqrt.(Float64.(collect(1:(n-1))) ./ 2.0)
    J = SymTridiagonal(zeros(Float64, n), β)
    eig = eigen(J)

    x = Vector{Float64}(eig.values)
    v1 = @view eig.vectors[1, :]
    w = Vector{Float64}(v1 .^ 2)  # already normalized (sum=1)

    _GH_CACHE[n] = (x, w)
    return x, w
end
