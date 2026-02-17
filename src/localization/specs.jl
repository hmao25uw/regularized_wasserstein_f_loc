# Localization specifications + data-dependent preparation -------------------

abstract type AbstractLocalization end

Base.@kwdef struct DKWLocalization <: AbstractLocalization
    alpha::Float64
    t_grid::Vector{Float64}
    radius_override::Union{Nothing,Float64} = nothing  # overrides epsilon if provided
end

Base.@kwdef struct GaussLocalization <: AbstractLocalization
    alpha::Float64
    x_grid::Vector{Float64}
    bandwidth::Float64
    B::Int = 2000
    radius_override::Union{Nothing,Float64} = nothing  # overrides bootstrap radius c if provided
end

Base.@kwdef struct WassersteinLocalization <: AbstractLocalization
    alpha::Float64
    t_grid::Vector{Float64}
    radius_method::Symbol = :dkw   # :dkw | :bootstrap | :clt
    support::Union{Nothing,Tuple{Float64,Float64}} = nothing  # required for :dkw (if not set, inferred from data)
    B::Int = 2000
    radius_override::Union{Nothing,Float64} = nothing  # overrides rho if provided

    # Optional *regularization* (Goldfeld et al. 2024 smooth Wasserstein)
    #
    # If `regularization == :smooth`, we localize using
    #
    #     W₁^σ(μ, ν) := W₁(μ * η_σ, ν * η_σ),
    #
    # where `η_σ` is a compactly supported kernel. The constraints remain LP
    # constraints in 1D because W₁ is still an L1 distance between CDFs.
    regularization::Symbol = :none   # :none | :smooth
    smooth_sigma::Float64 = 0.0      # σ > 0 required when regularization == :smooth
    kernel::Symbol = :uniform        # currently only :uniform
    quad_points::Int = 33            # trapezoid points for convolving model CDF
end

Base.@kwdef struct Chi2Localization <: AbstractLocalization
    alpha::Float64
    z_support::Vector{Float64}     # discrete support points for Z
    radius_override::Union{Nothing,Float64} = nothing
end

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

"""
DKW epsilon for sup-norm CDF deviation: P( sup_t |Fhat(t) - F(t)| <= eps ) >= 1-alpha.
"""
dkw_epsilon(alpha::Real, n::Integer) = sqrt(log(2 / alpha) / (2n))

"""
Empirical CDF evaluated at t_grid (no assumptions about sorting).
"""
function empirical_cdf(z::AbstractVector{<:Real}, t_grid::AbstractVector{<:Real})::Vector{Float64}
    z_sorted = sort(Float64.(z))
    n = length(z_sorted)
    out = Vector{Float64}(undef, length(t_grid))
    for (k, t) in enumerate(t_grid)
        out[k] = searchsortedlast(z_sorted, Float64(t)) / n
    end
    return out
end

"""
Empirical PMF on a discrete support.

Returns a probability vector p_hat of length m = length(z_support).
Errors if any observation is not in the provided support.
"""
function empirical_pmf(z::AbstractVector{<:Real}, z_support::AbstractVector{<:Real})::Vector{Float64}
    support = Float64.(z_support)
    m = length(support)
    n = length(z)
    n == 0 && error("z must be non-empty")

    # Map value -> index for O(1) lookup
    idx = Dict{Float64,Int}(support[i] => i for i in 1:m)
    counts = zeros(Float64, m)
    for zi in z
        key = Float64(zi)
        haskey(idx, key) || error("empirical_pmf: observation $(zi) not found in z_support")
        counts[idx[key]] += 1.0
    end
    return counts ./ n
end

# ---------------------------------------------------------------------------
# Data-dependent preparation for each localization type
# ---------------------------------------------------------------------------

"""
Prepare DKW localization statistics for the given data.
Returns a NamedTuple: (t_grid, Fhat, eps).
"""
function prepare_localization_stats(loc::DKWLocalization, z::AbstractVector{<:Real}; rng=nothing)
    n = length(z)
    eps = loc.radius_override === nothing ? dkw_epsilon(loc.alpha, n) : loc.radius_override
    t_grid = Float64.(loc.t_grid)
    Fhat = empirical_cdf(z, t_grid)
    return (t_grid=t_grid, Fhat=Fhat, eps=Float64(eps))
end

"""
Prepare Wasserstein localization statistics for the given data.
Returns a NamedTuple: (t_grid, Fhat, rho, Δt).
"""
function prepare_localization_stats(loc::WassersteinLocalization, z::AbstractVector{<:Real};
                                   rng::AbstractRNG=Random.default_rng())
    n = length(z)
    t_grid = sort(Float64.(loc.t_grid))
    Δt = diff(t_grid)

    # Empirical CDF (standard vs smooth/regularized)
    Fhat = if loc.regularization == :none
        empirical_cdf(z, t_grid)
    elseif loc.regularization == :smooth
        loc.smooth_sigma > 0 || error("WassersteinLocalization: smooth_sigma must be > 0 when regularization=:smooth")
        smooth_empirical_cdf(z, t_grid; sigma=loc.smooth_sigma, kernel=loc.kernel)
    else
        error("Unknown Wasserstein regularization=$(loc.regularization). Use :none or :smooth.")
    end

    rho = if loc.radius_override !== nothing
        Float64(loc.radius_override)
    else
        if loc.regularization == :none
            if loc.radius_method == :dkw
                supp = loc.support === nothing ? (minimum(z), maximum(z)) : loc.support
                wasserstein_radius_dkw(alpha=loc.alpha, n=n, support=supp)
            elseif loc.radius_method == :bootstrap
                wasserstein_radius_bootstrap(z; alpha=loc.alpha, B=loc.B, rng=rng)
            elseif loc.radius_method == :clt
                # CLT radius based on the limit law for √n·W₁(F, F̂ₙ).
                # We pass (F̂ₙ, Δt) so the calibration matches the LP discretization.
                _wasserstein_clt_from_cdf(Fhat, Δt, n; alpha=loc.alpha, B=loc.B, rng=rng, nballs=1)
            else
                error("Unknown Wasserstein radius_method=$(loc.radius_method). Use :dkw, :bootstrap, or :clt.")
            end
        elseif loc.regularization == :smooth
            if loc.radius_method == :clt
                wasserstein_smooth_clt(z;
                                      alpha=loc.alpha,
                                      t_grid=t_grid,
                                      smooth_sigma=loc.smooth_sigma,
                                      kernel=loc.kernel,
                                      B=loc.B,
                                      rng=rng,
                                      nballs=1)
            else
                error("For regularization=:smooth, only radius_method=:clt is currently implemented.")
            end
        else
            error("Unknown Wasserstein regularization=$(loc.regularization). Use :none or :smooth.")
        end
    end
    return (t_grid=t_grid, Fhat=Fhat, rho=rho, Δt=Δt)
end

"""
Prepare Gauss-F localization statistics for the given data.
Returns a NamedTuple: (x_grid, fhat, c).
"""
function prepare_localization_stats(loc::GaussLocalization, z::AbstractVector{<:Real};
                                   rng::AbstractRNG=Random.default_rng())
    x_grid = Float64.(loc.x_grid)
    fhat = gaussian_kde(z, x_grid, loc.bandwidth)
    c = loc.radius_override === nothing ?
        gauss_radius_bootstrap(z, x_grid, loc.bandwidth; alpha=loc.alpha, B=loc.B, rng=rng, fhat=fhat) :
        Float64(loc.radius_override)
    return (x_grid=x_grid, fhat=fhat, c=c)
end

"""
Prepare chi-square localization statistics for the given data.
Returns a NamedTuple: (support, p_hat, r).
"""
function prepare_localization_stats(loc::Chi2Localization, z::AbstractVector{<:Real}; rng=nothing)
    support = Float64.(loc.z_support)
    p_hat = empirical_pmf(z, support)

    # Guard against zeros in p_hat (they break the chi-square scaling).
    any(p_hat .<= 0) && error("Chi2Localization requires p_hat_i > 0 for all support points. Consider coarsening bins.")

    n = length(z)
    m = length(support)
    r = loc.radius_override === nothing ? chi2_radius(alpha=loc.alpha, n=n, m=m) : Float64(loc.radius_override)
    return (support=support, p_hat=p_hat, r=r)
end
