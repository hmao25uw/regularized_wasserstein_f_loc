# Wasserstein radii ----------------------------------------------------------

using LinearAlgebra

"""
Finite-sample radius for W1(F, Fhat) using DKW under bounded support.

In 1D, W1(F, Fhat) = ∫ |F(t) - Fhat(t)| dt.
If support is contained in [a, b], then W1 <= (b-a) * sup_t |F(t) - Fhat(t)|.
DKW gives: sup_t |F(t) - Fhat(t)| <= sqrt(log(2/alpha)/(2n)) with prob >= 1-alpha.
"""
function wasserstein_radius_dkw(; alpha::Real, n::Integer, support::Tuple{<:Real,<:Real})::Float64
    a, b = support
    width = Float64(b - a)
    width < 0 && error("support must satisfy a <= b; got support=$(support)")
    alpha <= 0 && error("alpha must be in (0,1); got alpha=$(alpha)")
    n <= 0 && error("n must be positive; got n=$(n)")
    eps = sqrt(log(2 / alpha) / (2n))
    return width * eps
end

"""
Bootstrap-calibrated radius for W1(F, Fhat).

We approximate the distribution of W1(Fhat*, Fhat) via bootstrap samples,
then take the (1-alpha)-quantile.

For 1D empirical measures with equal weights, W1(Fhat*, Fhat) is the mean
absolute difference between the sorted samples.
"""
function wasserstein_radius_bootstrap(z::AbstractVector{<:Real};
                                     alpha::Real,
                                     B::Integer=2000,
                                     rng::AbstractRNG=Random.default_rng())::Float64
    n = length(z)
    n == 0 && error("z must be non-empty")
    alpha <= 0 && error("alpha must be in (0,1); got alpha=$(alpha)")
    B <= 0 && error("B must be positive; got B=$(B)")

    z_sorted = sort(Float64.(z))
    d = Vector{Float64}(undef, B)
    for b in 1:B
        idx = rand(rng, 1:n, n)          # sample indices with replacement
        z_star = z_sorted[idx]           # bootstrap sample values
        sort!(z_star)
        d[b] = mean(abs.(z_star .- z_sorted))
    end
    return quantile(d, 1 - alpha)
end

"""
    wasserstein_clt(z; alpha, t_grid=nothing, B=2000, rng=Random.default_rng(), nballs=1)

CLT-calibrated *plug-in* radius for the **1D Wasserstein-1** distance between the
unknown distribution of `Z` and the empirical distribution based on the sample `z`.

In 1D,

    W₁(F, F̂ₙ) = ∫ |F(t) - F̂ₙ(t)| dt.

Under mild moment assumptions, and specializing the one-sample limit law in
Goldfeld–Kato–Rioux–Sadhu ("Statistical inference with regularized optimal transport")
to the case ν = μ, we have

    √n · W₁(F, F̂ₙ) ⇒ ∫ |G(t)| dt,

where `G` is a centered Gaussian process with covariance

    Cov(G(s), G(t)) = F(min(s,t)) - F(s)F(t).

We approximate the integral on an ordered grid `t_grid` by the *left Riemann sum*
used by the LP constraints in `constraints.jl`:

    ∫ |G(t)| dt  ≈  Σ_{k=1}^{m-1} |G(t_k)| · (t_{k+1} - t_k).

We then set the radius

    ρ = q / √n,

where `q` is the `(1-α)^(1/nballs)` quantile of the simulated limiting draws.

## Keyword arguments

- `alpha`: miscoverage level in (0, 1).
- `t_grid`: optional grid. If `nothing`, we use 200 equally spaced points on
  `[minimum(z), maximum(z)]`.
- `B`: number of Monte Carlo draws from the limiting Gaussian process.
- `rng`: random number generator.
- `nballs`: optional "product" adjustment for joint coverage of multiple independent
  balls. Default 1, giving quantile level `1-α`. If you need two independent balls with
  joint coverage ≈ `1-α`, set `nballs=2` (uses level `(1-α)^(1/2)`).
"""
function wasserstein_clt(z::AbstractVector{<:Real};
                         alpha::Real,
                         t_grid::Union{Nothing,AbstractVector{<:Real}}=nothing,
                         B::Integer=2000,
                         rng::AbstractRNG=Random.default_rng(),
                         nballs::Integer=1)::Float64

    n = length(z)
    n == 0 && error("z must be non-empty")
    (0 < alpha < 1) || error("alpha must be in (0,1); got alpha=$(alpha)")
    B > 0 || error("B must be positive; got B=$(B)")
    nballs > 0 || error("nballs must be positive; got nballs=$(nballs)")

    # Default grid: bounded-support truncation to [min, max].
    if t_grid === nothing
        a = Float64(minimum(z))
        b = Float64(maximum(z))
        if a == b
            return 0.0
        end
        t_grid = collect(range(a, b, length=200))
    else
        t_grid = sort(Float64.(t_grid))
        length(t_grid) >= 2 || error("t_grid must have length >= 2")
    end

    # Empirical CDF on the grid (sorted lookup).
    z_sorted = sort(Float64.(z))
    m = length(t_grid)
    Fhat = Vector{Float64}(undef, m)
    for (k, t) in enumerate(t_grid)
        Fhat[k] = searchsortedlast(z_sorted, Float64(t)) / n
    end
    Δt = diff(t_grid)

    return _wasserstein_clt_from_cdf(Fhat, Δt, n;
                                    alpha=alpha,
                                    B=B,
                                    rng=rng,
                                    nballs=nballs)
end

"""Internal helper: CLT radius using already-computed empirical CDF values and grid spacings."""
function _wasserstein_clt_from_cdf(Fhat::AbstractVector{<:Real},
                                  Δt::AbstractVector{<:Real},
                                  n::Integer;
                                  alpha::Real,
                                  B::Integer,
                                  rng::AbstractRNG,
                                  nballs::Integer)

    n > 0 || error("n must be positive; got n=$(n)")
    (0 < alpha < 1) || error("alpha must be in (0,1); got alpha=$(alpha)")
    B > 0 || error("B must be positive; got B=$(B)")
    nballs > 0 || error("nballs must be positive; got nballs=$(nballs)")

    m = length(Fhat)
    m >= 2 || error("Fhat must have length >= 2")
    length(Δt) == m - 1 || error("Δt must have length length(Fhat)-1")

    # Match the LP discretization: use t_1,...,t_{m-1} (left endpoints).
    r = m - 1
    F = Float64.(Fhat[1:r])
    w = Float64.(Δt)

    # Plug-in Brownian bridge covariance on the grid:
    #   Σ_{ij} = F(min(t_i,t_j)) - F(t_i)F(t_j).
    # Since t_grid is ordered, F is nondecreasing and min(t_i,t_j) corresponds to min(i,j).
    Σ = Matrix{Float64}(undef, r, r)
    for i in 1:r
        Fi = F[i]
        for j in i:r
            Σij = Fi - Fi * F[j]     # min(i,j)=i because j>=i
            Σ[i, j] = Σij
            Σ[j, i] = Σij
        end
    end

    # Eigen-based sampling works for PSD (possibly singular) covariance matrices.
    eig = eigen(Symmetric(Σ))
    vals = eig.values
    vecs = eig.vectors

    # Clamp small negative eigenvalues due to roundoff.
    vals = max.(vals, 0.0)

    # Draw B samples: Y = V * sqrt(Λ) * Z, Z ~ N(0, I).
    Z = randn(rng, r, B)
    Y = vecs * (Diagonal(sqrt.(vals)) * Z)

    draws = Vector{Float64}(undef, B)
    @inbounds for b in 1:B
        # left Riemann sum: Σ |G(t_k)| Δt_k
        col = view(Y, :, b)
        s = 0.0
        for k in 1:r
            s += abs(col[k]) * w[k]
        end
        draws[b] = s
    end

    level = (1 - alpha)^(1 / nballs)
    q = quantile(draws, level)

    return q / sqrt(n)
end

# ---------------------------------------------------------------------------
# Smooth / regularized Wasserstein-1 CLT radius
# ---------------------------------------------------------------------------

"""
    wasserstein_smooth_clt(z; alpha, t_grid, smooth_sigma, kernel=:uniform, B=2000, rng=Random.default_rng(), nballs=1)

CLT-calibrated radius for the **smooth (kernel-regularized) Wasserstein-1** distance

    W₁^σ(μ, ν) := W₁(μ * η_σ, ν * η_σ),

where `η_σ` is a compactly supported kernel (currently only `:uniform`).

For 1D measures, the distance is

    W₁(μ * η_σ, ν * η_σ) = ∫ |F_σ(t) - G_σ(t)| dt,

with `F_σ` and `G_σ` the CDFs of the convolved measures.

Let `μ` be the true distribution of `Z` and `μ̂_n` the empirical measure.
The convolved empirical CDF satisfies

    (μ̂_n * η_σ)((-∞, t]) = (1/n) Σ_i H_σ(t - Z_i),

so by a functional CLT,

    √n · (F̂_σ(t) - F_σ(t)) ⇒ G_σ(t),

where `G_σ` is a centered Gaussian process with covariance

    Cov(G_σ(s), G_σ(t)) = Cov(H_σ(s - Z), H_σ(t - Z)).

We approximate the limiting distribution of

    √n · W₁(μ̂_n * η_σ, μ * η_σ)

by simulating a multivariate normal draw of `(G_σ(t_k))_{k=1}^{m-1}` on the
*left endpoints* of `t_grid`, then applying the same left-Riemann discretization
used in the LP constraints.

This provides an asymptotic `1-α` radius `ρ = q / √n`.
"""
function wasserstein_smooth_clt(z::AbstractVector{<:Real};
                               alpha::Real,
                               t_grid::AbstractVector{<:Real},
                               smooth_sigma::Real,
                               kernel::Symbol=:uniform,
                               B::Integer=2000,
                               rng::AbstractRNG=Random.default_rng(),
                               nballs::Integer=1)::Float64

    n = length(z)
    n == 0 && error("z must be non-empty")
    (0 < alpha < 1) || error("alpha must be in (0,1); got alpha=$(alpha)")
    B > 0 || error("B must be positive; got B=$(B)")
    nballs > 0 || error("nballs must be positive; got nballs=$(nballs)")

    t = sort(Float64.(t_grid))
    length(t) >= 2 || error("t_grid must have length >= 2")
    Δt = diff(t)

    σ = Float64(smooth_sigma)
    σ > 0 || error("smooth_sigma must be > 0; got smooth_sigma=$(smooth_sigma)")

    # Match LP discretization: use left endpoints t[1:end-1].
    r = length(t) - 1
    w = Float64.(Δt)

    # Build matrix G[k, i] = H_σ(t_k - z_i) for k=1:r.
    zf = Float64.(z)
    G = Matrix{Float64}(undef, r, n)

    if kernel == :uniform
        @inbounds for k in 1:r
            tk = t[k]
            for i in 1:n
                u = tk - zf[i]
                if u <= -σ
                    G[k, i] = 0.0
                elseif u >= σ
                    G[k, i] = 1.0
                else
                    G[k, i] = (u + σ) / (2σ)
                end
            end
        end
    else
        @inbounds for k in 1:r
            tk = t[k]
            for i in 1:n
                G[k, i] = kernel_cdf(tk - zf[i]; sigma=σ, kernel=kernel)
            end
        end
    end

    # Plug-in covariance for the Gaussian limit: Σ = E[g gᵀ] - (E[g])(E[g])ᵀ.
    # Use the empirical analog: (1/n) GGᵀ - m mᵀ.
    mvec = vec(sum(G, dims=2)) ./ n
    Σ = (G * transpose(G)) ./ n .- (mvec * transpose(mvec))

    # Sample from N(0, Σ) using eigen decomposition (Σ may be singular).
    eig = eigen(Symmetric(Σ))
    vals = max.(eig.values, 0.0)
    vecs = eig.vectors

    Z = randn(rng, r, B)
    Y = vecs * (Diagonal(sqrt.(vals)) * Z)

    draws = Vector{Float64}(undef, B)
    @inbounds for b in 1:B
        col = view(Y, :, b)
        s = 0.0
        for k in 1:r
            s += abs(col[k]) * w[k]
        end
        draws[b] = s
    end

    level = (1 - alpha)^(1 / nballs)
    q = quantile(draws, level)
    return q / sqrt(n)
end

# Chi-square radius ---------------------------------------------------------

"""
Chi-square localization radius (asymptotic) for a multinomial support of size m.

Default choice:
    r = quantile(Chisq(m-1), 1-alpha) / n

This corresponds to an approximate chi-square confidence set for the pmf.
"""
function chi2_radius(; alpha::Real, n::Integer, m::Integer)::Float64
    alpha <= 0 && error("alpha must be in (0,1); got alpha=$(alpha)")
    n <= 0 && error("n must be positive; got n=$(n)")
    m <= 1 && error("m must be >= 2; got m=$(m)")
    q = quantile(Chisq(m - 1), 1 - alpha)
    return q / n
end
