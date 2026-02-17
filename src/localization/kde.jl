# Gaussian KDE + Gauss-F bootstrap radius -----------------------------------

"""
Gaussian kernel density estimate evaluated on `x_grid`.

Uses:
    fhat(x) = (1/(n*h)) * sum_i ϕ((x - z_i)/h)
where ϕ is the standard normal pdf.
"""
function gaussian_kde(z::AbstractVector{<:Real},
                      x_grid::AbstractVector{<:Real},
                      bandwidth::Real)::Vector{Float64}
    n = length(z)
    n == 0 && error("z must be non-empty")
    h = Float64(bandwidth)
    h <= 0 && error("bandwidth must be positive; got bandwidth=$(bandwidth)")

    zf = Float64.(z)
    inv_nh = 1.0 / (n * h)
    inv_sqrt2pi = 0.3989422804014327  # 1/sqrt(2π)

    out = Vector{Float64}(undef, length(x_grid))
    for (k, x) in enumerate(x_grid)
        s = 0.0
        for zi in zf
            u = (Float64(x) - zi) / h
            s += exp(-0.5 * u * u)
        end
        out[k] = inv_nh * inv_sqrt2pi * s
    end
    return out
end

"""
Bootstrap-calibrated radius for the Gauss-F localization constraint:

    max_{x in x_grid} |f(x) - fhat(x)| <= c

We bootstrap the KDE on `x_grid` and take the (1-alpha)-quantile of the
sup-norm deviation.
"""
function gauss_radius_bootstrap(z::AbstractVector{<:Real},
                               x_grid::AbstractVector{<:Real},
                               bandwidth::Real;
                               alpha::Real,
                               B::Integer=2000,
                               rng::AbstractRNG=Random.default_rng(),
                               fhat::Union{Nothing,Vector{Float64}}=nothing)::Float64
    n = length(z)
    n == 0 && error("z must be non-empty")
    alpha <= 0 && error("alpha must be in (0,1); got alpha=$(alpha)")
    B <= 0 && error("B must be positive; got B=$(B)")

    zf = Float64.(z)
    fhat0 = fhat === nothing ? gaussian_kde(zf, x_grid, bandwidth) : fhat

    deltas = Vector{Float64}(undef, B)
    for b in 1:B
        idx = rand(rng, 1:n, n)
        z_star = zf[idx]
        fhat_star = gaussian_kde(z_star, x_grid, bandwidth)
        deltas[b] = maximum(abs.(fhat_star .- fhat0))
    end
    return quantile(deltas, 1 - alpha)
end
