module Likelihoods

using Distributions

export gaussian_likelihood, binomial_likelihood, poisson_likelihood

"""
Gaussian location model:
    Z | μ ~ Normal(μ, σ)

Returns a NamedTuple with fields:
    pdf(z, μ), cdf(t, μ)
"""
function gaussian_likelihood(σ::Real)
    σf = Float64(σ)
    pdf_fun = (z, μ) -> pdf(Normal(Float64(μ), σf), Float64(z))
    cdf_fun = (t, μ) -> cdf(Normal(Float64(μ), σf), Float64(t))
    return (pdf=pdf_fun, cdf=cdf_fun)
end

"""
Binomial model:
    Z | μ ~ Binomial(N, μ),   μ in [0,1]

Returns pdf and cdf that coerce their first argument to Int as appropriate.
"""
function binomial_likelihood(N::Integer)
    Ni = Int(N)
    pdf_fun = (z, μ) -> pdf(Binomial(Ni, Float64(μ)), Int(round(z)))
    cdf_fun = (t, μ) -> cdf(Binomial(Ni, Float64(μ)), Int(floor(t)))
    return (pdf=pdf_fun, cdf=cdf_fun)
end

"""
Poisson model:
    Z | μ ~ Poisson(μ),   μ >= 0

Returns pdf and cdf that coerce their first argument to Int as appropriate.
"""
function poisson_likelihood()
    pdf_fun = (z, μ) -> pdf(Poisson(Float64(μ)), Int(round(z)))
    cdf_fun = (t, μ) -> cdf(Poisson(Float64(μ)), Int(floor(t)))
    return (pdf=pdf_fun, cdf=cdf_fun)
end

end # module
