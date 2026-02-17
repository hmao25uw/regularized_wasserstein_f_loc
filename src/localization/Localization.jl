module Localization

using Random
using Statistics
using Distributions
using JuMP

export AbstractLocalization,
       DKWLocalization, GaussLocalization, WassersteinLocalization, Chi2Localization,
       dkw_epsilon, empirical_cdf, empirical_pmf,
       gaussian_kde, gauss_radius_bootstrap,
       # Smooth/regularized Wasserstein helpers
       kernel_cdf, smooth_empirical_cdf, smooth_cdf, smooth_cdf_mat,
       wasserstein_radius_dkw, wasserstein_radius_bootstrap, wasserstein_clt, wasserstein_smooth_clt,
       chi2_radius,
       prepare_localization_stats,
       add_constraints!

# Core types + helpers
include(joinpath(@__DIR__, "smooth_wasserstein.jl"))
include(joinpath(@__DIR__, "radii.jl"))
include(joinpath(@__DIR__, "kde.jl"))
include(joinpath(@__DIR__, "specs.jl"))
include(joinpath(@__DIR__, "constraints.jl"))

end # module
