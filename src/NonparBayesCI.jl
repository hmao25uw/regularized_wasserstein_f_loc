module NonparBayesCI

# Submodules (kept explicit to avoid circular imports)
include(joinpath(@__DIR__, "types.jl"))
include(joinpath(@__DIR__, "optimization", "solvers.jl"))
include(joinpath(@__DIR__, "localization", "Localization.jl"))
include(joinpath(@__DIR__, "likelihood", "likelihoods.jl"))
include(joinpath(@__DIR__, "methods", "f_localization.jl"))
include(joinpath(@__DIR__, "experiments", "experiments.jl"))

# Re-exports for convenience
using .Types: SolverConfig, PriorConstraints
using .Localization: AbstractLocalization,
    DKWLocalization, GaussLocalization, WassersteinLocalization, Chi2Localization,
    dkw_epsilon, empirical_cdf,
    gaussian_kde, gauss_radius_bootstrap,
    kernel_cdf, smooth_empirical_cdf, smooth_cdf, smooth_cdf_mat,
    wasserstein_radius_dkw, wasserstein_radius_bootstrap, wasserstein_clt, wasserstein_smooth_clt,
    prepare_localization_stats, add_constraints!
using .Likelihoods: gaussian_likelihood, binomial_likelihood, poisson_likelihood
using .Methods: EBProblem, f_localization_ci, f_localization_bound
using .Experiments: run_config

export SolverConfig, PriorConstraints
export AbstractLocalization, DKWLocalization, GaussLocalization, WassersteinLocalization, Chi2Localization
export dkw_epsilon, empirical_cdf
export gaussian_kde, gauss_radius_bootstrap
export kernel_cdf, smooth_empirical_cdf, smooth_cdf, smooth_cdf_mat
export wasserstein_radius_dkw, wasserstein_radius_bootstrap, wasserstein_clt, wasserstein_smooth_clt
export prepare_localization_stats
export EBProblem, f_localization_ci, f_localization_bound
export gaussian_likelihood, binomial_likelihood, poisson_likelihood
export run_config

end # module
