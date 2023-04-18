import numpy as np
import pymc3 as pm

def compare_arrays(array1, array2):
    with pm.Model() as model:
        # Priors for the location and scale parameters of the Laplace distributions
        loc1 = pm.Uniform("loc1", lower=array1.min(), upper=array1.max())
        loc2 = pm.Uniform("loc2", lower=array2.min(), upper=array2.max())
        scale1 = pm.HalfNormal("scale1", sigma=10)
        scale2 = pm.HalfNormal("scale2", sigma=10)

        # Likelihoods of the arrays given the location and scale parameters
        obs1 = pm.Laplace("obs1", mu=loc1, b=scale1, observed=array1)
        obs2 = pm.Laplace("obs2", mu=loc2, b=scale2, observed=array2)

        # Difference of location parameters
        loc_diff = pm.Deterministic("loc_diff", loc1 - loc2)

        # Bayesian estimation
        trace = pm.sample(2000, tune=1000)

    # Summary of the results
    summary = pm.summary(trace, hdi_prob=0.95)
    print(summary)

    # Probability that the arrays have the same distribution
    prob_same_loc = np.mean(np.abs(trace["loc_diff"]) <= 0.01)  # Adjust the threshold as needed
    print(f"Probability that the arrays have the same distribution: {prob_same_loc:.2f}")

    # Difference between the means (location parameters)
    mean_diff = np.mean(trace["loc_diff"])
    print(f"Difference between the means: {mean_diff:.2f}")

# Example usage
array1 = np.array([1, 2, 3, 4, 5])
array2 = np.array([6, 7, 8, 9, 10, 11])

compare_arrays(array1, array2)
