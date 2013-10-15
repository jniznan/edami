import numpy as np
from pandas import DataFrame, Series


def generate_qmatrix(concepts, problems):
    """ Generates a Q-matrix of size "concepts x problems" with values
    sampled from a continuous uniform distribution between 0 and 1."""
    return np.random.rand(concepts, problems)


def generate_skills(concepts, students, sd=1):
    """ Generates a skill matrix of size "students x concepts" with values
    sampled from a normal distribution. """
    return np.random.normal(scale=sd, size=(students, concepts))


def generate_zscores(qmatrix, skills, noise=0.2, missing=0.0):
    """ Generates a Z-score matrix of shape "students x problems" by
    multiplying a skill matrix and a Q-matrix and adding Gaussian noise. """
    results = np.dot(skills, qmatrix)
    if noise > 0:
        results += np.random.normal(scale=noise, size=results.shape)
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            if np.random.rand() < missing:
                results[i, j] = np.NAN
    return results


############################## BASIC MODEL ###################################

def basic_model_generate_parameters(students, problems):
    S, P = students, problems
    a = Series(np.random.randn(P) * 0.4 - 1)
    b = Series(np.random.randn(P) * 2 + 7)
    c = Series(np.random.rand(P) + 0.1)
    theta = Series(np.random.randn(S) * 0.7)
    sigma = Series(np.random.rand(S) * 1)
    return dict(a=a, b=b, c=c, theta=theta, sigma=sigma)


def basic_model_generate_times(a, b, c, theta, sigma):
    S, P = len(theta), len(a)
    times = DataFrame(columns=a.index, index=theta.index)
    local_skill = DataFrame(np.random.randn(S, P),
                            columns=a.index, index=theta.index)
    local_skill = local_skill.apply(lambda problem: theta + problem * sigma)
    local_variance = DataFrame(np.random.randn(S, P),
                               columns=a.index, index=theta.index) * c
    times = b + local_skill * a + local_variance
    return times
