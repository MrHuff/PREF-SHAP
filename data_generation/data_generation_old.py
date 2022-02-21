import numpy as np
import pandas as pd


def experiment0(n, p, n_states, shift=1):
    """
    Args:
        n: number of items to be compared
        p: number of features for each item
        n_states: number of latent states determining the "type"

    Returns:
    """
    # Generate features
    mean_vec = np.zeros(p)
    # Generate states
    latent_states = np.random.choice(n_states, n)

    num_per_clus = [(latent_states==i).sum() for i in range(n_states)]
    X_per_c = [np.random.multivariate_normal(mean_vec + i*shift, np.eye(p), num_per_clus[i]) for i in range(n_states)]

    X = pd.DataFrame(np.zeros((n, p)))
    for i in range(n_states):
        X.iloc[latent_states==i, :] = X_per_c[i]
    X = np.array(X)

    # Generate different consideration factors
    beta_dict = dict()
    for i in range(n_states):
        for j in range(i, n_states):
            beta = np.random.normal(size=p)
            beta_dict[(i, j)] = beta
            beta_dict[(j, i)] = beta

    return X, beta_dict, latent_states


"""
Simulation on Clusters of clusterable items
"""

def experiment1(n, p, n_clusters, shift=1):
    """
    Args:
        n: number of items to be compared
        p: number of features for each item
        n_clusters: number of clusters within
    Returns:
        simulations
    """
    # Generate features
    mean_vec = np.zeros(p)
    # Generate states
    z = np.random.choice(n_clusters, n)
    n_c0 = (z==0).sum()
    n_c1 = (z==1).sum()
    X_c0 = np.random.multivariate_normal(mean_vec, np.eye(p), n_c0)
    X_c1 = np.random.multivariate_normal(mean_vec + shift, np.eye(p), n_c1)
    X = pd.DataFrame(np.zeros((n, p)))
    X.iloc[z==0, :] = X_c0
    X.iloc[z==1, :] = X_c1
    X = np.array(X)
    # Generate different consideration factors
    beta_dict = dict()
    for i in range(n_clusters):
        for j in range(i, n_clusters):
            beta = np.random.normal(size=p)
            beta_dict[(i, j)] = beta
            beta_dict[(j, i)] = beta
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            if z[i] == z[j]:
                skill_i, skill_j = X[[i, j], :] @ beta_dict[(z[i], z[j])]
                if skill_i >= skill_j:
                    C[i, j] = 1
                    C[j, i] = -1
                else:
                    C[i, j] = -1
                    C[j, i] = 1
            else:
                # Random outcome if not in the same clusters
                C[i, j] = np.random.choice([-1, 1])
                C[j, i] = -C[i, j]
    return C, X, beta_dict, z

def generate_match(x, beta_dict, z):
    """

    Args:
        x: features of size n x p
        beta_dict: dictionary of "size" zC2, all possible latent states combination
        z: latent states vector

    Returns:

    """
    n = x.shape[0]
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            skill_i, skill_j = x[[i, j], :] @ beta_dict[(z[i], z[j])]
            if skill_i >= skill_j:
                C[i, j] = 1
                C[j, i] = -1
            else:
                C[i, j] = -1
                C[j, i] = 1

    return C
