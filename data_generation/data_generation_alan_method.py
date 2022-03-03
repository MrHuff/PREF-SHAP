import numpy as np
import torch


def generate_x(n_pairs, n_samples, d=10, num_latent_states=3):

    if num_latent_states > d:
        raise ValueError("Due to experimental design, # of states has to be < d.")

    X = torch.randn(n_samples, d)
    latent_state_ls = np.random.choice(range(num_latent_states),
                                    replace=True,
                                    size=n_samples
                                    )

    # set up the comparison function based on features and latent states.
    # if items are from cluster 1 and 2, then the linear preference will have
    # linear weights 1 on feature 1 and 2.

    linear_weights = torch.zeros(size=(10, 1))

    # Generate match indicies
    match_indicies = []

    while len(match_indicies) < n_pairs:
        candidates = list(np.random.choice(range(n_samples), size=2, replace=False))
        candidates.sort()

        if candidates not in match_indicies:
            match_indicies.append(candidates)

    match_indicies = np.array(match_indicies)
    x_left = X[match_indicies[:, 0], :]
    x_right = X[match_indicies[:, 1], :]
    y_ls = []

    # now generate matches
    for i, match in enumerate(match_indicies):
        state_a, state_b = latent_state_ls[match[0]], latent_state_ls[match[1]]
        weights = linear_weights
        weights[state_a], weights[state_b] = 1, 1

        difference = x_left[i, :]@weights - x_right[i, :]@weights
        if difference > 0:
            # left wins
            y_ls.append(1)
        else:
            y_ls.append(0)

    return x_left.float(), x_right.float(), torch.tensor(y_ls).float()






