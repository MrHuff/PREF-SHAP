import numpy as np
import torch


def generate_x(n_pairs, n_samples, d=10, num_latent_states=3,in_choice=[0,1]):

    if num_latent_states > d:
        raise ValueError("Due to experimental design, # of states has to be < d.")

    X = torch.randn(n_samples, d)*0.1
    latent_state_ls = np.random.choice(in_choice, #[0,1]
                                    replace=True,
                                    size=n_samples
                                    )
    #[0.4,0.4,0.05,0.05....]

    # set up the comparison function based on features and latent states.
    # if items are from cluster 1 and 2, then the linear preference will have
    # linear weights 1 on feature 1 and 2.

    linear_weights = torch.zeros(size=(d, 1)) #vector

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
    state_vector = []
    # print(len(latent_state_ls))

    for i, match in enumerate(match_indicies):
        state_a, state_b = latent_state_ls[match[0]], latent_state_ls[match[1]]
        weights = linear_weights

        weights[state_a], weights[state_b] = 1., 1.

        difference =x_right[i, :]@weights-x_left[i, :]@weights
        # print(difference)
        # print(state_a,state_b)
        y_ls.append(difference.item())
        # if difference > 0:
        #     left wins
            # y_ls.append(1)
        # else:
        #     y_ls.append(-1)
        state_vector.append([state_a,state_b])

    return x_left.float(), x_right.float(), torch.tensor(y_ls).float().unsqueeze(-1), np.array(state_vector),X.numpy()






