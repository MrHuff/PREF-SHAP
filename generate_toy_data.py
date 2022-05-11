from data_generation.data_generation_alan_method import generate_x as generate_alan
import pickle
import os
from data_generation.data_generation import *
from data_generation.data_generation_cluster import *
import numpy as np


def save_files(job_name,l,r,predictors,y):
    with open(f'{job_name}/l_processed.npy', 'wb') as f:
        np.save(f, l)
    with open(f'{job_name}/r_processed.npy', 'wb') as f:
        np.save(f, r)
    with open(f'{job_name}/S.npy', 'wb') as f:
        np.save(f, predictors)
    with open(f'{job_name}/y.npy', 'wb') as f:
        np.save(f, y)


if __name__ == '__main__':
    n_pairs = 5000
    n_samples = 1000
    d = 10
    num_latent_states = 10
    d_imp = 2

    # job_name =f'alan_data_{n_pairs}_{n_samples}_{d}_{num_latent_states}'
    # if not os.path.exists(job_name):
    #     os.makedirs(job_name)
    # u, u_prime, y ,state,S= generate_alan(n_pairs=n_pairs, n_samples=n_samples, d=d, num_latent_states=num_latent_states,in_choice=[0,1])
    # y=torch.where(y<=0.0,-1,1).float().squeeze()
    # save_files(job_name,u.numpy(),u_prime.numpy(),S,y.numpy())

    job_name =f'toy_data_{n_pairs}_{d}_{d_imp}'
    if not os.path.exists(job_name):
        os.makedirs(job_name)
    u, u_prime,S = generate_mvp_X(n_pairs, d=d, d_imp=d_imp)
    fixed_ls=torch.tensor([1.0]*d_imp+[1.]*(d-d_imp)).float()
    y =GPGP_mvp_krr(u,u_prime,fixed_ls).squeeze()
    save_files(job_name,u.numpy(),u_prime.numpy(),S.numpy(),y.numpy())


    n_pairs = 5000
    n_samples = 100
    job_name =f'alan_data_{n_pairs}_{n_samples}'
    if not os.path.exists(job_name):
        os.makedirs(job_name)
    in_choice = list(range(2))
    u, u_prime, y ,S= generate_clusters()
    save_files(job_name,u,u_prime,S,y)



    #sort/filter on specific combinations, do local shap, expect state specific covariates to flare-up


