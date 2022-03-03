from data_generation.data_generation_alan_method import *
import pickle
import os

if __name__ == '__main__':
    job_name = 'alan_data'
    n_pairs = 1000
    n_samples = 100
    d = 5
    num_latent_states = 5

    if not os.path.exists(job_name):
        os.makedirs(job_name)

    u, u_prime, y = generate_x(n_pairs, n_samples, d, num_latent_states)

    with  open(f'{job_name}/toy_mvp_{n}_{d}.pickle', 'wb') as handle:
        pickle.dump({ 'X':u,'X_prime':u_prime,'Y':y}, handle, protocol=pickle.HIGHEST_PROTOCOL)