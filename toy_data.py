from data_generation.data_generation import *
import pickle
import os

if __name__ == '__main__':
    job_name = 'toy_data'
    n=1000
    d = 10
    if not os.path.exists(job_name):
        os.makedirs(job_name)
    u,u_prime = generate_mvp_X(n,d=d)
    y =GPGP_mvp_krr(u,u_prime)
    with  open(f'{job_name}/toy_mvp_{n}_{d}.pickle', 'wb') as handle:
        pickle.dump({ 'X':u,'X_prime':u_prime,'Y':y}, handle, protocol=pickle.HIGHEST_PROTOCOL)
