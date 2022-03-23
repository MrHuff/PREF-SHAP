from data_generation.data_generation import *
# from data_generation.data_generation_alan_method import *
import pickle
import os


#TODO rebuild this MVP pipeline... tailored for FALKON...

if __name__ == '__main__':
    job_name = 'toy_data'
    n=5000
    d = 10
    d_imp=2
    if not os.path.exists(job_name):
        os.makedirs(job_name)
    u,u_prime,S = generate_mvp_X(n,d=d,d_imp=d_imp)
    fixed_ls=torch.tensor([1.]*d_imp+[1.]*(d-d_imp)).float()
    y =GPGP_mvp_krr(u,u_prime,fixed_ls)
    # u,u_prime,y = generate_x(n_pairs=n,n_samples=500,d=10)
    with  open(f'{job_name}/toy_mvp_{n}_{d}.pickle', 'wb') as handle:
        pickle.dump({ 'X':u,'X_prime':u_prime,'Y':y}, handle, protocol=pickle.HIGHEST_PROTOCOL)
