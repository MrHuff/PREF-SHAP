import numpy as np
import torch

from GPGP.kernel import *

def generate_mvp_X(n_pairs,d=10,cluster_a_n=100,cluster_b_n=100,d_imp=2):
    important_feature_a = torch.randn(cluster_a_n,d_imp)*0.2
    important_feature_b = torch.randn(cluster_b_n,d_imp)*0.2

    #Try doing IID and lengthscaling properly if it doesn't work

    unimportant_feature_a = torch.ones(cluster_a_n,d-d_imp)
    unimportant_feature_b = torch.ones(cluster_b_n,d-d_imp)

    items_A = torch.cat([important_feature_a,unimportant_feature_a],dim=1)
    items_B = torch.cat([important_feature_b,unimportant_feature_b],dim=1)

    S=torch.cat([items_A,items_B],dim=0)


    indices_a = [i for i in range(cluster_a_n)]
    indices_b = [i for i in range(cluster_b_n)]

    U = []
    U_prime = []
    for p in range(n_pairs):
        a = np.random.choice(indices_a)
        b = np.random.choice(indices_b)
        U.append(items_A[a,:].unsqueeze(0))
        U_prime.append(items_B[b,:].unsqueeze(0))
    x,x_prime = torch.cat(U,dim=0),torch.cat(U_prime,dim=0)
    return x.float(),x_prime.float(),S

def GPGP_mvp_krr(x,x_prime,fixed_ls,task='classification'):
    n = x.shape[0]
    ref=1000 #signal propagation is important as well
    true_alpha = torch.cat([torch.randn(ref,1),torch.zeros(n-ref,1)],dim=0)
    GP_ker = GPGP_kernel(u=x,u_prime=x_prime)
    GP_ker.set_ls(fixed_ls)
    y_middle = (GP_ker.evaluate() + torch.eye(n)*1e-3) @ true_alpha
    if task=='regression':
        return y_middle.float()
    if task=='classification':
        # return y_middle.float()
        return torch.where(torch.sigmoid(y_middle)<0.5,-1,1).float()





