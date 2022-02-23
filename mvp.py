import torch

from GPGP.KRR import *
from utils.utils import *
from GPGP.kernel import *
from pref_shap.pref_shap import *
if __name__ == '__main__':
    n=1000
    d=10
    data = load_data('toy_data',f'toy_mvp_{n}_{d}.pickle')
    x,x_prime,y = data['X'],data['X_prime'],data['Y']
    k=GPGP_kernel(x,u_prime=x_prime).to('cuda:0')
    K=k.evaluate()
    inner_kernel = k.kernel
    krr=base_KRR(K,y)
    alpha=krr.fit()
    a = torch.unique(x,dim=0)
    b = torch.unique(x_prime,dim=0)
    X = torch.cat([a,b],dim=0)
    ps = pref_shap(alpha=alpha,k=inner_kernel,X_l=x,X_r=x_prime,X=X,max_S=2500,rff_mode=False,eps=1e-3,cg_max_its=10,lamb=1e-2,max_inv_row=0,cg_bs=20,device='cuda:0')
    num_features = 4
    output = ps.fit(x[0:num_features,:],x_prime[0:num_features,:])
    print(output)

    # print(alpha)