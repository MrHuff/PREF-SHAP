import numpy as np
import torch
from GPGP.KRR import *
from utils.utils import *
from GPGP.kernel import *
from pref_shap.pref_shap import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set()

if __name__ == '__main__':
    n=1000
    d=10
    data = load_data('toy_data',f'toy_mvp_{n}_{d}.pickle')
    x,x_prime,y = data['X'],data['X_prime'],data['Y']
    fixed_ls = torch.tensor([1.,1.] + [1.] * (d - 2)).float().cuda()
    k=GPGP_kernel(u=x,u_prime=x_prime).cuda()
    k.set_ls(fixed_ls)
    K=k.evaluate()
    krr=base_KRR(K,y,lambd=1e-3)
    alpha=krr.fit()
    _,r2=krr.predict(y=y.cuda())
    print(r2)


    a = torch.unique(x,dim=0)
    b = torch.unique(x_prime,dim=0)
    X = torch.cat([a,b],dim=0)
    inner_kernel = k.kernel
    ps = pref_shap(alpha=alpha,k=inner_kernel,X_l=x,X_r=x_prime,X=X,max_S=2500,rff_mode=False,eps=1e-6,cg_max_its=10,lamb=1e-2,max_inv_row=0,cg_bs=20,device='cuda:0')
    num_features = 100
    output = ps.fit(x[0:num_features,:],x_prime[0:num_features,:])
    tmp = output.cpu().numpy().flatten()
    tst= np.arange(1,d+1).repeat(num_features)



    # print(output.shape)
    # penguins = sns.load_dataset("penguins")
    # print(penguins.head())

    plot =  pd.DataFrame(np.stack([tst,tmp],axis=1),columns=['d','shapley_vals'])


    sns.displot(plot,x="shapley_vals", hue="d", kind="kde")
    plt.ylim(0,0.05 )
    plt.show()



    # print(output.abs().median(1))
    # print(output.abs().mean(1))

    # print(alpha)