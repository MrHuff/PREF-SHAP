import numpy as np
import torch
from utils.utils import *
from pref_shap.pref_shap import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
sns.set()

def return_feature_names(job):
    if job=='pokemon':
        l1=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary',
       'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire',
       'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison',
       'Psychic', 'Rock', 'Steel', 'Water']
        l2=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary','Type']
        return l2


if __name__ == '__main__':
    # d_imp = 2
    # d=10
    # palette =['r']*d_imp+ ['g']*(d-d_imp)
    job='pokemon'
    model='krr_GPGP'
    fold=0
    train_params={
        'dataset':job,
        'fold':fold,
        'epochs':100,
        'patience':5,
        'model_string':'krr_GPGP', #krr_vanilla
        'bs':1000
    }
    with open( f'{job}_results/{model}/run_{fold}.pickle' , 'rb') as handle:
        best_model = pickle.load(handle)
    ls = best_model['ls']
    alpha = best_model['alpha'].float()
    ind_points = best_model['inducing_points'].float()

    x_ind_l,x_ind_r  = torch.chunk(ind_points,dim=1,chunks=2)
    c=train_GP(train_params=train_params,m=1000)
    c.load_and_split_data()

    inner_kernel=RBF_multiple_ls(d=x_ind_l.shape[1])
    inner_kernel._set_lengthscale(ls)
    inner_kernel=inner_kernel.to('cuda:0')
    alpha=alpha.to('cuda:0')
    ps = pref_shap(alpha=alpha,k=inner_kernel,X_l=x_ind_l,X_r=x_ind_r,X=c.S,max_S=2500,rff_mode=False,eps=1e-3,cg_max_its=10,lamb=1e-3,max_inv_row=0,cg_bs=25,device='cuda:0')

    x,x_prime = torch.from_numpy(c.left_test).float(),torch.from_numpy(c.right_test).float()
    num_features = 500
    output = ps.fit(x[0:num_features,:],x_prime[0:num_features,:])

    part_one = output[:7,:]
    part_two = output[7:,:].sum(0,keepdim=True)
    p_output = torch.cat([part_one,part_two],dim=0)
    tmp = p_output.cpu().numpy().flatten()
    features_names=return_feature_names(job)
    tst= np.arange(1,len(features_names)+1).repeat(num_features)



    plot =  pd.DataFrame(np.stack([tst,tmp],axis=1),columns=['d','shapley_vals'])
    plot['d'] = plot['d'].apply(lambda x: features_names[int(x-1)])
    sns.displot(plot,x="shapley_vals", hue="d", kind="kde")
    plt.show()
    plot =  pd.DataFrame(np.abs(np.stack([tst,tmp],axis=1)),columns=['d','shapley_vals'])
    plot['d'] = plot['d'].apply(lambda x: features_names[int(x-1)])
    sns.displot(plot,x="shapley_vals", hue="d", kind="kde")
    plt.show()

