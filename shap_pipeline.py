import os.path

import numpy as np
import torch
from utils.utils import *
from pref_shap.pref_shap import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import dill
sns.set()

def return_feature_names(job):
    if job in ['pokemon','pokemon_wl']:
        l1= [1,1,1,1,1,1,1,19]
        l1.insert(0,0)
        l1=np.cumsum(l1).tolist()
        # l1=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary',
       # 'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire',
       # 'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison',
       # 'Psychic', 'Rock', 'Steel', 'Water']
        l2=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary','Type']
        coeffs= 10**np.linspace(-7,-2,10)
        return l1,l2,True,coeffs



    if job in ['chameleon','chameleon_wl']:
        l2 = ['ch.res', 'jl.res', 'tl.res', 'mass.res', 'SVL', 'prop.main',
       'prop.patch']
        coeffs= 10**np.linspace(-7,-2,10)
        return [],l2,False,coeffs

def cumsum_thingy(cumsum_indices,shapley_vals):
    cat_parts = []
    for i in range(len(cumsum_indices)-1):
        part = shapley_vals[cumsum_indices[i]:cumsum_indices[i+1],:].sum(0,keepdim=True)
        cat_parts.append(part)
    p_output = torch.cat(cat_parts,dim=0)
    return p_output

def get_shapley_vals(job,model,fold,train_params,num_matches,post_method,interventional):
    with open( f'{job}_results/{model}/run_{fold}.pickle' , 'rb') as handle:
        loaded_model = pickle.load(handle)
    best_model = dill.loads(loaded_model)
    ls = best_model['ls']
    alpha = best_model['alpha'].float()
    ind_points = best_model['inducing_points'].float()
    model = best_model['model'].to('cuda:0')

    x_ind_l,x_ind_r  = torch.chunk(ind_points,dim=1,chunks=2)
    c= train_GP(train_params=train_params)
    c.load_and_split_data()

    inner_kernel=RBF_multiple_ls(d=x_ind_l.shape[1])
    inner_kernel._set_lengthscale(ls)
    inner_kernel=inner_kernel.to('cuda:0')
    alpha=alpha.to('cuda:0')
    x,x_prime = torch.from_numpy(c.left_val).float(),torch.from_numpy(c.right_val).float()
    y_pred  =model.predict(c.dataloader.dataset.val_X.to('cuda:0'))
    if x.shape[0]<100:
        x, x_prime = torch.from_numpy(c.left_tr).float(), torch.from_numpy(c.right_tr).float()
        y_pred  =model.predict(c.dataloader.dataset.train_X.to('cuda:0'))

    ps = pref_shap(model=model,y_pred=y_pred, alpha=alpha,k=inner_kernel,X_l=x_ind_l,X_r=x_ind_r,X=c.S,max_S=2500,rff_mode=False,
                   eps=1e-3,cg_max_its=10,lamb=1e-3,max_inv_row=0,cg_bs=25,post_method=post_method,device='cuda:0',interventional=interventional)

    # num_matches = 100
    shap_l,shap_r = x[0:num_matches, :], x_prime[0:num_matches, :]
    Y_target, weights,Z= ps.fit(shap_l,shap_r)

    cooking_dict = {'Y':Y_target.cpu(), 'weights':weights.cpu(),'Z':Z.cpu(),
                    'n':shap_l.shape[0]}

    return cooking_dict


def get_shapley_vals_2(cooking_dict,job,post_method):
    sum_count,features_names,do_sum,coeffs=return_feature_names(job)
    outputs = construct_values(cooking_dict['Y'],cooking_dict['Z'],
                              cooking_dict['weights'],coeffs,post_method
                              )
    big_plot=[]
    for key,output in outputs.items():
        if do_sum:
            p_output = cumsum_thingy(sum_count,output)
        else:
            p_output = output
        tmp = p_output.cpu().numpy().flatten()
        tst= np.arange(1,len(features_names)+1).repeat(cooking_dict['n'])
        plot =  pd.DataFrame(np.stack([tst,tmp,np.ones_like(tst)*key],axis=1),columns=['d','shapley_vals','lambda'])
        big_plot.append(plot)
    plot = pd.concat(big_plot,axis=0).reset_index()
    return plot,features_names
if __name__ == '__main__':
    # d_imp = 2
    # d=10
    # palette =['r']*d_imp+ ['g']*(d-d_imp)
    for job in ['chameleon_wl','pokemon_wl']:
    # for job in ['pokemon_wl']:
        interventional=False
        model='SGD_krr'
        fold=0
        train_params={
            'dataset':job,
            'fold':fold,
            'epochs':100,
            'patience':5,
            'model_string':'SGD_krr', #krr_vanilla
            'bs':1000,
            'double_up':False,
            'm_factor':1.0
        }
        shutil.rmtree(f'{interventional}_{job}')
        if not os.path.exists(f'{interventional}_{job}'):
            os.makedirs(f'{interventional}_{job}')
        for f in [0,1,2]:
            if not os.path.exists(f'{interventional}_{job}/cooking_dict_{f}.pt'):
                cooking_dict = get_shapley_vals(job=job,model=model,fold=fold,train_params=train_params,num_matches=-1,post_method='OLS',interventional=interventional)
                torch.save(cooking_dict,f'{interventional}_{job}/cooking_dict_{f}.pt')
    for job in ['chameleon_wl', 'pokemon_wl']:
    # for job in ['pokemon_wl']:
        for post_method in ['lasso', 'ridge', 'elastic']:
        # for post_method in ['lasso']:
            big_plt = []
            for f in [0, 1, 2]:
            # for f in [0]:
                cooking_dict = torch.load(f'{interventional}_{job}/cooking_dict_{f}.pt')
                data,features_names= get_shapley_vals_2(cooking_dict,job,post_method)
                data['fold']=f
                big_plt.append(data)
            plot= pd.concat(big_plt,axis=0).reset_index(drop=True)
            plot['d'] = plot['d'].apply(lambda x: features_names[int(x-1)])
            plot.to_csv(f'{interventional}_{job}/{interventional}_{job}_{post_method}.csv')


