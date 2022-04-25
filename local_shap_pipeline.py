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

def return_feature_names(job,d):
    if job in ['pokemon','pokemon_wl']:
        l2=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary',
       'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire',
       'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison',
       'Psychic', 'Rock', 'Steel', 'Water']
        coeffs= 10**np.linspace(-7,-2,0)
        return [],l2,False,coeffs

    #Fire 13- Grass 16
    # Fire - Ice 18
    # Fire - Bug 7
    # Fire - Steel 23
    # Blastoise vs Charizard
    if job in [f'hard_data_10000_1000_{d[0]}_{d[1]}']:
        l2 = [f'important_{i}' for i in range(1,3)] + [f'Unimportant_{i}' for i in range(3,d[0]+1)]
        coeffs= 10**np.linspace(-7,-2,0)
        return [],l2,False,coeffs


def cumsum_thingy(cumsum_indices,shapley_vals):
    cat_parts = []
    for i in range(len(cumsum_indices)-1):
        part = shapley_vals[cumsum_indices[i]:cumsum_indices[i+1],:].sum(0,keepdim=True)
        cat_parts.append(part)
    p_output = torch.cat(cat_parts,dim=0)
    return p_output

def get_shapley_vals(job,model,fold,post_method,interventional,shap_l,shap_r):
    with open( f'{job}_results/{model}/run_{fold}.pickle' , 'rb') as handle:
        loaded_model = pickle.load(handle)
    best_model = dill.loads(loaded_model)
    ls = best_model['ls']
    alpha = best_model['alpha'].float()
    ind_points = best_model['inducing_points'].float()
    model = best_model['model'].to('cuda:0')

    x_ind_l,x_ind_r  = torch.chunk(ind_points,dim=1,chunks=2)
    inner_kernel=RBF_multiple_ls(d=x_ind_l.shape[1])
    inner_kernel._set_lengthscale(ls)
    inner_kernel=inner_kernel.to('cuda:0')
    alpha=alpha.to('cuda:0')

    #Filter func...
    c = train_GP(train_params=train_params)
    c.load_and_split_data()
    ps = pref_shap(model=model, alpha=alpha, k=inner_kernel, X_l=x_ind_l, X_r=x_ind_r, X=c.S, max_S=2500,
                   rff_mode=False, eps=1e-6, cg_max_its=25, lamb=1e-4, max_inv_row=0, cg_bs=50, post_method=post_method,
                   interventional=interventional, device='cuda:0')

    # num_matches = 100
    Y_target, weights,Z= ps.fit(shap_l,shap_r)

    cooking_dict = {'Y':Y_target.cpu(), 'weights':weights.cpu(),'Z':Z.cpu(),
                    'n':shap_l.shape[0]}

    return cooking_dict


def get_shapley_vals_2(cooking_dict,job,post_method,d):
    sum_count,features_names,do_sum,coeffs=return_feature_names(job,d)
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

def hard_data_get_vals(train_params,d):
    c = train_GP(train_params=train_params)
    c.load_and_split_data()
    if train_params['dataset']!='pokemon_wl':
        state = np.load(train_params['dataset'] + '/state.npy', allow_pickle=True)
        train_state = state[c.tr_ind,:]
        mask = np.isin(train_state,[0,1]).prod(axis=1)==1
        l2 = [f'important_{i}' for i in range(1,3)] + [f'Unimportant_{i}' for i in range(3,d[0]+1)]

    else:
        l2 = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary',
              'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire',
              'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison',
              'Psychic', 'Rock', 'Steel', 'Water']

        left_mask = (c.left_tr[:,d[0]]>0) |  (c.left_tr[:,d[1]]>0)
        right_mask = (c.right_tr[:,d[1]]>0) | (c.right_tr[:,d[0]]>0)
        type_idx = [i for i in range(7,(len(l2))) ]
        type_idx.remove(d[0])
        type_idx.remove(d[1])

        for t in type_idx:
            left_mask = left_mask & (c.left_tr[:, t]<=0)
            right_mask = right_mask & (c.right_tr[:, t]<=0)
        # fire_grass_mask = fire_mask & (c.left_tr[:,16]>0)
        # fire_grass_ice = fire_mask & (c.left_tr[:,18]>0)
        # fire_grass_bug = fire_mask & (c.left_tr[:,7]>0)
        # fire_grass_steel = fire_mask & (c.left_tr[:,23]>0)
        mask = left_mask & right_mask
        # mask = fire_grass_mask | fire_grass_ice | fire_grass_steel | fire_grass_bug
    y = (c.y_tr[mask]>0)[:,np.newaxis]*1.0
    shap_l = c.left_tr[mask,:]
    shap_r = c.right_tr[mask,:]
    left,right = torch.from_numpy(c.left_tr[mask,:]).float(),torch.from_numpy(c.right_tr[mask,]).float()

    winners = y * shap_r + (1 - y) * shap_l
    loosers = (1 - y) * shap_r + y * shap_l
    dat_abs = pd.DataFrame(winners-loosers,columns=l2)
    dat_abs['fold'] = train_params['fold']
    # return left[:50,:],right[:50,:],dat_abs.iloc[:50,:]
    return left,right,dat_abs


if __name__ == '__main__':
    # d=[13,24] #water v fire
    # d=[17,10] #electric v ground
    # d=[12,19] #Normal, Fightning
    d=[18,14] #Electric vs Flying

    # for job in [f'hard_data_10000_1000_{d[0]}_{d[1]}']:
    for job in ['pokemon_wl']:
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
            'm_factor':2.0,
            'seed': 42,
            'folds': 10,
        }
        if not os.path.exists(f'local_{job}'):
            os.makedirs(f'local_{job}')
        for f in [0]:
            left,right,dat_abs = hard_data_get_vals(train_params,d)
            cooking_dict = get_shapley_vals(job=job,model=model,fold=fold,post_method='OLS',
                                            interventional=interventional,
                                            shap_l=left,
                                            shap_r=right
                                            )
            torch.save(cooking_dict,f'local_{job}/cooking_dict_{f}.pt')
            dat_abs.to_csv(f'local_{job}/data_folds.csv')
    # for job in ['pokemon_wl','hard_data_10000_1000_10_10']:
        for post_method in ['lasso']:
            big_plt = []
            for f in [0]:
                cooking_dict = torch.load(f'local_{job}/cooking_dict_{f}.pt')
                data,features_names= get_shapley_vals_2(cooking_dict,job,post_method,d)
                data['fold']=f
                big_plt.append(data)
            plot= pd.concat(big_plt,axis=0).reset_index(drop=True)
            plot['d'] = plot['d'].apply(lambda x: features_names[int(x-1)])
            plot.to_csv(f'local_{job}/local_{job}_{post_method}.csv')


