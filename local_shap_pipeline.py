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
    if job == 'alan_data_5000_100':
        D = 3
        a = np.zeros((D, D))
        zip1, zip2 = np.triu_indices(D, 1)

        l2 = [r'$x^{[0]}$'] + [r'$x^{AB}$',r'$x^{AC}$',r'$x^{BC}$'] + ['in A','in B','in C']

        coeffs = 10 ** np.linspace(-7, -2, 0)
        return [], l2, False, coeffs

    if job =='alan_data_5000_1000':
        D=3
        a = np.zeros((D, D))
        zip1,zip2 = np.triu_indices(D, 1)

        l2 = ['within_cluster'] + [f'feature_{i}_{j}' for (i,j) in zip(zip1,zip2)] + [f'indicator']
        coeffs= 10**np.linspace(-7,-2,0)
        return [],l2,False,coeffs

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

def get_shapley_vals(job,model_string,fold,post_method,interventional,shap_l,shap_r):
    with open( f'{job}_results/{model_string}/run_{fold}.pickle' , 'rb') as handle:
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
                   rff_mode=False, eps=1e-6, cg_max_its=25, lamb=1e-4, max_inv_row=0, cg_bs=5, post_method=post_method,
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
    left_all_data = np.concatenate([c.left_tr, c.left_val, c.left_test], axis=0)
    right_all_data = np.concatenate([c.right_tr, c.right_val, c.right_test], axis=0)
    y = np.concatenate([c.y_tr, c.y_val, c.y_test], axis=0)

    if d in ['pokemon_squirtle_wl']:
        l2 = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary',
              'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire',
              'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison',
              'Psychic', 'Rock', 'Steel', 'Water']
        shap_l,shap_r,y = load_player(d)
        y = y[:,np.newaxis]
        winners = y * shap_r + (1 - y) * shap_l
        loosers = (1 - y) * shap_r + y * shap_l
        dat_abs = pd.DataFrame(winners - loosers, columns=l2)
        dat_abs['fold'] = train_params['fold']
        return torch.from_numpy(shap_l).float(), torch.from_numpy(shap_r).float(), dat_abs

    if train_params['dataset']=='alan_data_5000_100':

        left_mask = left_all_data[:,4+d[0]]>0
        right_mask = right_all_data[:,4+d[1]]>0
        mask_1 = left_mask & right_mask
        left_mask = left_all_data[:, 4 + d[1]] > 0
        right_mask = right_all_data[:, 4 + d[0]] > 0
        mask_2 = left_mask & right_mask
        mask=mask_1 | mask_2
        D = 3
        a = np.zeros((D, D))
        zip1, zip2 = np.triu_indices(D, 1)
        l2 = [r'$x^{[0]}$'] + [r'$x^{AB}$',r'$x^{AC}$',r'$x^{BC}$'] + ['in A','in B','in C']
    elif train_params['dataset'] == 'alan_data_5000_1000':
        elements = np.unique(left_all_data[:, 4]).tolist()
        left_mask = np.round(left_all_data[:, 4],2)==np.round(elements[d[0]],2)
        right_mask = np.round(right_all_data[:, 4],2)==np.round(elements[d[1]],2)
        mask_1 = left_mask | right_mask
        left_mask = np.round(left_all_data[:, 4],2)==np.round(elements[d[1]],2)
        right_mask = np.round(right_all_data[:, 4],2)==np.round(elements[d[0]],2)
        mask_2 = left_mask | right_mask

        mask = mask_1 | mask_2
        D = 3
        a = np.zeros((D, D))
        zip1, zip2 = np.triu_indices(D, 1)
        l2 = [r'$x^{[0]}$'] + [r'$x^{AB}$',r'$x^{AC}$',r'$x^{BC}$'] + ['in A','in B','in C']

    else:
        l2 = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary',
              'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire',
              'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison',
              'Psychic', 'Rock', 'Steel', 'Water']
        left_mask = (left_all_data[:,d[0]]>0)
        right_mask = (right_all_data[:,d[0]]>0)
        middle_mask = ((left_all_data[:,d[0]]>0) & (right_all_data[:,d[0]]>0))
        for d_ind in d[1:]:
            left_mask = left_mask | (left_all_data[:,d_ind]>0)
            right_mask = right_mask |  (right_all_data[:,d_ind]>0)
            middle_mask = middle_mask | ((left_all_data[:,d_ind]>0) & (right_all_data[:,d_ind]>0))

        # left_mask = (left_all_data[:,d[0]]>0) |  (left_all_data[:,d[1]]>0)
        # right_mask = (right_all_data[:,d[0]]>0) | (right_all_data[:,d[1]]>0)
        # middle_mask = ((left_all_data[:,d[0]]>0) & (right_all_data[:,d[0]]>0)) | ((left_all_data[:,d[1]]>0) & (right_all_data[:,d[1]]>0))
        # middle_mask = middle_mask

        type_idx = [i for i in range(7,(len(l2))) ]
        for d_ind in d:
            type_idx.remove(d_ind)

        for t in type_idx:
            left_mask = left_mask & (left_all_data[:, t]<=0)
            right_mask = right_mask & (right_all_data[:, t]<=0)
            middle_mask = middle_mask & ((left_all_data[:, t]<=0) | (right_all_data[:, t]<=0) )
        # fire_grass_mask = fire_mask & (left_all_data[:,16]>0)
        # fire_grass_ice = fire_mask & (left_all_data[:,18]>0)
        # fire_grass_bug = fire_mask & (left_all_data[:,7]>0)
        # fire_grass_steel = fire_mask & (left_all_data[:,23]>0)
        mask = left_mask & right_mask & (~middle_mask)
        # mask = fire_grass_mask | fire_grass_ice | fire_grass_steel | fire_grass_bug
    y = (y[mask]>0)[:,np.newaxis]*1.0

    shap_l = left_all_data[mask,:]
    shap_r = right_all_data[mask,:]
    left,right = torch.from_numpy(left_all_data[mask,:]).float(),torch.from_numpy(right_all_data[mask,]).float()

    winners = y * shap_r + (1 - y) * shap_l
    loosers = (1 - y) * shap_r + y * shap_l
    dat_abs = pd.DataFrame(winners-loosers,columns=l2)
    dat_abs['fold'] = train_params['fold']
    # return left[:50,:],right[:50,:],dat_abs.iloc[:50,:]
    return left,right,dat_abs
def load_player(job):
    dataset_string = f'{job}'
    l_load = np.load(dataset_string + '/l_processed.npy', allow_pickle=True)
    r_load = np.load(dataset_string + '/r_processed.npy', allow_pickle=True)
    y_load = np.load(dataset_string + '/y.npy', allow_pickle=True)
    return l_load,r_load,y_load

if __name__ == '__main__':
    # d=[13,24] #water v fire
    # d=[17,10] #electric v ground
    # d=[12,19] #Normal, Fightning
    # d=[18,14] #Electric vs Flying
    # d=[16,13] #Grass fire
    # for d in [[13, 24], [17, 10], [12, 19], [16, 13], [10, 14],[16, 24]]:#,[18,9],[18,14]]:
    for d in [[13, 24,15,12,9]]:#,[18,9],[18,14]]:
    # for d in ['pokemon_squirtle_wl']:
        for job in ['pokemon_wl']:
    # for d in [[0,0],[0,1],[1,1],[1,2],[0,2]]:
    #     for job in ['alan_data_5000_100']:
            for m in ['SGD_krr','SGD_krr_pgp']:
                interventional=False
                model=m
                fold=0
                train_params={
                    'dataset':job,
                    'fold':fold,
                    'epochs':100,
                    'patience':5,
                    'model_string':'SGD_krr_pgp', #krr_vanilla
                    'bs':1000,
                    'double_up':False,
                    'm_factor':2.0,
                    'seed': 42,
                    'folds': 10,
                }
                res_name = f'local_{job}_{d}_{model}'
                if not os.path.exists(f'{res_name}'):
                    os.makedirs(f'{res_name}')
                for f in [0]:
                    left,right,dat_abs = hard_data_get_vals(train_params,d)
                    cooking_dict = get_shapley_vals(job=job,model_string=model,fold=fold,post_method='OLS',
                                                    interventional=interventional,
                                                    shap_l=left,
                                                    shap_r=right
                                                    )
                    torch.save(cooking_dict,f'{res_name}/cooking_dict_{f}.pt')
                    dat_abs.to_csv(f'{res_name}/data_folds.csv')
                for post_method in ['lasso']:
                    big_plt = []
                    for f in [0]:
                        cooking_dict = torch.load(f'{res_name}/cooking_dict_{f}.pt')
                        data,features_names= get_shapley_vals_2(cooking_dict,job,post_method,d)
                        data['fold']=f
                        big_plt.append(data)
                    plot= pd.concat(big_plt,axis=0).reset_index(drop=True)
                    plot['d'] = plot['d'].apply(lambda x: features_names[int(x-1)])
                    plot.to_csv(f'{res_name}/{res_name}_{post_method}.csv')


