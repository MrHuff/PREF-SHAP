import os.path

import numpy as np
import torch
from utils.utils import *
from pref_shap.pref_shap import *
from pref_shap.rkhs_shap import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import dill
sns.set()

def cumsum_thingy_2(cumsum_indices,shapley_vals):
    if cumsum_indices:
        cat_parts = []
        for i in range(len(cumsum_indices)-1):
            part = shapley_vals[:,cumsum_indices[i]:cumsum_indices[i+1]].sum(1,keepdim=True)
            cat_parts.append(part)
        p_output = torch.cat(cat_parts,dim=1)
        return p_output
    else:
        return shapley_vals

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
        coeffs= 10**np.linspace(-7,-2,0)
        return l1,l2,True,coeffs

    if job in ['chameleon','chameleon_wl']:
        l2 = ['ch.res', 'jl.res', 'tl.res', 'mass.res', 'SVL', 'prop.main',
       'prop.patch']
        coeffs= 10**np.linspace(-7,-2,0)
        return [],l2,False,coeffs

    if job  == 'alan_data_5000_100':
        D=3
        a = np.zeros((D, D))
        zip1,zip2 = np.triu_indices(D, 1)

        # l2 = ['within_cluster'] + [f'feature_{i}_{j}' for (i,j) in zip(zip1,zip2)] + [f'indicator {d}' for d in range(D)]
        l2 = [r'$x^{[0]}$'] + [r'$x^{AB}$',r'$x^{AC}$',r'$x^{BC}$'] + ['in A','in B','in C']
        coeffs= 10**np.linspace(-7,-2,0)
        return [],l2,False,coeffs

    if job =='alan_data_5000_1000':
        D=3
        a = np.zeros((D, D))
        zip1,zip2 = np.triu_indices(D, 1)

        l2 = [r'$x^{[0]}$'] + [r'$x^{AB}$',r'$x^{AC}$',r'$x^{BC}$'] + ['in A','in B','in C']
        coeffs= 10**np.linspace(-7,-2,0)
        return [],l2,False,coeffs

    if job in ['alan_data_5000_1000_10_10','toy_data_5000_10_2']:
        l2 = [f'important_{i}' for i in range(1,3)] + [f'Unimportant_{i}' for i in range(3,11)]
        coeffs= 10**np.linspace(-7,-2,0)
        return [],l2,False,coeffs
    d=[5,3]
    if job in [f'hard_data_10000_1000_{d[0]}_{d[1]}']:
        l2 = [f'important_{i}' for i in range(1,d[1]+1)] + [f'Unimportant_{i}' for i in range(d[1]+1,d[0]+1)]
        coeffs= 10**np.linspace(-7,-2,0)
        return [],l2,False,coeffs


def cumsum_thingy(cumsum_indices,shapley_vals):
    cat_parts = []
    for i in range(len(cumsum_indices)-1):
        part = shapley_vals[cumsum_indices[i]:cumsum_indices[i+1],:].sum(0,keepdim=True)
        cat_parts.append(part)
    p_output = torch.cat(cat_parts,dim=0)
    return p_output

def get_shapley_vals(job,model_string,fold,train_params,num_matches,post_method,interventional):
    with open( f'{job}_results/{model_string}/run_{fold}.pickle' , 'rb') as handle:
        loaded_model = pickle.load(handle)
    best_model = dill.loads(loaded_model)
    ls = best_model['ls']
    alpha = best_model['alpha'].float()
    ind_points = best_model['inducing_points'].float()
    model = best_model['model'].to('cuda:0')
    c = train_GP(train_params=train_params)
    c.load_and_split_data()
    if model_string=='SGD_base':
        inner_kernel=RBF_multiple_ls(d=ind_points.shape[1])
        inner_kernel._set_lengthscale(ls)
        inner_kernel=inner_kernel.to('cuda:0')
        alpha=alpha.to('cuda:0')
        ps = rkhs_shap(model=model, alpha=alpha, k=inner_kernel, X=ind_points, max_S=2500,
                       rff_mode=False, eps=1e-3, cg_max_its=10, lamb=1e-3, max_inv_row=0, cg_bs=25, post_method=post_method,
                       interventional=interventional, device='cuda:0')
        x = torch.from_numpy(c.X_val).float()
        if x.shape[0]<100:
            x = torch.from_numpy(c.X_tr).float()
        Y_target, weights, Z = ps.fit(x)
        cooking_dict = {'Y': Y_target.cpu(), 'weights': weights.cpu(), 'Z': Z.cpu(),
                        'n': x.shape[0]}
        sum_count, features_names, do_sum, coeffs = return_feature_names(job)
        # if do_sum:
        #     chunk_l,chunk_r = torch.chunk(x,dim=1,chunks=2)
        #     data_l = cumsum_thingy_2(sum_count, chunk_l)
        #     data_r = cumsum_thingy_2(sum_count, chunk_r)
        #     data = torch.cat([data_l,data_r],dim=1)
        # else:
        #     data=x
        # cols =[f'{g} (l)' for g in features_names]+[f'{g} (r)' for g in features_names]
        # df = pd.DataFrame(data.numpy(), columns=cols)
        chunk_l, chunk_r = torch.chunk(x, dim=1, chunks=2)
        if do_sum:
            chunk_l = cumsum_thingy_2(sum_count, chunk_l)
            chunk_r = cumsum_thingy_2(sum_count, chunk_r)
        data = chunk_r+chunk_l #chunk_l+chunk_r
        # data = chunk_r-chunk_l #chunk_l+chunk_r
        print(features_names)
        df = pd.DataFrame(data.numpy(), columns=features_names )
        df['fold'] = f
        return cooking_dict,df
    else:
        pgp = model_string=='SGD_krr_pgp'
        x_ind_l,x_ind_r  = torch.chunk(ind_points,dim=1,chunks=2)
        inner_kernel=RBF_multiple_ls(d=x_ind_l.shape[1])
        inner_kernel._set_lengthscale(ls)
        inner_kernel=inner_kernel.to('cuda:0')
        alpha=alpha.to('cuda:0')
        ps = pref_shap(model=model, alpha=alpha, k=inner_kernel, X_l=x_ind_l, X_r=x_ind_r, X=c.S, max_S=2500,
                       rff_mode=False, eps=1e-3, cg_max_its=10, lamb=1e-3, max_inv_row=0, cg_bs=25, post_method=post_method,
                       interventional=interventional, device='cuda:0')
        x,x_prime = torch.from_numpy(c.left_val).float(),torch.from_numpy(c.right_val).float()
        y = torch.from_numpy((c.y_val > 0) * 1.0).unsqueeze(-1)
        if x.shape[0]<100:
            x, x_prime = torch.from_numpy(c.left_tr).float(), torch.from_numpy(c.right_tr).float()
            y = torch.from_numpy((c.y_tr > 0) * 1.0).unsqueeze(-1)

        shap_l,shap_r = x[0:num_matches, :], x_prime[0:num_matches, :]
        y = y[0:num_matches]
        Y_target, weights,Z= ps.fit(shap_l,shap_r,pgp=pgp)

        cooking_dict = {'Y':Y_target.cpu(), 'weights':weights.cpu(),'Z':Z.cpu(),
                        'n':shap_l.shape[0]}

        winners = y * shap_r + (1 - y) * shap_l
        loosers = (1 - y) * shap_r + y * shap_l
        sum_count, features_names, do_sum, coeffs = return_feature_names(job)
        diff_abs = winners - loosers
        data = cumsum_thingy_2(sum_count, diff_abs)
        df = pd.DataFrame(data.numpy(), columns=features_names)
        df['fold'] = f
        return cooking_dict,df


def get_shapley_vals_2(cooking_dict,job,post_method,m='SGD_krr'):
    sum_count,features_names,do_sum,coeffs=return_feature_names(job)
    if m=='SGD_base':
        pass
        # features_names =[f'{g} (l)' for g in features_names]+[f'{g} (r)' for g in features_names]
        # features_names=features_names
        # features_names = features_names + [f'{g}' for g in features_names]
    print(features_names)
    outputs = construct_values(cooking_dict['Y'],cooking_dict['Z'],
                              cooking_dict['weights'],coeffs,post_method
                              )
    big_plot=[]
    for key,output in outputs.items():

        if m == 'SGD_base':
            o_u, o_d = torch.chunk(output, dim=0, chunks=2)
            if do_sum:
                p_output_u = cumsum_thingy(sum_count, o_u)
                p_output_d = cumsum_thingy(sum_count, o_d)
                # p_output = p_output_d+p_output_u
                p_output = p_output_d-p_output_u
                # p_output = torch.cat([p_output_u, p_output_d], dim=0)
            else:
                # p_output = o_d-o_u
                p_output = o_d+o_u
        else:
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
    # for job in ['chameleon_wl','pokemon_wl']:
    # for job in ['chameleon_wl']:
    # for job in ['alan_data_5000_1000_10_10','toy_data_5000_10_2']:
    # for job in ['toy_data_5000_10_2']:
    # for job in ['chameleon_wl','pokemon_wl','alan_data_5000_100']:
    for job in ['alan_data_5000_100']:
    # d = [5, 3]
    # for job in [f'hard_data_10000_1000_{d[0]}_{d[1]}']:
    # for job in ['pokemon_wl']:
    #     for f in [0,1,2,3,4]:
    #     for m in ['SGD_krr','SGD_krr_pgp']:
        for m in ['SGD_base']:
            for f in [0]:
                interventional=False
                model=m
                fold=f
                train_params={
                    'dataset':job,
                    'fold':fold,
                    'epochs':100,
                    'patience':5,
                    'model_string':model, #krr_vanilla
                    'bs':1000,
                    'double_up':True,
                    'm_factor':1.0,
                    'seed': 42,
                    'folds': 10,
                }
                res_name = f'{interventional}_{job}_{model}'
                if not os.path.exists(f'{res_name}'):
                    os.makedirs(f'{res_name}')
                abs_data_container = []
                for f in [0]:
                    # if not os.path.exists(f'{interventional}_{job}/cooking_dict_{f}.pt'):
                    cooking_dict,abs_data = get_shapley_vals(job=job,model_string=model,fold=fold,train_params=train_params,num_matches=-1,post_method='OLS',interventional=interventional)
                    abs_data_container.append(abs_data)
                    torch.save(cooking_dict,f'{res_name}/cooking_dict_{f}.pt')
                big_df = pd.concat(abs_data_container,axis=0).reset_index(drop=True)
                big_df.to_csv(f'{res_name}/data_folds.csv')
                for post_method in ['lasso']:
                    big_plt = []
                    for f in [0]:
                        cooking_dict = torch.load(f'{res_name}/cooking_dict_{f}.pt')
                        data,features_names= get_shapley_vals_2(cooking_dict,job,post_method,m)
                        data['fold']=f
                        big_plt.append(data)
                    plot= pd.concat(big_plt,axis=0).reset_index(drop=True)
                    plot['d'] = plot['d'].apply(lambda x: features_names[int(x-1)])
                    plot.to_csv(f'{res_name}/{res_name}_{post_method}.csv')


