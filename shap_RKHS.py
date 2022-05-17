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

    if job=='LOL':
        l2=['id_x', 'matchid', 'player', 'championid', 'ss1', 'ss2', 'role', 'position', 'id_y', 'gameid', 'platformid', 'queueid', 'seasonid', 'duration', 'creation', 'version', 'id', 'item1', 'item2', 'item3', 'item4', 'item5', 'item6', 'trinket', 'Kills per min.', 'Deaths per min.', 'Assists per min.', 'largestkillingspree', 'largestmultikill', 'Killing sprees per min.', 'Longest time living as % of game', 'Double kills per min.', 'Triple kills per min.', 'Quadra kills per min.', 'Penta kills per min.', 'Legendary kills per min.', 'Total damage dealt per min.', 'Magic damage dealt per min.', 'Physical damage dealt per min.', 'True damage dealt per min.', 'largestcrit', 'Total damage to champions per min.', 'Magic damage to champions per min.', 'Physical damage to champions per min.', 'True damage to champions per min.', 'Total healing per min.', 'Total units healed per min.', 'dmgselfmit', 'Damage to objects per min.', 'Damage to turrets', 'visionscore', 'Time spent with crown control per min.', 'Total damage taken per min.', 'Magic damage taken per min.', 'Physical damage taken per min.', 'True damage taken per min.', 'Gold earned per min.', 'Gold spent per min.', '# of turret kills', '# of inhibitor kills', 'Total minions killed per min.', 'Neutral minions killed per min.', 'Own jungle kills per min.', 'Enemy jungle kills per min.', 'Total crown control time dealt per min.', 'champlvl', 'Pink wards bought per min.', 'Wards bought per min.', 'Wards placed per min.', 'wardskilled', 'firstblood']
    if job=='census':
        l2=['Age', 'Workclass', 'Education-Num', 'Marital Status', 'Occupation',
       'Relationship', 'Race', 'Sex', 'Capital Gain', 'Capital Loss',
       'Hours per week', 'Country']
    coeffs = 10 ** np.linspace(-7, -2, 0)
    return [], l2, False, coeffs


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
    c = train_krr_simple(train_params=train_params)
    c.load_and_split_data()
    inner_kernel=RBF_multiple_ls(d=ind_points.shape[1])
    inner_kernel._set_lengthscale(ls)
    inner_kernel=inner_kernel.to('cuda:0')
    alpha=alpha.to('cuda:0')
    ps = rkhs_shap(model=model, alpha=alpha, k=inner_kernel, X=ind_points, max_S=2500,
                   rff_mode=False, eps=1e-3, cg_max_its=10, lamb=1e-3, max_inv_row=0, cg_bs=10, post_method=post_method,
                   interventional=interventional, device='cuda:0')
    x = torch.from_numpy(c.X_val).float()
    if x.shape[0]<100:
        x = torch.from_numpy(c.X_tr).float()
    if x.shape[0]>10000:
        x = x[:10000,:]
    Y_target, weights, Z = ps.fit(x)
    cooking_dict = {'Y': Y_target.cpu(), 'weights': weights.cpu(), 'Z': Z.cpu(),
                    'n': x.shape[0]}
    sum_count, features_names, do_sum, coeffs = return_feature_names(job)
    data=x
    df = pd.DataFrame(data.numpy(), columns=features_names)
    df['fold'] = f
    return cooking_dict,df


def get_shapley_vals_2(cooking_dict,job,post_method,m='SGD_krr'):
    sum_count,features_names,do_sum,coeffs=return_feature_names(job)
    outputs = construct_values(cooking_dict['Y'],cooking_dict['Z'],
                              cooking_dict['weights'],coeffs,post_method
                              )
    big_plot=[]
    for key,output in outputs.items():
        tmp = output.cpu().numpy().flatten()
        tst= np.arange(1,len(features_names)+1).repeat(cooking_dict['n'])
        plot =  pd.DataFrame(np.stack([tst,tmp,np.ones_like(tst)*key],axis=1),columns=['d','shapley_vals','lambda'])
        big_plot.append(plot)
    plot = pd.concat(big_plot,axis=0).reset_index()
    return plot,features_names



if __name__ == '__main__':
    for job in ['LOL']:
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
                    'double_up':False,
                    'm_factor':1.0,
                    'seed': 42,
                    'folds': 5,
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


