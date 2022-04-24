import numpy as np
import torch
from utils.utils import *
from pref_shap.pref_shap import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import dill

sns.set()



def return_feature_names(job,case=2):
    if job in ['website_data_user','website_user_data_wl']:
        if case==2:
            l1=[2, 2, 32, 29, 5, 23]
            l1.insert(0,0)
            l1=np.cumsum(l1).tolist()

            l2= ['is_seasonless',
                       'is_carried_over',
                       'product_group_name',
                       'graphical_appearance_name',
                        'assortment_mix_name',
                       'garment_group_name',
                 ]
        else:
            l1 = [1,1,1,1,1]
            l1.insert(0, 0)
            l1 = np.cumsum(l1).tolist()
            l2 = ['year_of_birth','gender_code_1','gender_code_2','gender_code_0','gender_code_3'
                  ]
        coeffs= 10**np.linspace(-9,-3,10)

        return l1, l2, True, coeffs

    if job in ['tennis_data_processed','tennis_data_processed_wl']:
        if case==2:
            l1 = [1, 1, 1, 1, 1,1,1,1, 1,1,1]
            l1.insert(0, 0)
            l1 = np.cumsum(l1).tolist()
            l2 = ['birth_year', 'weight_kg', 'height_cm', 'pro_age', 'handedness_Ambidextrous',
 'handedness_Left-Handed',
 'handedness_Right-Handed',
 'handedness_unknown',
 'backhand_One-Handed Backhand',
 'backhand_Two-Handed Backhand',
 'backhand_unknown']
        else:
            l1 = [1,1, 1,1,1,1]
            l1.insert(0, 0)
            l1 = np.cumsum(l1).tolist()
            l2 = ['tourney_conditions_Indoor', 'tourney_conditions_Outdoor',
       'tourney_surface_Carpet', 'tourney_surface_Clay',
       'tourney_surface_Grass', 'tourney_surface_Hard']
        # coeffs = [1e-4, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3]
        coeffs= 10**np.linspace(-9,-3,10)

        return l1, l2, True, coeffs



def cumsum_thingy(cumsum_indices,shapley_vals):
    cat_parts = []
    for i in range(len(cumsum_indices)-1):
        part = shapley_vals[cumsum_indices[i]:cumsum_indices[i+1],:].sum(0,keepdim=True)
        cat_parts.append(part)
    p_output = torch.cat(cat_parts,dim=0)
    return p_output


# results = {'test_auc': best_test, 'val_auc': best_val,
#            'ls_i': best_model.kernel.lengthscale_items.detach().cpu(),
#            'ls_u': best_model.kernel.lengthscale_users.detach().cpu(),
#            'lamb': best_model.penalty.detach().cpu(),
#            'alpha': alpha.cpu(),
#            'inducing_points_i': ind_points_all[:, self.ulen:],
#            'inducing_points_u': ind_points_all[:, :self.ulen],
#            }
def get_shapley_vals(job,model,fold,train_params,num_matches,post_method,interventional,case):
    with open( f'{job}_results/{model}/run_{fold}.pickle' , 'rb') as handle:
        loaded_model = pickle.load(handle)
    best_model = dill.loads(loaded_model)
    ls_i = best_model['ls_i']
    ls_u = best_model['ls_u']
    alpha = best_model['alpha'].float()
    ind_points = best_model['inducing_points_i'].float()
    x_ind_l,x_ind_r  = torch.chunk(ind_points,dim=1,chunks=2)
    x_u = best_model['inducing_points_u']
    job = train_params['dataset']
    model = best_model['model'].to('cuda:0')

    c= train_GP(train_params=train_params)
    c.load_and_split_data()

    inner_kernel=RBF_multiple_ls(d=x_ind_l.shape[1])
    inner_kernel._set_lengthscale(ls_i)
    inner_kernel=inner_kernel.to('cuda:0')

    inner_kernel_u=RBF_multiple_ls(d=x_u.shape[1])
    inner_kernel_u._set_lengthscale(ls_u)
    inner_kernel_u=inner_kernel_u.to('cuda:0')
    alpha=alpha.to('cuda:0')

    x,x_prime = torch.from_numpy(c.left_val).float(),torch.from_numpy(c.right_val).float()
    y_pred  =model.predict(c.dataloader.dataset.val_X.to('cuda:0'))
    u = torch.from_numpy(c.val_u).float()
    if x.shape[0]<100:
        x, x_prime = torch.from_numpy(c.left_tr).float(), torch.from_numpy(c.right_tr).float()
        y_pred  =model.predict(c.dataloader.dataset.train_X.to('cuda:0'))
        u = torch.from_numpy(c.tr_u).float()
    u_shap = u[0:num_matches,:]
    shap_l,shap_r = x[0:num_matches, :], x_prime[0:num_matches, :]

    ps = pref_shap(model=model,y_pred=y_pred,alpha=alpha,k=inner_kernel,X_l=x_ind_l,X_r=x_ind_r,X=c.S,u=x_u,k_U=inner_kernel_u,X_U=c.S_u,max_S=2500,rff_mode=False,
                   eps=1e-3,cg_max_its=10,lamb=1e-3,max_inv_row=2500,cg_bs=5,post_method=post_method,device='cuda:0',interventional=interventional)


    Y_target, weights,Z= ps.fit(shap_l,shap_r,u_shap,case)

    cooking_dict = {'Y':Y_target.cpu(), 'weights':weights.cpu(),'Z':Z.cpu(),
                    'n':shap_l.shape[0]}
    return cooking_dict
def get_shapley_vals_2(cooking_dict,job,post_method,case):
    sum_count,features_names,do_sum,coeffs=return_feature_names(job,case)
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
    interventional=False
    for j in ['tennis_data_processed_wl','website_user_data_wl']:
        if not os.path.exists(f'{interventional}_{j}'):
            os.makedirs(f'{interventional}_{j}')

        job=j
        model='SGD_ukrr'
        fold=0
        case=2
        train_params={
            'dataset':job,
            'fold':fold,
            'epochs':100,
            'patience':5,
            'model_string':'SGD_ukrr', #krr_vanilla
            'bs':1000,
            'double_up': False,
            'm_factor': 1.

        }
        folds=[0,1,2]
        for case in [2, 1]:
            for f in folds:
                if not os.path.exists(f'{interventional}_{job}/cooking_dict_{f}_{case}.pt'):
                    cooking_dict = get_shapley_vals(job=job,model=model,fold=fold,train_params=train_params,num_matches=-1,post_method='OLS',interventional=interventional,case=case)
                    torch.save(cooking_dict,f'{interventional}_{job}/cooking_dict_{f}_{case}.pt')

    for job in ['tennis_data_processed_wl','website_user_data_wl']:
        for post_method in ['lasso','ridge','elastic']:
            for case in [1,2]:
                big_plt = []
                for f in [0,1,2]:
                    cooking_dict =torch.load(f'{interventional}_{job}/cooking_dict_{f}_{case}.pt')
                    data, features_names = get_shapley_vals_2(cooking_dict, job, post_method,case)
                    data['fold'] = f
                    big_plt.append(data)
                plot = pd.concat(big_plt, axis=0).reset_index(drop=True)
                plot['d'] = plot['d'].apply(lambda x: features_names[int(x - 1)])
                plot.to_csv(f'{interventional}_{job}/{interventional}_{job}_{post_method}_{case}.csv')