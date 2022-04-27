import numpy as np
import torch
from utils.utils import *
from pref_shap.pref_shap import *
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import dill

sns.set()

def cumsum_thingy_2(cumsum_indices,shapley_vals):
    cat_parts = []
    for i in range(len(cumsum_indices)-1):
        part = shapley_vals[:,cumsum_indices[i]:cumsum_indices[i+1]].sum(1,keepdim=True)
        cat_parts.append(part)
    p_output = torch.cat(cat_parts,dim=1)
    return p_output


def process_data(u,shap_l,shap_r,y,job,case,f):
    if case==2:
        sum_count, features_names, do_sum, coeffs = return_feature_names(job, case=case)
        # diff_abs = np.abs(x - x_prime)

        winners = y * shap_r + (1 - y) * shap_l
        loosers = (1 - y) * shap_r + y * shap_l
        diff_abs = winners - loosers

        data = cumsum_thingy_2(sum_count, diff_abs)
        df = pd.DataFrame(data, columns=features_names)
        df['fold'] = f
    elif case==1:
        sum_count, features_names, do_sum, coeffs = return_feature_names(job, case=case)
        data = cumsum_thingy_2(sum_count, u)
        df = pd.DataFrame(data, columns=features_names)
        df['fold'] = f
    return df

def return_feature_names(job,case=2):
    if job in ['website_data_user','website_user_data_wl']:
        if case==2:
            # l1=[2, 2, 32, 29, 5, 23]
            l1=[1,1, 1,1]+[1]*89

            #redo top 10 featuers only
            l1.insert(0,0)
            l1=np.cumsum(l1).tolist()
            l2 = [
                'is_seasonless_N',
                'is_seasonless_Y',
                'is_carried_over_N',
                'is_carried_over_Y',
                'product_group_name_Accessories',
                'product_group_name_Bags',
                'product_group_name_Body care',
                'product_group_name_Cleaning & Gardening',
                'product_group_name_Cosmetic',
                'product_group_name_Cosmetic Tools & Accessories',
                'product_group_name_Eyes Cosmetics',
                'product_group_name_Face Cosmetics',
                'product_group_name_Fragrance',
                'product_group_name_Fun',
                'product_group_name_Furniture',
                'product_group_name_Garment Full body',
                'product_group_name_Garment Lower body',
                'product_group_name_Garment Upper body',
                'product_group_name_Garment and Shoe care',
                'product_group_name_Interior decorations',
                'product_group_name_Interior textile',
                'product_group_name_Items',
                'product_group_name_Kitchen products',
                'product_group_name_Lighting',
                'product_group_name_Lips Cosmetics',
                'product_group_name_Nails Cosmetics',
                'product_group_name_Nightwear',
                'product_group_name_Shoes',
                'product_group_name_Skin care',
                'product_group_name_Socks & Tights',
                'product_group_name_Stationery',
                'product_group_name_Storage',
                'product_group_name_Swimwear',
                'product_group_name_Underwear',
                'product_group_name_Underwear/nightwear',
                'product_group_name_Unknown',
                'graphical_appearance_name_All over pattern',
                'graphical_appearance_name_Application/3D',
                'graphical_appearance_name_Argyle',
                'graphical_appearance_name_Chambray',
                'graphical_appearance_name_Check',
                'graphical_appearance_name_Colour blocking',
                'graphical_appearance_name_Contrast',
                'graphical_appearance_name_Denim',
                'graphical_appearance_name_Dot',
                'graphical_appearance_name_Embroidery',
                'graphical_appearance_name_Front print',
                'graphical_appearance_name_Glittering/Metallic',
                'graphical_appearance_name_Jacquard',
                'graphical_appearance_name_Lace',
                'graphical_appearance_name_Melange',
                'graphical_appearance_name_Mesh',
                'graphical_appearance_name_Metallic',
                'graphical_appearance_name_Mixed solid/pattern',
                'graphical_appearance_name_Neps',
                'graphical_appearance_name_Other pattern',
                'graphical_appearance_name_Other structure',
                'graphical_appearance_name_Placement print',
                'graphical_appearance_name_Sequin',
                'graphical_appearance_name_Slub',
                'graphical_appearance_name_Solid',
                'graphical_appearance_name_Stripe',
                'graphical_appearance_name_Tie Dye',
                'graphical_appearance_name_Transparent',
                'graphical_appearance_name_Treatment',
                'assortment_mix_name_Ca$h Cow',
                'assortment_mix_name_Questionmark',
                'assortment_mix_name_Star',
                'assortment_mix_name_Unknown',
                'assortment_mix_name_Unspecified',
                'garment_group_name_Accessories',
                'garment_group_name_Blouses',
                'garment_group_name_Cosmetic',
                'garment_group_name_Dressed',
                'garment_group_name_Dresses Ladies',
                'garment_group_name_Dresses/Skirts girls',
                'garment_group_name_H&M Home',
                'garment_group_name_Jersey Basic',
                'garment_group_name_Jersey Fancy',
                'garment_group_name_Knitwear',
                'garment_group_name_Outdoor',
                'garment_group_name_Shirts',
                'garment_group_name_Shoes',
                'garment_group_name_Shorts',
                'garment_group_name_Skirts',
                'garment_group_name_Socks and Tights',
                'garment_group_name_Special Offers ',
                'garment_group_name_Swimwear',
                'garment_group_name_Trousers',
                'garment_group_name_Trousers Denim',
                'garment_group_name_Under-, Nightwear',
                'garment_group_name_Unknown',
                'garment_group_name_Woven/Jersey/Knitted mix Baby',

            ]
            # l2= ['is_seasonless',
            #            'is_carried_over',
            #
            #            'product_group_name',
            #            'graphical_appearance_name',
            #             'assortment_mix_name',
            #            'garment_group_name',
            #      ]
        else:
            l1 = [1,1,1,1,1]
            l1.insert(0, 0)
            l1 = np.cumsum(l1).tolist()
            l2 = ['year_of_birth','gender_code_1','gender_code_2','gender_code_0','gender_code_3'
                  ]
        coeffs= 10**np.linspace(-9,-3,0)

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
        coeffs= 10**np.linspace(-9,-3,0)

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



    ps = pref_shap(model=model, alpha=alpha, k=inner_kernel, X_l=x_ind_l, X_r=x_ind_r, X=c.S, k_U=inner_kernel_u, u=x_u,
                   X_U=c.S_u, max_S=2500, rff_mode=False, eps=1e-3, cg_max_its=10, lamb=1e-3, max_inv_row=2500, cg_bs=5,
                   post_method=post_method, interventional=interventional, device='cuda:0')

    x, x_prime = torch.from_numpy(c.left_val).float(), torch.from_numpy(c.right_val).float()
    u = torch.from_numpy(c.val_u).float()
    y = torch.from_numpy((c.y_val > 0) * 1.0)[0:num_matches].unsqueeze(-1)

    if x.shape[0] < 100:
        x, x_prime = torch.from_numpy(c.left_tr).float(), torch.from_numpy(c.right_tr).float()
        u = torch.from_numpy(c.tr_u).float()
        y = torch.from_numpy((c.y_tr > 0) * 1.0)[0:num_matches].unsqueeze(-1)

    if x.shape[0] > 2500:
        x, x_prime = x[:2500,:],x_prime[:2500,:]
        u = u[:2500,:]
        y = y[:2500,:]

    u_shap = u[0:num_matches, :]
    shap_l, shap_r = x[0:num_matches, :], x_prime[0:num_matches, :]
    y = y[0:num_matches]

    df = process_data(u_shap,shap_l, shap_r, y, job, case, f)


    Y_target, weights,Z= ps.fit(shap_l,shap_r,u_shap,case)
    cooking_dict = {'Y':Y_target.cpu(), 'weights':weights.cpu(),'Z':Z.cpu(),
                    'n':shap_l.shape[0]}

    return cooking_dict,df
def get_shapley_vals_2(cooking_dict,job,post_method,case,big_weight=1e5):
    sum_count,features_names,do_sum,coeffs=return_feature_names(job,case)
    outputs = construct_values(cooking_dict['Y'],cooking_dict['Z'],
                              cooking_dict['weights'],coeffs,post_method,big_weight=big_weight
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
    # for j in ['website_user_data_wl']:
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
            'm_factor': 1.,
        'seed':2,
        'folds':5,

        }
        folds=[0]
        for case in [2, 1]:
            abs_data_container = []
            for f in folds:
                if not os.path.exists(f'{interventional}_{job}/cooking_dict_{f}_{case}.pt'):
                    cooking_dict,df = get_shapley_vals(job=job,model=model,fold=fold,train_params=train_params,num_matches=-1,post_method='OLS',interventional=interventional,case=case)
                    torch.save(cooking_dict,f'{interventional}_{job}/cooking_dict_{f}_{case}.pt')
                    abs_data_container.append(df)
                if case==2:
                    if not os.path.exists(f'{interventional}_{job}/data_folds.csv'):
                        big_df = pd.concat(abs_data_container, axis=0).reset_index(drop=True)
                        big_df.to_csv(f'{interventional}_{job}/data_folds.csv')
                if case==1:
                    if not os.path.exists(f'{interventional}_{job}/data_folds_user.csv'):
                        big_df = pd.concat(abs_data_container, axis=0).reset_index(drop=True)
                        big_df.to_csv(f'{interventional}_{job}/data_folds_user.csv')

    for job,big_weight in zip(['tennis_data_processed_wl','website_user_data_wl'],[1e5,1e5]):
        for post_method in ['lasso']:
            for case in [2,1]:
                big_plt = []
                for f in [0]:
                    print(job,case)
                    cooking_dict =torch.load(f'{interventional}_{job}/cooking_dict_{f}_{case}.pt')
                    data, features_names = get_shapley_vals_2(cooking_dict, job, post_method,case,big_weight=big_weight)
                    data['fold'] = f
                    big_plt.append(data)
                plot = pd.concat(big_plt, axis=0).reset_index(drop=True)
                plot['d'] = plot['d'].apply(lambda x: features_names[int(x - 1)])
                plot.to_csv(f'{interventional}_{job}/{interventional}_{job}_{post_method}_{case}.csv')