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
        l1= [1,1,1,1,1,1,1,19]
        l1.insert(0,0)
        l1=np.cumsum(l1).tolist()
        # l1=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary',
       # 'Bug', 'Dark', 'Dragon', 'Electric', 'Fairy', 'Fighting', 'Fire',
       # 'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison',
       # 'Psychic', 'Rock', 'Steel', 'Water']
        l2=['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary','Type']
        coeffs=[1e-4,1e-3,2e-3,3e-3,4e-3,5e-3]

        return l1,l2,True,coeffs

    if job=='website_data':
        l1=[2, 2, 1, 287, 33, 29, 99, 5, 29, 68, 212, 23]
        l1.insert(0,0)
        l1=np.cumsum(l1).tolist()

        l2= ['is_seasonless',
                   'is_carried_over',
                   'product_group_name',
                   'graphical_appearance_name',
                    'assortment_mix_name',
                   'garment_group_name',
             ]
        return l1,l2,True

    if job == 'flatlizard':
        l2 = ['throat.PC1', 'throat.PC2', 'throat.PC3', 'frontleg.PC1',
       'frontleg.PC2', 'frontleg.PC3', 'badge.PC1', 'badge.PC2', 'badge.PC3',
       'badge.size', 'testosterone', 'SVL', 'head.length', 'head.width',
       'head.height', 'condition', 'repro_resident', 'repro_floater']
        return [],l2,False

    if job == 'chameleon':
        l2 = ['ch.res', 'jl.res', 'tl.res', 'mass.res', 'SVL', 'prop.main',
       'prop.patch']
        return [],l2,False

def cumsum_thingy(cumsum_indices,shapley_vals):
    cat_parts = []
    for i in range(len(cumsum_indices)-1):
        part = shapley_vals[cumsum_indices[i]:cumsum_indices[i+1],:].sum(0,keepdim=True)
        cat_parts.append(part)
    p_output = torch.cat(cat_parts,dim=0)
    return p_output

def get_shapley_vals(job,model,fold,train_params,num_matches,post_method,interventional):
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
    ps = pref_shap(alpha=alpha,k=inner_kernel,X_l=x_ind_l,X_r=x_ind_r,X=c.S,max_S=2500,rff_mode=False,
                   eps=1e-3,cg_max_its=10,lamb=1e-3,max_inv_row=0,cg_bs=25,post_method=post_method,device='cuda:0',interventional=interventional)
    sum_count,features_names,do_sum,coeffs=return_feature_names(job)
    x,x_prime = torch.from_numpy(c.left_val).float(),torch.from_numpy(c.right_val).float()
    # num_matches = 100
    shap_l,shap_r = x[0:num_matches, :], x_prime[0:num_matches, :]
    ps.fit(shap_l,shap_r)
    outputs = ps.construct_values(coeffs)

    big_plot=[]
    for key,output in outputs.items():
        if do_sum:
            p_output = cumsum_thingy(sum_count,output)
        else:
            p_output = output
        tmp = p_output.cpu().numpy().flatten()
        tst= np.arange(1,len(features_names)+1).repeat(shap_l.shape[0])

        # Save stuff
        # pd.DataFrame(shap_l.cpu().detach().numpy()).to_csv("tr_shap_l.csv")
        # pd.DataFrame(shap_r.cpu().detach().numpy()).to_csv("tr_shap_r.csv")
        # pd.DataFrame(output.cpu().detach().numpy()).to_csv("tr_ps_output.csv")
        # pd.DataFrame(c.y_tr).to_csv("tr_y.csv")
        plot =  pd.DataFrame(np.stack([tst,tmp,np.ones_like(tst)*key],axis=1),columns=['d','shapley_vals','lambda'])
        big_plot.append(plot)
    plot = pd.concat(big_plot,axis=0).reset_index()
    return plot,features_names
if __name__ == '__main__':
    # d_imp = 2
    # d=10
    # palette =['r']*d_imp+ ['g']*(d-d_imp)
    post_method='lasso'
    interventional=False
    job='pokemon'
    model='SGD_krr'
    fold=0
    train_params={
        'dataset':job,
        'fold':fold,
        'epochs':100,
        'patience':5,
        'model_string':'SGD_krr', #krr_vanilla
        'bs':1000
    }
    big_plt=[]
    for f in [0,1,2,3,4,5]:
        data,features_names = get_shapley_vals(job=job,model=model,fold=fold,train_params=train_params,num_matches=-1,post_method=post_method,interventional=interventional)
        big_plt.append(data)
    plot= pd.concat(big_plt,axis=0).reset_index()

    if not os.path.exists(f'{interventional}_{post_method}_{job}'):
        os.makedirs(f'{interventional}_{post_method}_{job}')
    plot['d'] = plot['d'].apply(lambda x: features_names[int(x-1)])
    sns.displot(plot,x="shapley_vals", hue="d", kind="kde",col='lambda',facet_kws=dict(sharey=False,sharex=False))
    plt.savefig(f'{interventional}_{post_method}_{job}/displot_1.png')
    plt.clf()

    axs=sns.catplot(data=plot,x="d",y='shapley_vals',col='lambda',sharey=False)
    for ax in axs.axes[0]:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    plt.tight_layout()
    plt.savefig(f'{interventional}_{post_method}_{job}/stripplot_2.png')
    plt.clf()

        # plot =  pd.DataFrame(np.abs(np.stack([tst,tmp],axis=1)),columns=['d','shapley_vals'])
        # plot['d'] = plot['d'].apply(lambda x: features_names[int(x-1)])
        # sns.displot(plot,x="shapley_vals", hue="d", kind="kde")
        # # plt.show()
        # plt.savefig(f'{interventional}_{post_method}_{job}/{key}_2.png')
        # plt.clf()

    # with open( f'{job}_results/{model}/run_{fold}.pickle' , 'rb') as handle:
    #     best_model = pickle.load(handle)
    # ls = best_model['ls']
    # alpha = best_model['alpha'].float()
    # ind_points = best_model['inducing_points'].float()
    #
    # x_ind_l,x_ind_r  = torch.chunk(ind_points,dim=1,chunks=2)
    # c=train_GP(train_params=train_params,m=1000)
    # c.load_and_split_data()
    #
    # inner_kernel=RBF_multiple_ls(d=x_ind_l.shape[1])
    # inner_kernel._set_lengthscale(ls)
    # inner_kernel=inner_kernel.to('cuda:0')
    # alpha=alpha.to('cuda:0')
    # ps = pref_shap(alpha=alpha,k=inner_kernel,X_l=x_ind_l,X_r=x_ind_r,X=c.S,max_S=2500,rff_mode=False,
    #                eps=1e-3,cg_max_its=10,lamb=1e-3,max_inv_row=0,cg_bs=25,post_method=post_method,device='cuda:0',interventional=interventional)
    # sum_count,features_names,do_sum,coeffs=return_feature_names(job)
    # x,x_prime = torch.from_numpy(c.left_tr).float(),torch.from_numpy(c.right_tr).float()
    # num_matches = 100
    # shap_l,shap_r = x[0:num_matches, :], x_prime[0:num_matches, :]
    # ps.fit(shap_l,shap_r)
    # outputs = ps.construct_values(coeffs)
    # if not os.path.exists(f'{interventional}_{post_method}_{job}'):
    #     os.makedirs(f'{interventional}_{post_method}_{job}')
    # big_plot=[]
    # for key,output in outputs.items():
    #     if do_sum:
    #         p_output = cumsum_thingy(sum_count,output)
    #     else:
    #         p_output = output
    #     tmp = p_output.cpu().numpy().flatten()
    #     tst= np.arange(1,len(features_names)+1).repeat(shap_l.shape[0])
    #
    #     # Save stuff
    #     # pd.DataFrame(shap_l.cpu().detach().numpy()).to_csv("tr_shap_l.csv")
    #     # pd.DataFrame(shap_r.cpu().detach().numpy()).to_csv("tr_shap_r.csv")
    #     # pd.DataFrame(output.cpu().detach().numpy()).to_csv("tr_ps_output.csv")
    #     # pd.DataFrame(c.y_tr).to_csv("tr_y.csv")
    #     plot =  pd.DataFrame(np.stack([tst,tmp,np.ones_like(tst)*key],axis=1),columns=['d','shapley_vals','lambda'])
    #     big_plot.append(plot)
    # plot = pd.concat(big_plot,axis=0).reset_index()