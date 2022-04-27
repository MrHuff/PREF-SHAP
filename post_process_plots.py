import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import shap

# train XGBoost model



def pivot(df):
    cols = df['d'].unique().tolist()
    values=[]
    for c in cols:
        sub = df[df['d']==c]
        vals = sub['shapley_vals'].values
        vals=vals[:,np.newaxis]
        values.append(vals)
    lf = sub[['fold','lambda']].values
    values.append(lf)
    values = np.concatenate(values,axis=1)
    all_cols=cols+['fold','lambda']
    d = pd.DataFrame(values,columns=all_cols)
    return d,cols

def slice_df(df):
    return df

def produce_plots(dirname,data_name,fn):
    savedir = dirname + f'{data_name}_plots'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    df = pd.read_csv(f'{dirname}/{dirname}_{fn}.csv', index_col=0)
    data = pd.read_csv(f'{dirname}/{data_name}.csv', index_col=0)
    piv_df, cols = pivot(df)

    piv_df = piv_df[piv_df['fold'] == 0]
    data = data[data['fold'] == 0]
    lambs = piv_df['lambda'].unique().tolist()

    for lamb in [0.0]:
        piv_df_lamb = piv_df[piv_df['lambda'] == lamb]
        vals  = piv_df_lamb[cols].values
        max_norm = np.abs(vals).max()
        vals = vals/max_norm
        shap_values = shap.Explanation(values=vals, feature_names=cols, data=data[cols].values)
        fig = plt.gcf()
        shap.summary_plot(shap_values,max_display=15)
        fig.savefig(f'{savedir}/bee_plot_lamb={lamb}_{fn}.png', bbox_inches='tight')
        plt.clf()
        fig = plt.gcf()
        shap.plots.bar(shap_values,max_display=15)
        fig.savefig(f'{savedir}/bar_plot_lamb={lamb}_{fn}.png', bbox_inches='tight')
        plt.clf()
    # vars_data = []
    tmp = df.groupby(['fold', 'd', 'lambda'])['shapley_vals'].var().reset_index()
    tmp['log_var'] = tmp['shapley_vals'].apply(lambda x: np.log(x))
    sns.lineplot(data=tmp, x="lambda", y="log_var", hue="d")
    plt.savefig(f'{savedir}/{fn}.png', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    # d=[10,5]
    # for dirname in [f'local_hard_data_10000_1000_{d[0]}_{d[1]}']:
    for model in ['SGD_krr','SGD_krr_pgp']:
        for d in [[13,24],[17,10],[12,19],[16,13],[10,14],[18,9],[18,14]]:
            for dirname in [f'local_pokemon_wl_{d}_{model}']:
                fns = [f'lasso']
                data_name = 'data_folds'
                for fn in fns:
                    produce_plots(dirname,data_name,fn)


    # for dirname in [ 'False_alan_data_5000_1000_10_10_SGD_krr','False_toy_data_5000_10_2_SGD_krr']:
    #     # fns = [f'elastic',f'lasso',f'ridge']
    #     fns = [f'lasso']
    #     data_name = 'data_folds'
    #     for fn in fns:
    #         produce_plots(dirname,data_name,fn)

    # for dirname in ['False_chameleon_wl','False_pokemon_wl','False_chameleon_wl_SGD_krr_pgp','False_pokemon_wl_SGD_krr_pgp']:
    #     # fns = [f'elastic',f'lasso',f'ridge']
    #     fns = [f'lasso']
    #     data_name = 'data_folds'
    #     for fn in fns:
    #         produce_plots(dirname,data_name,fn)
    #
    # for ds in ['website_user_data_wl','tennis_data_processed_wl']:
    #     dirname = f'False_{ds}'
    #     for i,user in zip([2,1],['','_user']):
    #         fns = [f'lasso_{i}']
    #         data_name = 'data_folds' + user
    #         for fn in fns:
    #             produce_plots(dirname,data_name,fn)