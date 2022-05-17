import os.path

import numpy as np
import pandas as pd
import seaborn as sns
# sns.set()
import shap
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 50})
# train XGBoost model

plt.rcParams['text.usetex'] = False  # not really needed


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

def filter_out_names(cols,dataset):
    if  'tennis_data_processed_wl_lasso_2' in dataset:
        feature_name = []

        for c in cols:
            c = c.replace('handedness_','')
            feature_name.append(c)
        return feature_name

    if 'tennis_data_processed_wl_lasso_1' in dataset:
        feature_name = []

        for c in cols:
            c = c.replace('tourney_surface_','')
            c = c.replace('tourney_conditions_','')
            feature_name.append(c)
        return feature_name

    if dataset=='False_website_user_data_wl_lasso_2.csv':
        feature_name = []

        for c in cols:
            c = c.replace('garment_group_name_','')
            c = c.replace('product_group_name_','')
            c = c.replace('assortment_mix_name_','')
            feature_name.append(c)

        return feature_name
    return cols
def filter_type(df,data,d):
    columns = df['d'].unique().tolist()
    mask = (df['d']==columns[d[0]]) |  (df['d']==columns[d[1]])
    df_subset = df[mask]

    data_subset = data.iloc[:,[d[0],d[1],-1]]


    return df_subset,data_subset
def produce_plots_pokemon_local(dirname,data_name,fn,d,max_disp=7):
    savedir = dirname + f'{data_name}_plots'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    df = pd.read_csv(f'{dirname}/{dirname}_{fn}.csv', index_col=0)
    data = pd.read_csv(f'{dirname}/{data_name}.csv', index_col=0)


    df,data,=filter_type(df,data,d)

    piv_df, cols = pivot(df)

    piv_df = piv_df[piv_df['fold'] == 0]
    data = data[data['fold'] == 0]
    ds_nam = f'{dirname}_{fn}.csv'
    for lamb in piv_df['lambda'].unique().tolist():
        piv_df_lamb = piv_df[piv_df['lambda'] == lamb]
        vals  = piv_df_lamb[cols].values
        max_norm = np.abs(vals).max()
        vals = vals/max_norm

        # vals = vals + np.random.randn(*vals.shape)*0.025

        feature_cols = filter_out_names(cols,ds_nam)
        shap_values = shap.Explanation(values=vals, feature_names=feature_cols, data=data[cols].values)
        fig = plt.gcf()
        shap.plots.beeswarm(shap_values,max_display=max_disp,show=False, color_bar=False)
        fig.savefig(f'{savedir}/bee_plot_lamb={lamb}_{fn}.png', bbox_inches='tight')
        plt.clf()
        fig = plt.gcf()
        shap.plots.bar(shap_values,max_display=max_disp+1,show=False)
        fig.savefig(f'{savedir}/bar_plot_lamb={lamb}_{fn}.png', bbox_inches='tight')
        plt.clf()
    # vars_data = []
    tmp = df.groupby(['fold', 'd', 'lambda'])['shapley_vals'].var().reset_index()
    tmp['log_var'] = tmp['shapley_vals'].apply(lambda x: np.log(x))
    sns.lineplot(data=tmp, x="lambda", y="log_var", hue="d")
    plt.savefig(f'{savedir}/{fn}.png', bbox_inches='tight')
    plt.clf()


def produce_plots(dirname,data_name,fn,max_disp=7):
    savedir = dirname + f'{data_name}_plots'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    df = pd.read_csv(f'{dirname}/{dirname}_{fn}.csv', index_col=0)
    data = pd.read_csv(f'{dirname}/{data_name}.csv', index_col=0)
    piv_df, cols = pivot(df)

    piv_df = piv_df[piv_df['fold'] == 0]
    data = data[data['fold'] == 0]
    ds_nam = f'{dirname}_{fn}.csv'
    for lamb in piv_df['lambda'].unique().tolist():
        piv_df_lamb = piv_df[piv_df['lambda'] == lamb]
        vals  = piv_df_lamb[cols].values
        max_norm = np.abs(vals).max()
        vals = vals/max_norm
        feature_cols = filter_out_names(cols,ds_nam)
        shap_values = shap.Explanation(values=vals, feature_names=feature_cols, data=data[cols].values)
        fig = plt.gcf()
        shap.plots.beeswarm(shap_values,max_display=max_disp,show=False, color_bar=False)
        plt.xlabel("RKHS-SHAP values \n Impact on winning probability")
        fig.savefig(f'{savedir}/bee_plot_lamb={lamb}_{fn}.png', bbox_inches='tight')
        plt.clf()
        fig = plt.gcf()
        shap.plots.bar(shap_values,max_display=max_disp+1,show=False,)
        plt.xlabel("mean(|RKHS-SHAP values|) \n Impact on winning probability")
        fig.savefig(f'{savedir}/bar_plot_lamb={lamb}_{fn}.png', bbox_inches='tight')
        plt.clf()
    # vars_data = []
    tmp = df.groupby(['fold', 'd', 'lambda'])['shapley_vals'].var().reset_index()
    tmp['log_var'] = tmp['shapley_vals'].apply(lambda x: np.log(x))
    sns.lineplot(data=tmp, x="lambda", y="log_var", hue="d")
    plt.savefig(f'{savedir}/{fn}.png', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    for dirname in ['False_LOL_SGD_base']:#,'False_LOL_SGD_base']:
        fns = [f'lasso']
        data_name = 'data_folds'
        for fn in fns:
            produce_plots(dirname,data_name,fn,50)
