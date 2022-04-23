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

    df = pd.read_csv(f'{dirname}/{fn}.csv', index_col=0)
    data = pd.read_csv(f'{dirname}/{data_name}.csv', index_col=0)
    piv_df, cols = pivot(df)

    piv_df = piv_df[piv_df['fold'] == 0]
    data = data[data['fold'] == 0]
    lambs = piv_df['lambda'].unique().tolist()

    for lamb in lambs:
        piv_df_lamb = piv_df[piv_df['lambda'] == lamb]
        shap_values = shap.Explanation(values=piv_df_lamb[cols].values, feature_names=cols, data=data[cols].values)
        fig = plt.gcf()
        shap.summary_plot(shap_values)
        fig.savefig(f'{savedir}/bee_plot_lamb={lamb}.png', bbox_inches='tight')
        plt.clf()
        fig = plt.gcf()
        shap.plots.bar(shap_values)
        fig.savefig(f'{savedir}/bar_plot_lamb={lamb}.png', bbox_inches='tight')
        plt.clf()
    # vars_data = []
    tmp = df.groupby(['fold', 'd', 'lambda'])['shapley_vals'].var().reset_index()
    tmp['log_var'] = tmp['shapley_vals'].apply(lambda x: np.log(x))
    sns.lineplot(data=tmp, x="lambda", y="log_var", hue="d")
    plt.savefig(f'{savedir}/{fn}.png', bbox_inches='tight')
    plt.clf()


if __name__ == '__main__':
    # dirname = 'False_pokemon_wl'
    for dirname in ['False_chameleon_wl','False_pokemon_wl']:
        # fns = [f'{dirname}_elastic',f'{dirname}_lasso',f'{dirname}_ridge']
        fns = [f'{dirname}_lasso']
        data_name = 'data_folds'
        for fn in fns:
            produce_plots(dirname,data_name,fn)

    # for ds in ['tennis_data_processed','website_data_user']:
    #     dirname = f'False_{ds}'
    #     for i,user in zip([2,1],['','_user']):
    #         fns = [f'False_{ds}_elastic_{i}',f'False_{ds}_lasso_{i}',f'False_{ds}_ridge_{i}']
    #         data_name = 'data_folds' + user
    #         for fn in fns:
    #             produce_plots(dirname,data_name,fn)