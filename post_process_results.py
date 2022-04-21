import pandas as pd
import pickle

if __name__ == '__main__':
    columns = ['model','data','fold','tr_auc','val_auc','test_auc']
    for ds in ['website_user_data_wl','tennis_data_processed_wl']:
    # for ds in ['chameleon_wl', 'pokemon_wl']:
        data = []
        for method in ['SGD_ukrr','SGD_ukrr_pgp']:
        # for method in ['SGD_krr', 'SGD_krr_pgp']:
            for f in [0, 1, 2,]:
                with open(f'{ds}_results/{method}/run_{f}.pickle', 'rb') as handle:
                    best_model = pickle.load(handle)
                    data.append([method,ds,f,best_model['tr_auc'],best_model['val_auc'],best_model['test_auc']])

        df = pd.DataFrame(data,columns=columns)
        grouped_df_mean = df.groupby(['model','data'])[['tr_auc','val_auc','test_auc']].mean().reset_index()
        grouped_df_std = df.groupby(['model','data'])[['tr_auc','val_auc','test_auc']].std().reset_index()

        grouped_df_mean['AUC']=grouped_df_mean['test_auc'].apply(lambda x: rf'${round(x,3)} \pm')+grouped_df_std['test_auc'].apply(lambda x: f'{round(x,3)}$')
        save = grouped_df_mean[['model','data','tr_auc','val_auc','AUC']]
        save.to_latex(buf=f"{ds}.tex", escape=False)

