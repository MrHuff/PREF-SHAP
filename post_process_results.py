import pandas as pd
import pickle
import dill
if __name__ == '__main__':
    columns = ['model','data','fold','tr_auc','val_auc','test_auc']

    dict_job = [       {'data':['alan_data_5000_100'] , 'models': ['SGD_base']},
        {'data':['chameleon_wl', 'pokemon_wl','alan_data_5000_100'] , 'models': ['SGD_krr', 'SGD_krr_pgp']},
                {'data':['website_user_data_wl','tennis_data_processed_wl'] ,'models': ['SGD_ukrr','SGD_ukrr_pgp']}
                ]
    data = []
    for d in dict_job:
        ds_list = d['data']
        method_list = d['models']
        for ds in ds_list:
            for method in method_list:
                for f in [0, 1, 2,]:
                    with open(f'{ds}_results/{method}/run_{f}.pickle', 'rb') as handle:
                        loaded_model = pickle.load(handle)
                    best_model = dill.loads(loaded_model)
                    data.append([method,ds,f,best_model['tr_auc'],best_model['val_auc'],best_model['test_auc']])

        df = pd.DataFrame(data,columns=columns)
        grouped_df_mean = df.groupby(['data','model'])[['tr_auc','val_auc','test_auc']].mean().reset_index()
        grouped_df_std = df.groupby(['data','model'])[['tr_auc','val_auc','test_auc']].std().reset_index()

        grouped_df_mean['Test AUC']=grouped_df_mean['test_auc'].apply(lambda x: rf'${round(x,3)} \pm')+grouped_df_std['test_auc'].apply(lambda x: f'{round(x,3)}$')
        grouped_df_mean['Val AUC']=grouped_df_mean['val_auc'].apply(lambda x: rf'${round(x,3)} \pm')+grouped_df_std['val_auc'].apply(lambda x: f'{round(x,3)}$')
        grouped_df_mean['Tr AUC']=grouped_df_mean['tr_auc'].apply(lambda x: rf'${round(x,3)} \pm')+grouped_df_std['tr_auc'].apply(lambda x: f'{round(x,3)}$')
        save = grouped_df_mean[['model','data','Tr AUC','Val AUC','Test AUC']]
        save.to_latex(buf=f"all.tex", escape=False)

