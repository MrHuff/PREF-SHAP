import pandas as pd
import pickle

if __name__ == '__main__':
    data=[]
    columns = ['model','data','fold','val_auc','test_auc']
    for method in ['SGD_krr']:
        for ds in ['pokemon']:
            for f in [0, 1, 2, 3, 4]:
                with open(f'{ds}_results/{method}/run_{f}.pickle', 'rb') as handle:
                    best_model = pickle.load(handle)
                    data.append([method,ds,f,best_model['val_auc'],best_model['test_auc']])

    df = pd.DataFrame(data,columns=columns)
    grouped_df_mean = df.groupby(['model','data'])[['val_auc','test_auc']].mean().reset_index()
    grouped_df_std = df.groupby(['model','data'])[['val_auc','test_auc']].std().reset_index()

    grouped_df_mean['AUC']=grouped_df_mean['test_auc'].apply(lambda x: rf'${round(x,3)} \pm')+grouped_df_std['test_auc'].apply(lambda x: f'{round(x,3)}$')
    save = grouped_df_mean[['model','data','val_auc','AUC']]
    save.to_latex(buf=f"final_results_pokemon.tex", escape=False)

