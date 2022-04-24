
from shap_pipeline import *
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

if __name__ == '__main__':
    interventional = False
    model = 'SGD_krr'
    for job in ['chameleon_wl','pokemon_wl']:
        data_container = []
        num_matches= -1
        # for f in [0, 1, 2]:
        for f in [0,1,2]:
            train_params = {
                'dataset': job,
                'fold': f,
                'epochs': 100,
                'patience': 5,
                'model_string': 'SGD_krr',  # krr_vanilla
                'bs': 1000,
                'double_up':False,
                'm_factor':1.0
            }
            c= train_GP(train_params=train_params)
            c.load_and_split_data()

            x,x_prime = torch.from_numpy(c.left_val).float(),torch.from_numpy(c.right_val).float()
            y=torch.from_numpy((c.y_val > 0)*1.0)[0:num_matches].unsqueeze(-1)

            if x.shape[0] < 100:
                x, x_prime = torch.from_numpy(c.left_tr).float(), torch.from_numpy(c.right_tr).float()
                y = torch.from_numpy((c.y_tr > 0) * 1.0)[0:num_matches].unsqueeze(-1)

            shap_l, shap_r = x[0:num_matches, :], x_prime[0:num_matches, :]
            winners = y * shap_r + (1-y)*shap_l
            loosers = (1-y)*shap_r + y *shap_l
            sum_count,features_names,do_sum,coeffs=return_feature_names(job)
            # diff_abs = np.abs(winners -loosers)
            diff_abs = winners -loosers
            data = cumsum_thingy_2(sum_count,diff_abs)
            df = pd.DataFrame(data,columns=features_names)
            df['fold'] = f
            data_container.append(df)
        big_df = pd.concat(data_container,axis=0).reset_index(drop=True)
        if not os.path.exists(f'{interventional}_{job}'):
            os.makedirs(f'{interventional}_{job}')
        big_df.to_csv(f'{interventional}_{job}/data_folds.csv')





