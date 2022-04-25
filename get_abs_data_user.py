
from user_shap_pipeline import *
sns.set()



def cumsum_thingy_2(cumsum_indices,shapley_vals):
    cat_parts = []
    for i in range(len(cumsum_indices)-1):
        part = shapley_vals[:,cumsum_indices[i]:cumsum_indices[i+1]].sum(1,keepdim=True)
        cat_parts.append(part)
    p_output = torch.cat(cat_parts,dim=1)
    return p_output


def process_data(x,x_prime,job,case,f):
    sum_count, features_names, do_sum, coeffs = return_feature_names(job, case=case)
    # diff_abs = np.abs(x - x_prime)
    diff_abs = x - x_prime
    data = cumsum_thingy_2(sum_count, diff_abs)
    df = pd.DataFrame(data, columns=features_names)
    df['fold'] = f
    return df

def process_data_u(u,job,case,f):
    sum_count, features_names, do_sum, coeffs = return_feature_names(job, case=case)
    data = cumsum_thingy_2(sum_count, u)
    df = pd.DataFrame(data, columns=features_names)
    df['fold'] = f
    return df
if __name__ == '__main__':
    interventional = False
    folds = [0, 1, 2]
    num_matches = -1
    for j in ['tennis_data_processed_wl','website_user_data_wl']:
        job = j
        model = 'SGD_ukrr'
        data_container = []
        data_container_u = []
        for f in folds:
            train_params = {
                'dataset': job,
                'fold': f,
                'epochs': 100,
                'patience': 5,
                'model_string': 'SGD_ukrr',  # krr_vanilla
                'bs': 1000,
                'double_up': False,
                'm_factor': 5.,
                'seed': 42,
                'folds': 10,
            }
            c= train_GP(train_params=train_params)
            c.load_and_split_data()
            x, x_prime = torch.from_numpy(c.left_val).float(), torch.from_numpy(c.right_val).float()
            u = torch.from_numpy(c.val_u).float()
            u_shap = u[0:num_matches, :]
            shap_l, shap_r = x[0:num_matches, :], x_prime[0:num_matches, :]
            df = process_data(shap_l,shap_r,job,2,f)
            df_u = process_data_u(u_shap,j,1,f)
            data_container.append(df)
            data_container_u.append(df_u)
        big_df = pd.concat(data_container,axis=0).reset_index(drop=True)
        big_df_u = pd.concat(data_container_u,axis=0).reset_index(drop=True)
        if not os.path.exists(f'{interventional}_{job}'):
            os.makedirs(f'{interventional}_{job}')
        big_df.to_csv(f'{interventional}_{job}/data_folds.csv')
        big_df_u.to_csv(f'{interventional}_{job}/data_folds_user.csv')






