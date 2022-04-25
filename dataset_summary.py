import os.path
import pickle
import dill
import torch
import numpy as np


if __name__ == '__main__':
    ls = ['chameleon_wl','pokemon_wl','tennis_data_processed_wl','website_user_data_wl']
    data = {}
    for dataset_string in ls:
        data[dataset_string]={}
        l_load=np.load(dataset_string+'/l_processed.npy',allow_pickle=True)
        S= np.load(dataset_string+'/S.npy',allow_pickle=True)
        data[dataset_string]['n'] = l_load.shape[0]
        data[dataset_string]['n_S'] = S.shape[0]

        if dataset_string in ['tennis_data_processed_wl','website_user_data_wl']:
            S_u = np.load(dataset_string + '/S_u.npy',allow_pickle=True)
            data[dataset_string]['n_S_u'] = S_u.shape[0]
    print(data)
    pickle.dump(data,
                open( f'dataset_summary.pickle',
                     "wb"))
    #item and user
    #n
    #n_{x_S}
    #cont cols
    #cat cols
