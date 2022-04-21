from utils.utils import *
# import pykeops
# pykeops.clean_pykeops()

import os
# pykeops.verbose = True
# pykeops.build_type = 'Debug'

# Clean up the already compiled files
# pykeops.clean_pykeops()
#
# pykeops.test_torch_bindings()

if __name__ == '__main__':

    train_params={
        'dataset':'website_data_user',
        'fold':0,
        'epochs':100,
        'patience':5,
        'model_string':'SGD_ukrr', #krr_vanilla
        'bs':1000,
        'double_up':True,
    }
    for method in ['SGD_ukrr','SGD_ukrr_pgp']:
        for ds in ['website_user_data_wl']:
            for f in [0,1,2]:
                train_params['dataset']=ds
                train_params['model_string']=method
                train_params['fold']=f
                c=train_GP(train_params=train_params, m_fac=5.0)
                c.train_model()

