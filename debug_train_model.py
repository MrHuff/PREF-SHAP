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
        'dataset':'website_data',
        'fold':0,
        'epochs':100,
        'patience':5,
        'model_string':'krr_GPGP',
        'bs':1000
    }
    c=train_GP(train_params=train_params,m=1000)
    c.train_model()

