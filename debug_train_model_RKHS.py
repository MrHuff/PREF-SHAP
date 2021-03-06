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
    # }
    # # for ds in [['chameleon_wl',5.0],['pokemon_wl',2.0]]:
    # for ds in [['pokemon_wl',2.0]]:
    #     for method in ['SGD_krr','SGD_krr_pgp']:
    #     # for method in ['SGD_krr_pgp']:
    #     #     for f in [0,1,2,3,4]:
    #         for f in [0]:
    #             train_params['dataset']=ds[0]
    #             train_params['model_string']=method
    #             train_params['fold']=f
    #             train_params['m_factor']=ds[1]
    #
    #             c= train_GP(train_params=train_params)
    #             c.train_model()
    #TOY
    train_params={
        'dataset':'website_data',
        'fold':0,
        'epochs':100,
        'patience':5,
        'bs':1000,
        'double_up': False,
        'm_factor': 5.,
        'seed': 42,
        'folds': 5,
    }
    # for ds in [['LOL',1.0],['census',1.0]]:
    for ds in [['census',1.0]]:
        # for f in [0,1,2]:
        for f in [0]:
            train_params['dataset']=ds[0]
            train_params['model_string']='SGD_base'
            train_params['fold']=f
            train_params['m_factor']=ds[1]
            c= train_krr_simple(train_params=train_params)
            c.train_model()