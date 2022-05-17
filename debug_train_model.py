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
    #REAL DATA
    # train_params={
    #     'dataset':'website_data',
    #     'fold':0,
    #     'epochs':100,
    #     'patience':5,
    #     'model_string':'SGD_krr', #krr_vanilla
    #     'bs':1000,
    #     'double_up': True,
    #     'm_factor': 5.,
    # 'seed': 1,
    # 'folds': 10,
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
        'model_string':'SGD_krr', #krr_vanilla
        'bs':1000,
        'double_up': True,
        'm_factor': 5.,
        'seed': 42,
        'folds': 10,
    }
    # for ds in [['hard_data_10000_1000_5_3',2.0],['alan_data_5000_1000_10_10',2.0],['toy_data_5000_10_2',2.0]]:
    # for ds in [['hard_data_10000_1000_10_5',10.0]]:
    # for ds in [['alan_data_5000_1000_10_10',1.0],['toy_data_5000_10_2',1.0]]:
    for ds in [['pokemon_wl',1.0]]:
        for method in ['SGD_base']:
            # for f in [0,1,2]:
            for f in [0]:
                train_params['dataset']=ds[0]
                train_params['model_string']=method
                train_params['fold']=f
                train_params['m_factor']=ds[1]

                c= train_GP(train_params=train_params)
                c.train_model()