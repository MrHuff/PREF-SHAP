from utils.utils import *

if __name__ == '__main__':

    train_params={
        'dataset':'website_data',
        'fold':0,
        'epochs':100,
        'patience':5,
        'model_string':'GPGP_exact',
        'bs':1000
    }
    c=train_GP(train_params=train_params,m=1000)
    c.train_model()

