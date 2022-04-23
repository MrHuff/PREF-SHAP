import os.path
import pickle
import dill
import torch

from GPGP.GP_model import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
import tqdm
from GPGP.KRR_model import SGD_KRR,SGD_UKRR,SGD_UKRR_PGP,SGD_KRR_PGP


def sq_dist( x1, x2):
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment
    x2 = x2 - adjustment  # x1 and x2 should be identical in all dims except -2 at this point

    # Compute squared distance matrix using quadratic expansion
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)
    x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))
    # Zero out negative values
    res.clamp_min_(0)

    # res  = torch.cdist(x1,x2,p=2)
    # Zero out negative values
    # res.clamp_min_(0)
    return res


def covar_dist( x1, x2):
    return sq_dist(x1, x2).sqrt()


def get_median_ls(X, Y=None):
    with torch.no_grad():
        if Y is None:
            d = covar_dist(x1=X, x2=X)
        else:
            d = covar_dist(x1=X, x2=Y)
        ret = torch.sqrt(torch.median(d[d >= 0]))  # print this value, should be increasing with d
        if ret.item() == 0:
            ret = torch.tensor(1.0)
        return ret

class general_chunk_iterator():
    def __init__(self,X,y,shuffle,batch_size):
        self.X = X
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.n = self.X.shape[0]
        self.chunks=self.n//batch_size+1
        self.perm = torch.randperm(self.n)
        if self.shuffle:
            self.X = self.X[self.perm,:]
            self.y = self.y[self.perm,:]
        self._index = 0
        self.it_X = torch.chunk(self.X,self.chunks)
        self.it_y = torch.chunk(self.y,self.chunks)
        self.true_chunks = len(self.it_X)

    def __next__(self):
        ''''Returns the next value from team object's lists '''
        if self._index < self.true_chunks:
            result = (self.it_X[self._index],self.it_y[self._index])
            self._index += 1
            return result
        # End of Iteration
        raise StopIteration

    def __len__(self):
        return len(self.it_X)

class custom_dataloader():
    def __init__(self,dataset,batch_size=32,shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = self.dataset.X.shape[0]
        self.len=self.n//batch_size+1
    def __iter__(self):
        return general_chunk_iterator(X =self.dataset.X,
                              y = self.dataset.y,
                              shuffle = self.shuffle,
                              batch_size=self.batch_size,
                              )
    def __len__(self):
        self.n = self.dataset.X.shape[0]
        self.len = self.n // self.batch_size + 1
        return self.len

class general_dataset():
    def __init__(self,X_tr,y_tr,X_val,y_val,X_test,y_test):
        self.train_X=torch.from_numpy(X_tr).float()
        self.train_y=torch.from_numpy(y_tr).float()
        self.val_X=torch.from_numpy(X_val).float()
        self.val_y=torch.from_numpy(y_val).float()
        self.test_X=torch.from_numpy(X_test).float()
        self.test_y=torch.from_numpy(y_test).float()

    def set(self, mode='train'):
        self.X = getattr(self, f'{mode}_X')
        self.y = getattr(self, f'{mode}_y')

    def __getitem__(self, index):
        return self.X[index, :], self.y[index]

    def __len__(self):
        return self.X.shape[0]


class StratifiedKFold3(StratifiedKFold):
    def split(self, X, y, groups=None):
        s = super().split(X, y, groups)
        fold_indices=[]
        for train_indxs, test_indxs in s:
            y_train = y[train_indxs]
            train_indxs, cv_indxs = train_test_split(train_indxs,random_state=42,stratify=y_train, test_size=(1 / (self.n_splits - 1)))
            # yield train_indxs, cv_indxs, test_indxs
            fold_indices.append((train_indxs, cv_indxs, test_indxs))
        return fold_indices

def load_data(data_dir_load,files):
    with open(data_dir_load + '/' + files, 'rb') as handle:
        data_dict = pickle.load(handle)
    return data_dict

def save_data(data_dir_load,files,u,u_prime,y):
    with open(data_dir_load + '/' + files, 'rb') as handle:
        pickle.dump({ 'X':u,'X_prime':u_prime,'Y':y}, handle, protocol=pickle.HIGHEST_PROTOCOL)

class train_GP():
    def __init__(self, train_params, device='cuda:0'):
        self.device=device
        self.dataset_string = train_params['dataset']
        self.fold = train_params['fold']
        self.epochs=train_params['epochs']
        self.double_up = train_params['double_up']
        self.patience=train_params['patience']
        self.model_string = train_params['model_string']
        self.bs = train_params['bs']
        self.save_dir = f'{self.dataset_string}_results/{self.model_string}/'
        self.m=train_params['m_factor']
        self.load_and_split_data()
        self.init_model()

    def init_model(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if self.model_string=='SGD_krr':
            self.model = SGD_KRR
        if self.model_string=='SGD_ukrr':
            self.model = SGD_UKRR
        if self.model_string=='SGD_ukrr_pgp':
            self.model = SGD_UKRR_PGP
        if self.model_string=='SGD_krr_pgp':
            self.model = SGD_KRR_PGP


    def load_and_split_data(self):
        l_load=np.load(self.dataset_string+'/l_processed.npy',allow_pickle=True)
        r_load=np.load(self.dataset_string+'/r_processed.npy',allow_pickle=True)
        y_load=np.load(self.dataset_string+'/y.npy',allow_pickle=True)
        if self.double_up:
            l = np.concatenate([l_load,r_load],axis=0)
            r = np.concatenate([r_load,l_load],axis=0)
            y = np.concatenate([y_load,-1.*y_load],axis=0)
        else:
            l = l_load
            r = r_load
            y = y_load

        if self.model_string in ['SGD_ukrr','SGD_ukrr_pgp']:
            u_load = np.load(self.dataset_string + '/u.npy',allow_pickle=True)
            if self.double_up:
                u = np.concatenate([u_load,u_load],axis=0)
            else:
                u = u_load
            self.ulen=u.shape[1]
            S_u = np.load(self.dataset_string + '/S_u.npy',allow_pickle=True)
            s_u_scaler = StandardScaler()
            S_u = s_u_scaler.fit_transform(S_u)
            self.S_u = torch.from_numpy(S_u).float()
        S= np.load(self.dataset_string+'/S.npy',allow_pickle=True)
        s_scaler = StandardScaler()
        S = s_scaler.fit_transform(S)
        self.S=torch.from_numpy(S).float()

        indices = np.arange(y.shape[0])
        tr_ind,val_ind,test_ind = StratifiedKFold3(n_splits=10,shuffle=True,random_state=1337).split(indices,y)[self.fold]
        if self.model_string in ['SGD_ukrr','krr_user','SGD_ukrr_pgp']:
            scaler_u = StandardScaler()
            self.tr_u = scaler_u.fit_transform(u[tr_ind])
            self.val_u = scaler_u.transform(u[val_ind])
            self.test_u = scaler_u.transform(u[test_ind])
        # test_set = np.concatenate([val_ind,test_ind],axis=0)
        scaler=StandardScaler()
        self.left_tr,self.right_tr = l[tr_ind],r[tr_ind]
        self.left_tr=scaler.fit_transform(self.left_tr)
        self.right_tr=scaler.fit_transform(self.right_tr)
        self.left_val,self.right_val = l[val_ind],r[val_ind]
        self.left_test,self.right_test = l[test_ind],r[test_ind]
        self.left_val=scaler.fit_transform(self.left_val)
        self.right_val=scaler.fit_transform(self.right_val)
        self.left_test=scaler.fit_transform(self.left_test)
        self.right_test=scaler.fit_transform(self.right_test)

        self.y_tr = y[tr_ind]
        self.y_val = y[val_ind]
        self.y_test = y[test_ind]

        if self.model_string in ['SGD_ukrr','krr_user','SGD_ukrr_pgp']:
            self.X_tr = np.concatenate([self.tr_u,self.left_tr, self.right_tr], axis=1)
            self.X_val = np.concatenate([self.val_u,self.left_val, self.right_val], axis=1)
            self.X_test = np.concatenate([self.test_u,self.left_test, self.right_test], axis=1)
        else:
            self.X_tr = np.concatenate([self.left_tr,self.right_tr],axis=1)
            self.X_val = np.concatenate([self.left_val,self.right_val],axis=1)
            self.X_test = np.concatenate([self.left_test,self.right_test],axis=1)

        self.dataset=general_dataset(X_tr=self.X_tr,y_tr=self.y_tr,X_val=self.X_val,y_val=self.y_val,X_test=self.X_test,y_test=self.y_test)
        self.dataset.set('train')
        self.dataloader= custom_dataloader(self.dataset,batch_size=self.bs)

    def SGD_krr_loop(self):
        train_X=self.dataloader.dataset.train_X
        train_y=self.dataloader.dataset.train_y
        val_X=self.dataloader.dataset.val_X
        val_y=self.dataloader.dataset.val_y
        test_X=self.dataloader.dataset.test_X
        test_y=self.dataloader.dataset.test_y

        if self.model_string in ['SGD_ukrr','SGD_ukrr_pgp']:
            base_ls_i = get_median_ls(train_X[:,self.ulen:])
            base_ls_u = get_median_ls(train_X[:,:self.ulen])
            inp = (base_ls_u,base_ls_i,self.ulen)
        else:
            inp = get_median_ls(train_X)
        best_model,best_tr,best_val,best_test = self.model(train_X, train_y, val_X, val_y, test_X, test_y, inp, 1e-5,self.m)
        alpha = best_model.get_alpha()
        if self.model_string in ['SGD_ukrr','SGD_ukrr_pgp']:
            ind_points_all = best_model.centers.detach().cpu()
            results = {'model':best_model,
                        'test_auc':best_test ,'val_auc':best_val,'tr_auc':best_tr,
                       'ls_i':best_model.kernel.lengthscale_items.detach().cpu(),
                       'ls_u':best_model.kernel.lengthscale_users.detach().cpu(),
                       'lamb':best_model.penalty.detach().cpu(),
                       'alpha':alpha.cpu(),
                       'inducing_points_i':ind_points_all[:,self.ulen:],
                       'inducing_points_u':ind_points_all[:,:self.ulen],
                       }
        else:
            results = {'model':best_model,
                'test_auc':best_test ,'val_auc':best_val,'tr_auc':best_tr,
                       'ls':best_model.kernel.lengthscale.detach().cpu(),
                       'lamb':best_model.penalty.detach().cpu(),
                       'alpha':alpha.cpu(),
                       'inducing_points':best_model.centers.detach().cpu()}
        return results

    def train_model(self):
        if self.model_string in ['SGD_krr_pgp','SGD_krr','SGD_ukrr','SGD_ukrr_pgp']:
            results = self.SGD_krr_loop()
            model_copy = dill.dumps(results)
            pickle.dump(model_copy,
                        open(self.save_dir + f'run_{self.fold}.pickle',
                             "wb"))














