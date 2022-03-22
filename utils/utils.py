import os.path
import pickle

import torch

from GPGP.GP_model import *
import numpy as np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import tqdm
from GPGP.KRR_model import train_KRR,train_KRR_PGP,train_vanilla_KRR


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
            train_indxs, cv_indxs = train_test_split(train_indxs,stratify=y_train, test_size=(1 / (self.n_splits - 1)))
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
    def __init__(self,train_params,m=10,device='cuda:0'):
        self.device=device
        self.dataset_string = train_params['dataset']
        self.fold = train_params['fold']
        self.epochs=train_params['epochs']
        self.patience=train_params['patience']
        self.model_string = train_params['model_string']
        self.bs = train_params['bs']
        self.save_dir = f'{self.dataset_string}_results/{self.model_string}/'
        self.m=m
        self.load_and_split_data()
        self.init_model()

    def init_model(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        inducing_points = self.dataset.train_X[torch.randperm(self.dataset.train_X.shape[0])[:self.m], :].to(self.device)
        if self.model_string=='GPGP_exact':
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self.model = ExactGPGP(likelihood=self.likelihood,train_x=self.dataset.train_X,train_y=self.dataset.train_y)
        if self.model_string=='GPGP_approx':
            self.model = ApproximateGPGP(inducing_points=inducing_points,dim=inducing_points.shape[1])
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        if self.model_string=='krr_GPGP':
            self.model = train_KRR
        if self.model_string=='krr_PGP':
            self.model = train_KRR_PGP
        if self.model_string=='krr_vanilla':
            self.model = train_vanilla_KRR
        if self.model_string=='PGP_exact':
            pass
        if self.model_string=='PGP_approx':
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        if self.model_string=='vanilla_exact':
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
            self.model = ExactVanilla(likelihood=self.likelihood,train_x=self.dataset.train_X,train_y=self.dataset.train_y)
        if self.model_string=='vanilla_approx':
            self.model = ApproximateVanilla(inducing_points=inducing_points,dim=inducing_points.shape[1])


    def load_and_split_data(self):
        l=np.load(self.dataset_string+'/l_processed.npy')
        r=np.load(self.dataset_string+'/r_processed.npy')
        y=np.load(self.dataset_string+'/y.npy')
        indices = np.arange(y.shape[0])
        tr_ind,val_ind,test_ind = StratifiedKFold3(5).split(indices,y)[self.fold]
        # test_set = np.concatenate([val_ind,test_ind],axis=0)
        self.left_tr,self.right_tr = l[tr_ind],r[tr_ind]
        self.left_val,self.right_val = l[val_ind],r[val_ind]
        self.left_test,self.right_test = l[test_ind],r[test_ind]
        self.y_tr = y[tr_ind]
        self.y_val = y[val_ind]
        self.y_test = y[test_ind]

        self.X_tr = np.concatenate([self.left_tr,self.right_tr],axis=1)
        self.X_val = np.concatenate([self.left_val,self.right_val],axis=1)
        self.X_test = np.concatenate([self.left_test,self.right_test],axis=1)

        self.dataset=general_dataset(X_tr=self.X_tr,y_tr=self.y_tr,X_val=self.X_val,y_val=self.y_val,X_test=self.X_test,y_test=self.y_test)
        self.dataset.set('train')
        self.dataloader= custom_dataloader(self.dataset,batch_size=self.bs)


    def full_gp_loop(self,optimizer,mll):
        self.model.train()
        self.likelihood.train()
        optimizer.zero_grad()
        # Output from model
        X = self.dataset.train_X.to(self.device)
        y= self.dataset.train_y.to(self.device)
        output = self.model(X)
        # Calc loss and backprop gradients
        loss = -mll(output, y)
        loss.backward()
        optimizer.step()
        return loss.item(),self.model.covar_module.base_kernel.lengthscale.item(),self.model.likelihood.noise.item()

    def full_krr_loop(self):
        lambdas =np.linspace(1e-5,1,25)
        factors =np.linspace(0.25,2,25)
        train_X=self.dataloader.dataset.train_X
        train_y=self.dataloader.dataset.train_y
        val_X=self.dataloader.dataset.val_X
        val_y=self.dataloader.dataset.val_y
        test_X=self.dataloader.dataset.test_X
        test_y=self.dataloader.dataset.test_y
        base_ls  = get_median_ls(train_X)

        val_scores = []
        test_scores = []
        params=[]
        model_list = []
        for l in lambdas:
            for f in factors:
                model,val_score,test_score = self.model(train_X,train_y,val_X,val_y,test_X,test_y,base_ls*f,l)
                val_scores.append(val_score)
                test_scores.append(test_score)
                model_list.append(model)
                params.append([base_ls*f,l])
        best_ind=np.argmax(test_scores)
        best_test=test_scores[best_ind]
        best_val=val_scores[best_ind]
        best_params=params[best_ind]
        best_model = model_list[best_ind]

        results = {'test_auc':best_test ,'val_auc':best_val,'ls':best_params[0],'lamb':best_params[1],'model':best_model,'alpha':best_model.alpha_,'inducing_points':best_model.ny_points_}
        return results

    def full_approx_loop(self,optimizer,mll):
        self.model.train()
        self.likelihood.train()
        self.dataloader.dataset.set('train')
        pbar = tqdm.tqdm(self.dataloader)
        for i, (x_batch, y_batch) in enumerate(pbar):
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            optimizer.zero_grad()
            output= self.model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

    def train_model(self):

        if self.model_string in ['GPGP_exact','PGP_exact','vanilla_exact']:
            self.model = self.model.to(self.device)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
            pbar = tqdm.tqdm(range(self.epochs))
            for i,j in enumerate(pbar):
                l,ls,noise=self.full_gp_loop(optimizer,mll)
                pbar.set_description(f"loss: {l} ls: {ls} noise: {noise}")

        if self.model_string in ['PGP_approx','GPGP_approx','vanilla_approx']:
            self.model = self.model.to(self.device)
            optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.likelihood.parameters()},
            ], lr=0.01)

            # Our loss object. We're using the VariationalELBO
            mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=self.dataset.train_y.size(0))
            pbar = tqdm.tqdm(range(self.epochs))
            for i,j in enumerate(pbar):
                self.full_approx_loop(optimizer,mll)

        if self.model_string in ['krr_GPGP','krr_PGP','krr_vanilla']:
            results = self.full_krr_loop()
            print(results)
            pickle.dump(results,
                        open(self.save_dir + f'run_{self.fold}.pickle',
                             "wb"))

    def validate_model(self,mode='val'):
        if 'exact' in self.model_string:

            self.model.eval()
            self.likelihood.eval()
            if mode=='val':
                test_data= self.dataset.val_X.to(self.device)
            else:
                test_data= self.dataset.test_X.to(self.device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = self.likelihood(self.model(test_data))
                mean = observed_pred.mean
                lower, upper = observed_pred.confidence_region()
        else:
            self.model.eval()
            self.likelihood.eval()
            self.dataloader.dataset.set(mode)
            pbar = tqdm.tqdm(self.dataloader)
            big_pred=[]
            with torch.no_grad():
                for i,(x_batch, y_batch) in enumerate(pbar):
                    preds = self.model(x_batch)
                    big_pred.append(preds.cpu())
            big_pred=torch.cat(big_pred,dim=0)













