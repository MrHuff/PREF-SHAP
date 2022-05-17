import numpy as np
from itertools import combinations
import random
import torch
from sklearn.linear_model import Ridge, Lasso,ElasticNet
from GPGP.kernel import *
from tqdm import tqdm
from scipy.special import binom
from pref_shap.tensor_CG import tensor_CG
from numpy.random import default_rng
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class ElasticLinear(pl.LightningModule):
    def __init__(
            self, n_inputs: int = 1,n_outputs:int = 10, learning_rate=0.05, l1_lambda=0.05, l2_lambda=0.05
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.output_layer = torch.nn.Linear(n_inputs, n_outputs)
        self.train_log = []

    def forward(self, x):
        outputs = self.output_layer(x)
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def l1_reg(self):
        l1_norm = self.output_layer.weight.abs().sum()

        return self.l1_lambda * l1_norm

    def l2_reg(self):
        l2_norm = self.output_layer.weight.pow(2).sum()

        return self.l2_lambda * l2_norm

    def weighted_mse_loss(self,input, target, weight):
        return (weight * (input - target) ** 2).mean()

    def training_step(self, batch, batch_idx):
        x, y_dat = batch
        w = y_dat[:,0].unsqueeze(-1)
        y = y_dat[:,1:]
        y_hat = self(x)
        loss = self.weighted_mse_loss(y_hat, y,w) + self.l1_reg() + self.l2_reg()
        self.log("loss", loss)
        self.train_log.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_dat = batch
        w = y_dat[:, 0].unsqueeze(-1)
        y = y_dat[:, 1:]
        y_hat = self(x)
        loss = self.weighted_mse_loss(y_hat, y, w) + self.l1_reg() + self.l2_reg()
        self.log("val_loss", loss)
def elastic_regression(l1_weight,l2_weight,X,w,y):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=25, verbose=False, mode="min")
    y_dat = torch.cat([w,y],dim=1)
    model = ElasticLinear(
        n_inputs=X.shape[1],
        n_outputs=y.shape[1],
        l1_lambda=l1_weight,
        l2_lambda=l2_weight,
        learning_rate=0.05,
    )
    n = X.shape[0]
    n_tr = int(round(n*0.9))
    X_tr = X[:n_tr,:] #shapley vals x dims
    y_tr = y_dat[:n_tr,:] #shapley vals x number of observations you are doing
    X_te = X[n_tr:,:]
    y_te = y_dat[n_tr:,:]
    ###Fix this
    dataset_train = TensorDataset(X_tr, y_tr)
    dataloader_train = DataLoader(dataset_train, batch_size=X_tr.shape[0]//10, shuffle=True)
    dataset_test = TensorDataset(X_te, y_te)
    dataloader_test = DataLoader(dataset_test, batch_size=X_tr.shape[0], shuffle=True)
    trainer = pl.Trainer(max_epochs=100,accelerator="gpu", devices=[0],callbacks=early_stop_callback)
    trainer.fit(model, dataloader_train, dataloader_test)
    shapley_vals = model.output_layer.weight.detach().cpu().t()
    return shapley_vals

def base_10_base_2(indices: np.array,d:int=10):
    S= np.zeros((indices.shape[0],d))
    rest = indices
    valid_rows = rest>0
    while True:
        set_to_1 =  np.floor(np.log2(rest)).astype(int)
        set_to_1_prime=set_to_1[valid_rows][:,np.newaxis]
        p = S[valid_rows,:]
        np.put_along_axis(p, set_to_1_prime, 1, axis=1)
        S[valid_rows, :] = p
        # S[valid_rows,:][:,set_to_1_prime]=1
        rest = rest-2**(np.clip(set_to_1,0,np.inf))
        valid_rows = rest>0
        if valid_rows.sum()==0:
            return S

def base_2_base_10(N:int=2500,d:int=10):
    P = np.random.binomial(size=(N,d), n=1, p=0.5)
    b = 2 ** np.arange(0, d)
    unique_ref = (b*P).sum(1)
    _,idx=np.unique(unique_ref,return_index=True)
    return P[idx,:]

def sample_Z(D,max_S):
    max_range = min(2**D,2**63-1)
    if max_S>=max_range:
        configs= np.arange(max_range)
        return base_10_base_2(configs, D)
    else:
        return base_2_base_10(max_S, D)

class pref_shap():
    def __init__(self, model, alpha, k, X_l, X_r, X, k_U=None, u=None, X_U=None, max_S: int = 5000, rff_mode=False,
                 eps=1e-3, cg_max_its=10, lamb=1e-3, max_inv_row=0, cg_bs=20, post_method='OLS', interventional=False,
                 device='cuda:0'):
        if max_inv_row >0:
            X = X[torch.randperm(X.shape[0])[:max_inv_row],:]
        self.model=model
        self.post_method=post_method
        self.alpha = alpha.t()
        self.cg_bs=cg_bs
        self.X_l,self.X_r,self.X= X_l.to(device),X_r.to(device),X.to(device)
        self.max_S=max_S
        self.eps =eps
        self.cg_max_its=cg_max_its
        if X_U is not None:
            if max_inv_row > 0:
                X_U = X_U[torch.randperm(X_U.shape[0])[:max_inv_row], :]
            self.u = u.to(device)
            self.X_U = X_U.to(device)
            self.k_U = k_U
        self.device = device
        self.lamb=lamb

        self.rff=rff_mode
        self.interventional = interventional
        self.k = k


    def setup_user_vals(self):
        self.N_x,self.m = self.X_U.shape
        self.reg = self.N_x*self.lamb
        self.eye = torch.eye(self.N_x).to(self.device)*self.reg
        L = torch.linalg.cholesky(self.k_U(self.X_U)+ self.eye)
        self.precond = torch.cholesky_inverse(L)

        self.setup_the_rest()

    def setup_the_rest(self):
        self.tensor_CG = tensor_CG(precond=self.precond,reg=self.reg,eps=self.eps,maxits=self.cg_max_its,device=self.device)
        Z = torch.from_numpy(sample_Z(self.eff_dim,self.max_S)).float().to(self.device)
        # self.m = self.eff_dim
        const = torch.lgamma(torch.tensor(self.eff_dim) + 1)
        abs_S = Z.sum(1)
        a = torch.exp(const - torch.lgamma((self.eff_dim - abs_S) + 1) - torch.lgamma(abs_S + 1))
        self.weights = (self.eff_dim - 1) / (a * (abs_S) * (self.eff_dim - abs_S))
        bool = torch.isfinite(self.weights.squeeze())
        # edge_vals = torch.sum(self.weights[bool]).unsqueeze(0).to(self.device)
        edge_vals = torch.tensor([1e5]).float().to(self.device)
        middle_weights = self.weights[bool]
        middle_weights = middle_weights /middle_weights.sum()
        middle_Z = Z[bool,:]
        Z = torch.concat([torch.zeros(1, self.eff_dim).to(self.device), middle_Z,torch.ones(1, self.eff_dim).to(self.device)], dim=0)
        self.Z = torch.zeros(Z.shape[0],self.m).to(self.device)
        self.Z[:,self.mask] = Z
        self.weights = torch.concat([edge_vals,middle_weights,edge_vals],dim=0)
        self.weights=self.weights.unsqueeze(-1)
        self.batched_Z = torch.chunk(self.Z,max(self.Z.shape[0]//self.cg_bs,1),dim=0)

    def setup_item_vals(self):
        self.N_x,self.m = self.X.shape
        self.reg = self.N_x*self.lamb
        self.eye = torch.eye(self.N_x).to(self.device)*self.reg
        L = torch.linalg.cholesky(self.k(self.X)+ self.eye)
        self.precond = torch.cholesky_inverse(L)
        self.setup_the_rest()

    # def weighted_moore_penrose(self):
    #     ztw = self.Z * self.weights
    #     self.L = torch.linalg.cholesky(ztw.t()@self.Z)
    #     self.zwz= torch.cholesky_inverse(L)

    def kernel_tensor_batch(self,S_list_batch,x,x_prime):
        inv_tens=[]
        vec = []
        cat_xls_xs = []
        cat_xls_xs_prime = []
        cat_xrs_xs = []
        cat_xrs_xs_prime = []
        cat_klsc_xsc = []
        cat_krsc_xsc = []
        first_flag=False
        last_flag=False
        for i in range(S_list_batch.shape[0]): #build-up
            S = S_list_batch[i,:].bool()
            S_C = ~S
            if S.sum()==0:
                first_flag=True
            elif S_C.sum()==0:
                last_flag=True
            else:
                inv_tens.append(self.k(self.X[:,S],None,S))
                x_S,x_prime_S= x[:,S],x_prime[:,S]
                xs_cat = torch.cat([x_S,x_prime_S],dim=0)
                vec_cat  = self.k(self.X[:,S], xs_cat,S)
                vec.append(vec_cat)
                stacked_lr = torch.cat([self.X_l[:,S],self.X_r[:,S]],dim=0)
                l_hadamard_mat = self.k(stacked_lr,xs_cat,S)
                r,c = l_hadamard_mat.shape
                xls_xs,xls_xs_prime,xrs_xs,xrs_xs_prime, = l_hadamard_mat[:r//2,:][:,:c//2],l_hadamard_mat[:r//2,:][:,c//2:],l_hadamard_mat[r//2:,:][:,:c//2],l_hadamard_mat[r//2:,:][:,c//2:]
                klsc_xsc = self.k( self.X_l[:,S_C],self.X[:,S_C],S_C)
                krsc_xsc = self.k( self.X_r[:,S_C],self.X[:,S_C],S_C)
                cat_xls_xs.append(xls_xs)
                cat_xls_xs_prime.append(xls_xs_prime)
                cat_xrs_xs.append(xrs_xs)
                cat_xrs_xs_prime.append(xrs_xs_prime)
                cat_klsc_xsc.append(klsc_xsc)
                cat_krsc_xsc.append(krsc_xsc)
        inv_tens=torch.stack(inv_tens,dim=0)
        vec = torch.stack(vec,dim=0)

        cg_output = self.tensor_CG.solve(inv_tens,vec) #BS x N x D
        #Write assertion error

        if torch.isnan(cg_output).any():
            cg_output=torch.nan_to_num(cg_output, nan=0.0)

        cat_klsc_xsc = torch.stack(cat_klsc_xsc,dim=0)
        cat_krsc_xsc = torch.stack(cat_krsc_xsc,dim=0)

        cg_output_a = torch.bmm(cat_klsc_xsc,cg_output)
        cg_output_b = torch.bmm(cat_krsc_xsc,cg_output)

        cg_l_a,cg_r_a = torch.chunk(cg_output_a,2,dim=-1)
        cg_l_b,cg_r_b = torch.chunk(cg_output_b,2,dim=-1)

        cat_xls_xs = torch.stack(cat_xls_xs,dim=0)
        cat_xls_xs_prime = torch.stack(cat_xls_xs_prime,dim=0)
        cat_xrs_xs = torch.stack(cat_xrs_xs,dim=0)
        cat_xrs_xs_prime = torch.stack(cat_xrs_xs_prime,dim=0)
        output = (cat_xls_xs*cat_xrs_xs_prime )* (cg_l_a * cg_r_a) - (cat_xls_xs_prime*cat_xrs_xs)*(cg_l_b*cg_r_b)

        return output,first_flag,last_flag

    def kernel_tensor_batch_pgp(self,S_list_batch,x,x_prime):
        inv_tens=[]
        vec = []
        cat_xls_xs = []
        cat_xls_xs_prime = []
        cat_xrs_xs = []
        cat_xrs_xs_prime = []
        cat_klsc_xsc = []
        cat_krsc_xsc = []
        first_flag=False
        last_flag=False
        for i in range(S_list_batch.shape[0]): #build-up
            S = S_list_batch[i,:].bool()
            S_C = ~S
            if S.sum()==0:
                first_flag=True
            elif S_C.sum()==0:
                last_flag=True
            else:
                inv_tens.append(self.k(self.X[:,S],None,S))
                x_S,x_prime_S= x[:,S],x_prime[:,S]
                xs_cat = torch.cat([x_S,x_prime_S],dim=0)
                vec_cat  = self.k(self.X[:,S], xs_cat,S)
                vec.append(vec_cat)
                stacked_lr = torch.cat([self.X_l[:,S],self.X_r[:,S]],dim=0)
                l_hadamard_mat = self.k(stacked_lr,xs_cat,S)
                r,c = l_hadamard_mat.shape
                xls_xs,xls_xs_prime,xrs_xs,xrs_xs_prime, = l_hadamard_mat[:r//2,:][:,:c//2],l_hadamard_mat[:r//2,:][:,c//2:],l_hadamard_mat[r//2:,:][:,:c//2],l_hadamard_mat[r//2:,:][:,c//2:]
                klsc_xsc = self.k( self.X_l[:,S_C],self.X[:,S_C],S_C)
                krsc_xsc = self.k( self.X_r[:,S_C],self.X[:,S_C],S_C)
                cat_xls_xs.append(xls_xs)
                cat_xls_xs_prime.append(xls_xs_prime)
                cat_xrs_xs.append(xrs_xs)
                cat_xrs_xs_prime.append(xrs_xs_prime)
                cat_klsc_xsc.append(klsc_xsc)
                cat_krsc_xsc.append(krsc_xsc)
        inv_tens=torch.stack(inv_tens,dim=0)
        vec = torch.stack(vec,dim=0)

        cg_output = self.tensor_CG.solve(inv_tens,vec) #BS x N x D
        #Write assertion error

        if torch.isnan(cg_output).any():
            cg_output=torch.nan_to_num(cg_output, nan=0.0)

        cat_klsc_xsc = torch.stack(cat_klsc_xsc,dim=0)
        cat_krsc_xsc = torch.stack(cat_krsc_xsc,dim=0)

        cg_output_a = torch.bmm(cat_klsc_xsc,cg_output)
        cg_output_b = torch.bmm(cat_krsc_xsc,cg_output)

        cg_l_a,cg_r_a = torch.chunk(cg_output_a,2,dim=-1)
        cg_l_b,cg_r_b = torch.chunk(cg_output_b,2,dim=-1)

        cat_xls_xs = torch.stack(cat_xls_xs,dim=0)
        cat_xls_xs_prime = torch.stack(cat_xls_xs_prime,dim=0)
        cat_xrs_xs = torch.stack(cat_xrs_xs,dim=0)
        cat_xrs_xs_prime = torch.stack(cat_xrs_xs_prime,dim=0)
        output = (cat_xls_xs*cg_l_a  + cat_xrs_xs_prime* cg_r_a) - (cat_xls_xs_prime*cg_l_b + cat_xrs_xs*cg_r_b)
        return output,first_flag,last_flag


    def kernel_tensor_batch_user_case_1(self,u,S_batch,x,x_prime):
        inv_tens=[]
        vec = []
        vec_c=[]
        vec_a=[]
        RH = []
        first_flag=False
        last_flag=False
        for i in range(S_batch.shape[0]):
            S = S_batch[i,:].bool()
            S_C = ~S
            if S.sum()==0:
                first_flag=True
            elif S_C.sum()==0:
                last_flag=True
            else:
                inv_tens.append(self.k_U(self.X_U[:,S],None,S))
                u_S,u_Sc= u[:,S],u[:,S_C]
                u_bar,u_bar_Sc = self.u[:,S],self.u[:,S_C]
                vec_cat  = self.k_U(self.X_U[:,S],u_S,S)
                vec_c_cat  = self.k_U(u_bar_Sc,self.X_U[:,S_C],S_C)
                vec_a_cat  = self.k_U(u_bar,u_S,S)
                RH_cat = self.k(self.X_l,x)*self.k(self.X_r,x_prime)-self.k(self.X_l,x_prime)*self.k(self.X_r,x)
                RH.append(RH_cat)
                vec.append(vec_cat)
                vec_c.append(vec_c_cat)
                vec_a.append(vec_a_cat)
        inv_tens=torch.stack(inv_tens,dim=0)
        vec = torch.stack(vec,dim=0)
        vec_c = torch.stack(vec_c,dim=0)
        vec_a = torch.stack(vec_a,dim=0)
        RH = torch.stack(RH,dim=0)

        cg_output = self.tensor_CG.solve(inv_tens,vec) #BS x N x D
        #Write assertion error

        if torch.isnan(cg_output).any():
            cg_output=torch.nan_to_num(cg_output, nan=0.0)
        LH = vec_a* torch.bmm(vec_c,cg_output)
        output = LH*RH
        return output,first_flag,last_flag

    def kernel_tensor_batch_user_case_2(self,u, S_list_batch,x,x_prime):
        output,first_flag,last_flag=self.kernel_tensor_batch(S_list_batch,x,x_prime)
        user_k = self.k_U(self.u,u)
        output = user_k*output
        return output,first_flag,last_flag
    def flag_adjustment(self, first_flag, last_flag, output):

        if first_flag:
            y_preds  = self.y_pred - self.y_pred_mean
            if len(output.shape)==1:
                output = torch.cat([ y_preds.squeeze(0),output],dim=0)
            else:
                output = torch.cat([ y_preds.t(),output],dim=0)
        if last_flag:
            if len(output.shape)==1:
                o=torch.zeros_like(output)
                output = torch.stack([output, o], dim=0)
            else:
                o=torch.zeros_like(output[0,:]).unsqueeze(0)
                output = torch.cat([output, o], dim=0)
        if len(output.shape) == 1:
            output = output.unsqueeze(0)
        return output

    def value_observation(self,S_batch,x,x_prime,u=None,case=2,pgp=False):
        if u is None:
            if pgp:
                output,first_flag,last_flag= self.kernel_tensor_batch_pgp(S_batch, x, x_prime)
            else:
                output,first_flag,last_flag= self.kernel_tensor_batch(S_batch, x, x_prime)
        else:
            if case == 1:
                output, first_flag, last_flag = self.kernel_tensor_batch_user_case_1(u, S_batch, x, x_prime)
            elif case == 2:
                output, first_flag, last_flag = self.kernel_tensor_batch_user_case_2(u, S_batch, x, x_prime)
        output = (self.alpha@output).squeeze() - self.y_pred_mean
        return self.flag_adjustment(first_flag, last_flag, output)

    def fit(self,x,x_prime,u=None,case=2,pgp=False):
        x=x.to(self.device)
        x_prime=x_prime.to(self.device)
        if u is not None:
            u = u.to(self.device)
            tmp = torch.cat([u,x,x_prime],dim=1)
            if case==1:
                self.mask = (u.var(0) > 0).cpu()
                self.eff_dim = self.mask.sum().item()
                self.setup_user_vals()
            else:
                self.mask = ((x.var(0) + x_prime.var(0)) > 0).cpu()
                self.eff_dim = self.mask.sum().item()
                self.setup_item_vals()
        else:
            self.mask = ((x.var(0) + x_prime.var(0)) > 0).cpu()
            self.eff_dim = self.mask.sum().item()
            tmp = torch.cat([x,x_prime],dim=1)
            self.setup_item_vals()

        self.y_pred = self.model.predict(tmp)
        self.y_pred_mean = self.y_pred.mean().item()

        Y_target = []
        for batch in tqdm(self.batched_Z):
            Y_target.append(self.value_observation(batch,x,x_prime,u,case,pgp=pgp))
        Y_cat = torch.cat(Y_target, dim=0)
        return Y_cat,self.weights,self.Z

def OLS_solve(Y_target,Z_in,weights,big_weight=1e5):
    mask = Z_in.sum(0)>0
    shapley_vals = torch.zeros(Z_in.shape[1],Y_target.shape[1])
    weights[0] = big_weight
    weights[-1] = big_weight
    Z = Z_in[:,mask]
    b = Z.t()@Y_target
    A = Z.t()@(Z*weights)
    L = torch.linalg.cholesky(A)
    sol = torch.cholesky_solve(b,L)
    shapley_vals[mask,:] = sol
    return shapley_vals

def construct_values(Y_cat,Z,weights,coeffs,post_method,big_weight=1e4):
    if len(Y_cat.shape) == 1:
        Y_target = weights.squeeze() * Y_cat
    else:
        Y_target = weights * Y_cat
    shap_dict={}
    shap_dict[0]=OLS_solve(Y_target,Z,weights,big_weight)
    for coeff in coeffs:
        if post_method == 'lasso':
            l1 = coeff
            l2 = 0.0
        if post_method == 'ridge':
            l1 = 0.0
            l2 = coeff
        if post_method == 'elastic':
            l1 = coeff
            l2 = coeff
        shap_dict[coeff] = elastic_regression(l1, l2, Z, weights, Y_cat)
    return shap_dict

# if __name__ == '__main__':
#     test=sample_Z(5,20000)



        # clf = Ridge(1e-5)
        # clf.fit(self.Z, Y_target, sample_weight=weights)

        # self.full_shapley_values_ = np.concatenate([clf.intercept_.reshape(-1, 1), clf.coef_], axis=1)
        # self.SHAP_LM = clf

        # return clf.coef_


    # def kernel_tensor_batch_interventional(self,S_list_batch,x,x_prime):
    #     cat_xls_xs = []
    #     cat_xls_xs_prime = []
    #     cat_xrs_xs = []
    #     cat_xrs_xs_prime = []
    #     cat_klsc_xsc = []
    #     cat_krsc_xsc = []
    #     first_flag=False
    #     last_flag=False
    #     for i in range(S_list_batch.shape[0]):
    #         S = S_list_batch[i,:].bool()
    #         S_C = ~S
    #         if S.sum()==0:
    #             first_flag=True
    #         elif S_C.sum()==0:
    #             last_flag=True
    #         else:
    #             x_S,x_prime_S= x[:,S],x_prime[:,S]
    #             xs_cat = torch.cat([x_S,x_prime_S],dim=0)
    #             stacked_lr = torch.cat([self.X_l[:,S],self.X_r[:,S]],dim=0)
    #             l_hadamard_mat = self.k(stacked_lr,xs_cat,S)
    #             r,c = l_hadamard_mat.shape
    #             xls_xs,xls_xs_prime,xrs_xs,xrs_xs_prime, = l_hadamard_mat[:r//2,:][:,:c//2],l_hadamard_mat[:r//2,:][:,c//2:],l_hadamard_mat[r//2:,:][:,:c//2],l_hadamard_mat[r//2:,:][:,c//2:]
    #             cat_xls_xs.append(xls_xs)
    #             cat_xls_xs_prime.append(xls_xs_prime)
    #             cat_xrs_xs.append(xrs_xs)
    #             cat_xrs_xs_prime.append(xrs_xs_prime)
    #             klsc_xsc = self.k( self.X_l[:,S_C],self.X[:,S_C],S_C).mean(1,keepdim=True).expand(-1,c//2)
    #             krsc_xsc = self.k( self.X_r[:,S_C],self.X[:,S_C],S_C).mean(1,keepdim=True).expand(-1,c//2)
    #             cat_klsc_xsc.append(klsc_xsc)
    #             cat_krsc_xsc.append(krsc_xsc)
    #
    #     cat_xls_xs = torch.stack(cat_xls_xs,dim=0)
    #     cat_xls_xs_prime = torch.stack(cat_xls_xs_prime,dim=0)
    #     cat_xrs_xs = torch.stack(cat_xrs_xs,dim=0)
    #     cat_xrs_xs_prime = torch.stack(cat_xrs_xs_prime,dim=0)
    #     cg_l_a=torch.stack(cat_klsc_xsc,dim=0)
    #     cg_r_a=torch.stack(cat_krsc_xsc,dim=0)
    #
    #     output = (cat_xls_xs*cat_xrs_xs_prime )* (cg_l_a * cg_r_a) - (cat_xls_xs_prime*cat_xrs_xs)*(cg_l_a * cg_r_a)
    #
    #     return output,first_flag,last_flag