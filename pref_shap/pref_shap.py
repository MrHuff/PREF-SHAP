import numpy as np
from itertools import combinations
import random
import torch

from GPGP.kernel import *
from tqdm import tqdm
from scipy.special import binom
from pref_shap.tensor_CG import tensor_CG

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

def sample_Z(D,max_S):
    max_range = 2**D
    if max_S>=max_range:
        configs= np.array([i for i in range(max_range)])
    else:
        configs= np.array(random.sample([i for i in range(max_range)],max_S))
    configs=np.sort(configs)
    return base_10_base_2(configs,D)


class pref_shap():
    def __init__(self,alpha,k,X_l,X_r,X,max_S: int=5000,rff_mode=False,eps=1e-3,cg_max_its=10,lamb=1e-3,max_inv_row=0,cg_bs=20,device='cuda:0'):
        if max_inv_row >0:
            X = X[torch.randperm(X.shape[0])[:max_inv_row],:]

        self.alpha = alpha.t()
        self.cg_bs=cg_bs
        self.X_l,self.X_r,self.X= X_l.to(device),X_r.to(device),X.to(device)
        self.N_x,self.m = self.X.shape
        self.reg = self.N_x*lamb
        self.device = device
        self.eye = torch.eye(self.N_x).to(device)*self.reg
        self.rff=rff_mode
        if self.rff:
            ls = k.ls
        else:
            self.k=k
        self.precond = torch.inverse(self.k(self.X)+ self.eye)
        self.tensor_CG = tensor_CG(precond=self.precond,reg=self.reg,eps=eps,maxits=cg_max_its,device=device)
        self.Z = torch.from_numpy(sample_Z(self.m,max_S)).float().to(device)

        const = torch.lgamma(torch.tensor(self.m) + 1)
        abs_S = self.Z.sum(1)
        a = torch.exp(const - torch.lgamma((self.m - abs_S) + 1) - torch.lgamma(abs_S + 1))
        self.weights = (self.m - 1) / (a * (abs_S) * (self.m - abs_S))
        self.weights = torch.nan_to_num(self.weights.unsqueeze(-1),posinf=1e6).to(device)
        self.weighted_moore_penrose()
        self.batched_Z = torch.chunk(self.Z,self.Z.shape[0]//cg_bs,dim=0)


    def weighted_moore_penrose(self):
        ztw = self.Z * self.weights
        self.zwz=torch.inverse(ztw.t()@self.Z)


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
        for i in range(S_list_batch.shape[0]):
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

    def value_observation(self,S_batch,x,x_prime):
        output,first_flag,last_flag= self.kernel_tensor_batch(S_batch, x, x_prime)
        output = (self.alpha@output).squeeze()

        if first_flag:
            if len(output.shape)==1:
                o=torch.ones_like(output)
                output = torch.stack([ o,output],dim=0)

            else:
                o=torch.ones_like(output[0,:]).unsqueeze(0)
                output = torch.cat([ o,output],dim=0)
        if last_flag:
            if len(output.shape)==1:
                o=torch.ones_like(output)
                output = torch.stack([output, o], dim=0)
            else:
                o=torch.ones_like(output[0,:]).unsqueeze(0)
                output = torch.cat([output, o], dim=0)

        # if len(output.shape) == 1:
        #     output = output.unsqueeze(0)
        return output

    def fit(self,x,x_prime ):
        """[Running the RKHS SHAP Algorithm to explain kernel ridge regression]
        Args:
            X_new (np.array): [New X data]
            method (str, optional): [Interventional Shapley values (I) or Observational Shapley Values (O)]. Defaults to "O".
            sample_method (str, optional): [What sampling methods to use for the permutations
                                                if "MC" then you do sampling.
                                                if None, you look at all potential 2**M permutation ]. Defaults to "MC".
            num_samples (int, optional): [number of samples to use]. Defaults to None.
            verbose (str, optional): [description]. Defaults to False.
        """
        x=x.to(self.device)
        x_prime=x_prime.to(self.device)
        # Set up containers
        Y_target = []
        for batch in tqdm(self.batched_Z):
            Y_target.append(self.value_observation(batch,x,x_prime))
        Y_cat = torch.cat(Y_target, dim=0)
        if len(Y_cat.shape) == 1:
            Y_target = self.weights.squeeze() * Y_cat
        else:
            Y_target = self.weights* Y_cat
        return self.zwz@(self.Z.t()@Y_target)



# if __name__ == '__main__':
#     test=sample_Z(5,20000)



        # clf = Ridge(1e-5)
        # clf.fit(self.Z, Y_target, sample_weight=weights)

        # self.full_shapley_values_ = np.concatenate([clf.intercept_.reshape(-1, 1), clf.coef_], axis=1)
        # self.SHAP_LM = clf

        # return clf.coef_


