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
        return base_2_base_10(max_range, D)


class pref_shap():
    def __init__(self,alpha,k,X_l,X_r,X,k_U=None,u=None,X_U=None,max_S: int=5000,rff_mode=False,eps=1e-3,cg_max_its=10,lamb=1e-3,max_inv_row=0,cg_bs=20,post_method='OLS',interventional=False,device='cuda:0'):
        if max_inv_row >0:
            X = X[torch.randperm(X.shape[0])[:max_inv_row],:]
        self.post_method=post_method
        self.alpha = alpha.t()
        self.cg_bs=cg_bs
        self.X_l,self.X_r,self.X= X_l.to(device),X_r.to(device),X.to(device)
        self.max_S=max_S
        self.eps =eps
        self.cg_max_its=cg_max_its
        if X_U is not None:
            self.u = u.to(device)
            self.X_U = X_U.to(device)
            self.m_u =self.u.shape[1]
        else:
            self.X_U=X_U
            self.u=u
        self.k_U = k_U
        self.N_x,self.m = self.X.shape
        self.reg = self.N_x*lamb
        self.device = device
        self.eye = torch.eye(self.N_x).to(device)*self.reg
        self.rff=rff_mode
        self.interventional = interventional
        if self.rff:
            ls = k.ls
        else:
            self.k=k

    def setup_user_vals(self):
        self.precond = torch.inverse(self.k_U(self.X_U)+ self.eye)
        self.tensor_CG = tensor_CG(precond=self.precond,reg=self.reg,eps=self.eps,maxits=self.cg_max_its,device=self.device)
        self.Z = torch.from_numpy(sample_Z(self.m_u,self.max_S)).float().to(self.device)
        const = torch.lgamma(torch.tensor(self.m_u) + 1)
        abs_S = self.Z.sum(1)
        a = torch.exp(const - torch.lgamma((self.m_u - abs_S) + 1) - torch.lgamma(abs_S + 1))
        self.weights = (self.m_u - 1) / (a * (abs_S) * (self.m_u - abs_S))
        self.weights = torch.nan_to_num(self.weights.unsqueeze(-1),posinf=1e6).to(self.device)
        self.weighted_moore_penrose()
        self.batched_Z = torch.chunk(self.Z,self.Z.shape[0]//self.cg_bs,dim=0)

    def setup_item_vals(self):
        self.precond = torch.inverse(self.k(self.X)+ self.eye)
        self.tensor_CG = tensor_CG(precond=self.precond,reg=self.reg,eps=self.eps,maxits=self.cg_max_its,device=self.device)
        self.Z = torch.from_numpy(sample_Z(self.m,self.max_S)).float().to(self.device)
        const = torch.lgamma(torch.tensor(self.m) + 1)
        abs_S = self.Z.sum(1)
        a = torch.exp(const - torch.lgamma((self.m - abs_S) + 1) - torch.lgamma(abs_S + 1))
        self.weights = (self.m - 1) / (a * (abs_S) * (self.m - abs_S))
        self.weights = torch.nan_to_num(self.weights.unsqueeze(-1),posinf=1e6).to(self.device)
        self.weighted_moore_penrose()
        self.batched_Z = torch.chunk(self.Z,self.Z.shape[0]//self.cg_bs,dim=0)

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


    def kernel_tensor_batch_user_case_1(self,u,S_batch,x,x_prime):
        inv_tens=[]
        vec = []
        vec_c=[]
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
                u_S,u_Sc= u[:,S],u[:,~S]
                vec_cat  = self.k_U(self.X_U[:,S],u_S,S)
                vec_c_cat  = self.k_U(self.X_U[:,~S],u_Sc,S_C)
                RH_cat = self.k(self.X_l,x)*self.k(self.X_r,x_prime)-self.k(self.X_l,x_prime)*self.k(self.X_r,x)
                RH.append(RH_cat)
                vec.append(vec_cat)
                vec_c.append(vec_c_cat)

        inv_tens=torch.stack(inv_tens,dim=0)
        vec = torch.stack(vec,dim=0)
        vec_c = torch.stack(vec_c,dim=0)
        RH = torch.stack(RH,dim=0)

        cg_output = self.tensor_CG.solve(inv_tens,vec) #BS x N x D
        #Write assertion error

        if torch.isnan(cg_output).any():
            cg_output=torch.nan_to_num(cg_output, nan=0.0)
        LH = torch.bmm(vec_c,cg_output)
        output = LH*RH
        return output,first_flag,last_flag

    def kernel_tensor_batch_user_case_2(self,u, S_list_batch,x,x_prime):
        output,first_flag,last_flag=self.kernel_tensor_batch(S_list_batch,x,x_prime)
        user_k = self.k_U(self.u,u)
        output = user_k*output
        return output,first_flag,last_flag

    def kernel_tensor_batch_interventional(self,S_list_batch,x,x_prime):
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
                x_S,x_prime_S= x[:,S],x_prime[:,S]
                xs_cat = torch.cat([x_S,x_prime_S],dim=0)
                stacked_lr = torch.cat([self.X_l[:,S],self.X_r[:,S]],dim=0)
                l_hadamard_mat = self.k(stacked_lr,xs_cat,S)
                r,c = l_hadamard_mat.shape
                xls_xs,xls_xs_prime,xrs_xs,xrs_xs_prime, = l_hadamard_mat[:r//2,:][:,:c//2],l_hadamard_mat[:r//2,:][:,c//2:],l_hadamard_mat[r//2:,:][:,:c//2],l_hadamard_mat[r//2:,:][:,c//2:]
                cat_xls_xs.append(xls_xs)
                cat_xls_xs_prime.append(xls_xs_prime)
                cat_xrs_xs.append(xrs_xs)
                cat_xrs_xs_prime.append(xrs_xs_prime)
                klsc_xsc = self.k( self.X_l[:,S_C],self.X[:,S_C],S_C).mean(1,keepdim=True).expand(-1,c//2)
                krsc_xsc = self.k( self.X_r[:,S_C],self.X[:,S_C],S_C).mean(1,keepdim=True).expand(-1,c//2)
                cat_klsc_xsc.append(klsc_xsc)
                cat_krsc_xsc.append(krsc_xsc)

        cat_xls_xs = torch.stack(cat_xls_xs,dim=0)
        cat_xls_xs_prime = torch.stack(cat_xls_xs_prime,dim=0)
        cat_xrs_xs = torch.stack(cat_xrs_xs,dim=0)
        cat_xrs_xs_prime = torch.stack(cat_xrs_xs_prime,dim=0)
        cg_l_a=torch.stack(cat_klsc_xsc,dim=0)
        cg_r_a=torch.stack(cat_krsc_xsc,dim=0)

        output = (cat_xls_xs*cat_xrs_xs_prime )* (cg_l_a * cg_r_a) - (cat_xls_xs_prime*cat_xrs_xs)*(cg_l_a * cg_r_a)

        return output,first_flag,last_flag

    def value_observation(self,S_batch,x,x_prime):
        if self.interventional:
            output,first_flag,last_flag= self.kernel_tensor_batch_interventional(S_batch, x, x_prime)
        else:
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

    def user_observation(self,S_batch,x,x_prime,u,case):

        if case==1:
            output, first_flag, last_flag =self.kernel_tensor_batch_user_case_1(u,S_batch,x,x_prime)
        elif case==2:
            output, first_flag, last_flag =self.kernel_tensor_batch_user_case_2(u,S_batch,x,x_prime)

        # if self.interventional:
        #     output,first_flag,last_flag= self.kernel_tensor_batch_interventional(S_batch, x, x_prime)
        # else:
        #     output,first_flag,last_flag= self.kernel_tensor_batch(S_batch, x, x_prime)
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
        self.setup_item_vals()
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

        self.Y_target=Y_target


    def fit_user(self,u,x,x_prime,case=2):
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
        if case==1:
            self.setup_user_vals()
        if case==2:
            self.setup_item_vals()
        x=x.to(self.device)
        x_prime=x_prime.to(self.device)
        u=u.to(self.device)
        # Set up containers
        Y_target = []
        for batch in tqdm(self.batched_Z):
            Y_target.append(self.user_observation(batch,x,x_prime,u,case))
        Y_cat = torch.cat(Y_target, dim=0)
        if len(Y_cat.shape) == 1:
            Y_target = self.weights.squeeze() * Y_cat
        else:
            Y_target = self.weights* Y_cat
        self.Y_target=Y_target

    def construct_values(self,coeffs,post_method=None):
        if not post_method is None:
            self.post_method=post_method
        shap_dict={}
        # if self.post_method=='OLS':
        shap_dict[0]=self.zwz@(self.Z.t()@self.Y_target)
            # return self.zwz@(self.Z.t()@Y_target)
        for coeff in coeffs:
            if self.post_method == 'lasso':
                clf = Lasso(coeff)
            if self.post_method == 'ridge':
                clf = Ridge(coeff)
            if self.post_method == 'elastic':
                clf = ElasticNet(coeff)
            clf.fit(self.Z.cpu().numpy(), self.Y_target.cpu().numpy(), sample_weight=self.weights.cpu().numpy().squeeze())
            vals = torch.from_numpy(clf.coef_.transpose())
            shap_dict[coeff]=vals
        return shap_dict





# if __name__ == '__main__':
#     test=sample_Z(5,20000)



        # clf = Ridge(1e-5)
        # clf.fit(self.Z, Y_target, sample_weight=weights)

        # self.full_shapley_values_ = np.concatenate([clf.intercept_.reshape(-1, 1), clf.coef_], axis=1)
        # self.SHAP_LM = clf

        # return clf.coef_


