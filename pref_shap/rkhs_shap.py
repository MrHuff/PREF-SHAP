from pref_shap.pref_shap import *

class rkhs_shap():
    def __init__(self, model, alpha, k, X, max_S: int = 5000, rff_mode=False,
                 eps=1e-3, cg_max_its=10, lamb=1e-3, max_inv_row=0, cg_bs=20, post_method='OLS', interventional=False,
                 device='cuda:0'):
        if max_inv_row >0:
            X = X[torch.randperm(X.shape[0])[:max_inv_row],:]
        self.model=model
        self.post_method=post_method
        self.alpha = alpha.t()
        self.cg_bs=cg_bs
        self.max_S=max_S
        self.eps =eps
        self.cg_max_its=cg_max_its
        self.device = device
        self.lamb=lamb
        self.rff=rff_mode
        self.interventional = interventional
        self.k = k
        self.X=X.to(self.device)


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

    def kernel_tensor_batch(self,S_list_batch,x):
        inv_tens=[]
        vec = []
        vec_sc=[]
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
                vec.append(self.k(self.X[:,S],x[:,S],S))
                vec_sc.append(self.k(self.X[:,S_C],None,S_C))
        inv_tens=torch.stack(inv_tens,dim=0)
        vec = torch.stack(vec,dim=0)
        vec_sc = torch.stack(vec_sc,dim=0)
        cg_output = self.tensor_CG.solve(inv_tens,vec) #BS x N x D
        if torch.isnan(cg_output).any():
            cg_output=torch.nan_to_num(cg_output, nan=0.0)
        output = vec*(vec_sc@cg_output)
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

    def value_observation(self,S_batch,x):
        output, first_flag, last_flag = self.kernel_tensor_batch( S_batch, x,)
        output = (self.alpha@output).squeeze() - self.y_pred_mean
        return self.flag_adjustment(first_flag, last_flag, output)

    def fit(self,x):
        x=x.to(self.device)
        self.mask = (x.var(0) > 0).cpu()
        self.eff_dim = self.mask.sum().item()
        self.setup_item_vals()
        self.y_pred = self.model.predict(x)
        self.y_pred_mean = self.y_pred.mean().item()
        Y_target = []
        for batch in tqdm(self.batched_Z):
            Y_target.append(self.value_observation(batch,x))
        Y_cat = torch.cat(Y_target, dim=0)
        return Y_cat,self.weights,self.Z