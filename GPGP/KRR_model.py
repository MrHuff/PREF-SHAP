#Implement SUPER FALKON with sparse kernel implementation!
import copy

from sklearn import datasets
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import falkon
from falkon import FalkonOptions
from falkon.kernels import Kernel, DiffKernel, KeopsKernelMixin
from sklearn.metrics import roc_auc_score
from falkon.hopt.objectives import NystromCompReg
import tqdm

class NystromCompRegSHAP(NystromCompReg):

    def get_alpha(self):
        if self.x_train is None or self.y_train is None:
            raise RuntimeError("Call forward at least once before calling predict.")
        with torch.autograd.no_grad():
            L, A, AAT, LB, c = self._calc_intermediate(self.x_train, self.y_train)
            tmp1 = torch.triangular_solve(c, LB, upper=False, transpose=True).solution
            tmp2 = torch.triangular_solve(tmp1, L, upper=False, transpose=True).solution
            return tmp2

def accuracy(true,pred):
    true=true.cpu().numpy()
    pred=pred.cpu().numpy()
    true_zero_one = np.clip(true,0,1)
    pred_zero_one = pred>0
    acc = (true_zero_one==pred_zero_one).sum()/true_zero_one.shape[0]
    return acc

def auc(true,pred):
    true=true.cpu().numpy()
    pred=pred.cpu().numpy()
    true_zero_one = np.clip(true,0,1)
    pred_zero_one = pred>0
    auc = roc_auc_score(true_zero_one,pred_zero_one)
    return auc

def rmse(true, pred):
    return torch.sqrt(torch.mean((true.reshape(-1, 1) - pred.reshape(-1, 1))**2))

def calc_2(true, pred):
    res = (true-pred)**2
    return  (1- res.mean()/true.mean()).item()

def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)

def train_krr(kernel,Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,pen,m_fac=1.0):
    penalty_init = torch.tensor(pen, dtype=torch.float32)
    M = int(round(Xtrain.shape[0] ** 0.5 * m_fac))
    centers_init = Xtrain[np.random.choice(Xtrain.shape[0], size=(M,), replace=False)].clone()
    model = NystromCompRegSHAP(
        kernel=kernel, penalty_init=penalty_init, centers_init=centers_init,  # The initial hp values
        opt_penalty=True, opt_centers=True,  # Whether the various hps are to be optimized
    )
    Xtrain, Ytrain = Xtrain.to('cuda:0'), Ytrain.unsqueeze(-1).to('cuda:0')
    Xval, Yval = Xval.to('cuda:0'), Yval.unsqueeze(-1).to('cuda:0')
    Xtest, Ytest = Xtest.to('cuda:0'), Ytest.unsqueeze(-1).to('cuda:0')
    model = model.to('cuda:0')
    opt_hp = torch.optim.Adam(model.parameters(), lr=1e-2)
    patience = 50
    best_val = -np.inf
    best_test = -np.inf
    best_tr = -np.inf
    best_model = copy.deepcopy(model)
    counter = 0
    pbar = tqdm.tqdm(list(range(500)))
    for i, j in enumerate(pbar):
        opt_hp.zero_grad()
        loss = model(Xtrain, Ytrain)
        loss.backward()
        opt_hp.step()
        try:
            tr_err = auc(Ytrain, model.predict(Xtrain))
            val_err = auc(Yval, model.predict(Xval))
            ts_err = auc(Ytest, model.predict(Xtest))
        except Exception:
            print('auc not working')
            tr_err = accuracy(Ytrain, model.predict(Xtrain))
            val_err = accuracy(Yval, model.predict(Xval))
            ts_err = accuracy(Ytest, model.predict(Xtest))
        if val_err > best_val:
            best_model = copy.deepcopy(model)
            best_val = val_err
            best_test = ts_err
            best_tr = tr_err
            counter = 0
        pbar.set_description(f"Tr acc: {tr_err} Val acc: {val_err} Test acc: {ts_err}")
        counter += 1
        if counter > patience:
            break
    return best_model, best_tr,best_val, best_test

def SGD_KRR(Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,ls,pen=1e-5,m_fac=1.0):
    lengthscale_init = torch.tensor([ls]*(Xtrain.shape[1]//2)).requires_grad_()
    kernel = diffrentiable_FALKON_GPGP(lengthscale=lengthscale_init, options=falkon.FalkonOptions())
    return train_krr(kernel,Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,pen,m_fac)

def SGD_KRR_PGP(Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,ls,pen=1e-5,m_fac=1.0):
    lengthscale_init = torch.tensor([ls]*(Xtrain.shape[1]//2)).requires_grad_()
    kernel = diffrentiable_FALKON_PGP(lengthscale=lengthscale_init, options=falkon.FalkonOptions())
    return train_krr(kernel,Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,pen,m_fac)

def SGD_UKRR(Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,dat,pen=1e-5,m_fac=1.0):
    ls_u, ls_i, user_dim = dat
    lengthscale_i = torch.tensor([ls_i]*((Xtrain.shape[1]-user_dim)//2)).requires_grad_()
    lengthscale_u = torch.tensor([ls_u]*user_dim).requires_grad_()
    kernel = diffrentiable_FALKON_UGPGP(lengthscale_items=lengthscale_i,lengthscale_users=lengthscale_u,user_dim=user_dim, options=falkon.FalkonOptions())
    return train_krr(kernel,Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,pen,m_fac)


def SGD_UKRR_PGP(Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,dat,pen=1e-5,m_fac=1.0):
    ls_u, ls_i, user_dim = dat
    lengthscale_i = torch.tensor([ls_i]*((Xtrain.shape[1]-user_dim)//2)).requires_grad_()
    lengthscale_u = torch.tensor([ls_u]*user_dim).requires_grad_()
    kernel = diffrentiable_FALKON_UPGP(lengthscale_items=lengthscale_i,lengthscale_users=lengthscale_u,user_dim=user_dim, options=falkon.FalkonOptions())
    return train_krr(kernel,Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,pen,m_fac)

class diffrentiable_FALKON_GPGP(DiffKernel):
    def __init__(self, lengthscale, options):
        # Super-class constructor call. We do not specify core_fn
        # but we must specify the hyperparameter of this kernel (lengthscale)
        super().__init__("diffrentiable_FALKON_GPGP", options,core_fn=None,
                         lengthscale=lengthscale)

    def compute(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor, diag: bool):
        ls = self.lengthscale
        xa,xb = torch.chunk(X1,dim=1,chunks=2)
        xc,xd = torch.chunk(X2,dim=1,chunks=2)
        xa_ = xa.div(ls)
        xb_ = xb.div(ls)
        xc_ = xc.div(ls)
        xd_ = xd.div(ls)


        if diag:
            # xa_ = xa_
            # xb_ = xb_
            # xc_ = xc_
            # xd_ = xd_
            K = (-((xa_ - xc_) ** 2 + (xb_ - xd_) ** 2) / 2).sum(-1).exp() - (
                    -((xa_ - xd_) ** 2 + (xb_ - xc_) ** 2) / 2).sum(-1).exp()
            out.copy_(K)
        else:
            K = (-(pairwise_distances(xa_,xc_)+ pairwise_distances(xb_,xd_)) / 2).exp() - (
                    -(pairwise_distances(xa_,xd_) + pairwise_distances(xb_,xc_)) / 2).exp()
            out.copy_(K)
        return out

    def compute_diff(self, X1: torch.Tensor, X2: torch.Tensor, diag: bool):
        # The implementation here is similar to `compute` without in-place operations.
        ls = self.lengthscale.to(device=X1.device, dtype=X1.dtype)
        xa,xb = torch.chunk(X1,dim=1,chunks=2)
        xc,xd = torch.chunk(X2,dim=1,chunks=2)
        xa_ = xa.div(ls)
        xb_ = xb.div(ls)
        xc_ = xc.div(ls)
        xd_ = xd.div(ls)
        if diag:
            # xa_ = xa_.unsqueeze(1)
            # xb_ = xb_.unsqueeze(1)
            # xc_ = xc_.unsqueeze(1)
            # xd_ = xd_.unsqueeze(1)
            K = (-((xa_ - xc_) ** 2 + (xb_ - xd_) ** 2) / 2).sum(-1).exp() - (
                    -((xa_ - xd_) ** 2 + (xb_ - xc_) ** 2) / 2).sum(-1).exp()
            return K
        K = (-(pairwise_distances(xa_, xc_) + pairwise_distances(xb_, xd_)) / 2).exp() - (
                -(pairwise_distances(xa_, xd_) + pairwise_distances(xb_, xc_)) / 2).exp()
        return K

    def detach(self):
        # Clones the class with detached hyperparameters
        return diffrentiable_FALKON_GPGP(
            lengthscale=self.lengthscale.detach(),
            options=self.params
        )

    def compute_sparse(self, X1, X2, out, diag, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Sparse not implemented")

class diffrentiable_FALKON_UGPGP(DiffKernel):
    def __init__(self, lengthscale_items,lengthscale_users,user_dim, options):
        # Super-class constructor call. We do not specify core_fn
        # but we must specify the hyperparameter of this kernel (lengthscale)
        super().__init__("diffrentiable_FALKON_UGPGP", options,core_fn=None,
                         lengthscale_items=lengthscale_items,
                         lengthscale_users=lengthscale_users
                         )
        # self.lengthscale = lengthscale_items
        # self.lengthscale_users = lengthscale_users
        self.user_dim = user_dim

    def compute(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor, diag: bool):

        ls = self.lengthscale_items#.to(device=X1.device, dtype=X1.dtype)
        ls_u = self.lengthscale_users#".to(device=X1.device, dtype=X1.dtype)
        u_1 = X1[:,:self.user_dim]
        u_2 = X2[:,:self.user_dim]

        xa,xb = torch.chunk(X1[:,self.user_dim:],dim=1,chunks=2)
        xc,xd = torch.chunk(X2[:,self.user_dim:],dim=1,chunks=2)

        xa_ = xa.div(ls)
        xb_ = xb.div(ls)
        xc_ = xc.div(ls)
        xd_ = xd.div(ls)

        u_1_ = u_1.div(ls_u)
        u_2_ = u_2.div(ls_u)

        if diag:
            K = (-((xa_ - xc_) ** 2 + (xb_ - xd_) ** 2) / 2).sum(-1).exp() - (
                    -((xa_ - xd_) ** 2 + (xb_ - xc_) ** 2) / 2).sum(-1).exp()
            L = K * (-(u_1_ - u_2_) ** 2 ).sum(-1).exp()
            out.copy_(L)
        else:
            K = (-(pairwise_distances(xa_,xc_)+ pairwise_distances(xb_,xd_)) / 2).exp() - (
                    -(pairwise_distances(xa_,xd_) + pairwise_distances(xb_,xc_)) / 2).exp()
            L = K * (-pairwise_distances(u_1_,u_2_) ).exp()
            out.copy_(L)
        return out

    def compute_diff(self, X1: torch.Tensor, X2: torch.Tensor, diag: bool):
        # The implementation here is similar to `compute` without in-place operations.
        ls = self.lengthscale_items#.to(device=X1.device, dtype=X1.dtype)
        ls_u = self.lengthscale_users#".to(device=X1.device, dtype=X1.dtype)
        u_1 = X1[:,:self.user_dim]
        u_2 = X2[:,:self.user_dim]

        xa,xb = torch.chunk(X1[:,self.user_dim:],dim=1,chunks=2)
        xc,xd = torch.chunk(X2[:,self.user_dim:],dim=1,chunks=2)

        xa_ = xa.div(ls)
        xb_ = xb.div(ls)
        xc_ = xc.div(ls)
        xd_ = xd.div(ls)

        u_1_ = u_1.div(ls_u)
        u_2_ = u_2.div(ls_u)


        if diag:

            K = (-((xa_ - xc_) ** 2 + (xb_ - xd_) ** 2) / 2).sum(-1).exp() - (
                    -((xa_ - xd_) ** 2 + (xb_ - xc_) ** 2) / 2).sum(-1).exp()
            L = K * (-(u_1_ - u_2_) ** 2).sum(-1).exp()
            return L
        else:
            K = (-(pairwise_distances(xa_, xc_) + pairwise_distances(xb_, xd_)) / 2).exp() - (
                    -(pairwise_distances(xa_, xd_) + pairwise_distances(xb_, xc_)) / 2).exp()
            L = K * (-pairwise_distances(u_1_, u_2_)).exp()
        return L

    def detach(self):
        # Clones the class with detached hyperparameters
        return diffrentiable_FALKON_GPGP(
            lengthscale=self.lengthscale.detach(),
            options=self.params
        )

    def compute_sparse(self, X1, X2, out, diag, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Sparse not implemented")


class diffrentiable_FALKON_UPGP(DiffKernel):
    def __init__(self, lengthscale_items,lengthscale_users,user_dim, options):
        # Super-class constructor call. We do not specify core_fn
        # but we must specify the hyperparameter of this kernel (lengthscale)
        super().__init__("diffrentiable_FALKON_UPGP", options,core_fn=None,
                         lengthscale_items=lengthscale_items,
                         lengthscale_users=lengthscale_users
                         )
        # self.lengthscale = lengthscale_items
        # self.lengthscale_users = lengthscale_users
        self.user_dim = user_dim

    def compute(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor, diag: bool):

        ls = self.lengthscale_items#.to(device=X1.device, dtype=X1.dtype)
        ls_u = self.lengthscale_users#".to(device=X1.device, dtype=X1.dtype)
        u_1 = X1[:,:self.user_dim]
        u_2 = X2[:,:self.user_dim]

        xa,xb = torch.chunk(X1[:,self.user_dim:],dim=1,chunks=2)
        xc,xd = torch.chunk(X2[:,self.user_dim:],dim=1,chunks=2)

        xa_ = xa.div(ls)
        xb_ = xb.div(ls)
        xc_ = xc.div(ls)
        xd_ = xd.div(ls)

        u_1_ = u_1.div(ls_u)
        u_2_ = u_2.div(ls_u)

        if diag:
            K = (-(xa_-xc_)**2/2).sum(-1).exp() + (-(xb_-xd_)**2/2).sum(-1).exp() - (-(xa_-xd_)**2/2).sum(-1).exp() - (-(xb_-xc_)**2/2).sum(-1).exp()
            L = K * (-(u_1_ - u_2_) ** 2 ).sum(-1).exp()
            out.copy_(L)
        else:
            K = (-pairwise_distances(xa_,xc_)/2).exp()+(-pairwise_distances(xb_,xd_)/2).exp()-(-pairwise_distances(xa_,xd_)/2).exp()-(-pairwise_distances(xb_,xc_)/2).exp()
            L = K * (-pairwise_distances(u_1_,u_2_) ).exp()
            out.copy_(L)
        return out

    def compute_diff(self, X1: torch.Tensor, X2: torch.Tensor, diag: bool):
        # The implementation here is similar to `compute` without in-place operations.
        ls = self.lengthscale_items#.to(device=X1.device, dtype=X1.dtype)
        ls_u = self.lengthscale_users#".to(device=X1.device, dtype=X1.dtype)
        u_1 = X1[:,:self.user_dim]
        u_2 = X2[:,:self.user_dim]

        xa,xb = torch.chunk(X1[:,self.user_dim:],dim=1,chunks=2)
        xc,xd = torch.chunk(X2[:,self.user_dim:],dim=1,chunks=2)

        xa_ = xa.div(ls)
        xb_ = xb.div(ls)
        xc_ = xc.div(ls)
        xd_ = xd.div(ls)

        u_1_ = u_1.div(ls_u)
        u_2_ = u_2.div(ls_u)


        if diag:
            K = (-(xa_-xc_)**2/2).sum(-1).exp() + (-(xb_-xd_)**2/2).sum(-1).exp() - (-(xa_-xd_)**2/2).sum(-1).exp() - (-(xb_-xc_)**2/2).sum(-1).exp()
            L = K * (-(u_1_ - u_2_) ** 2).sum(-1).exp()
            return L
        else:
            K = (-pairwise_distances(xa_,xc_)/2).exp()+(-pairwise_distances(xb_,xd_)/2).exp()-(-pairwise_distances(xa_,xd_)/2).exp()-(-pairwise_distances(xb_,xc_)/2).exp()
            L = K * (-pairwise_distances(u_1_, u_2_)).exp()
        return L

    def detach(self):
        # Clones the class with detached hyperparameters
        return diffrentiable_FALKON_GPGP(
            lengthscale=self.lengthscale.detach(),
            options=self.params
        )

    def compute_sparse(self, X1, X2, out, diag, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Sparse not implemented")


class diffrentiable_FALKON_PGP(DiffKernel):
    def __init__(self, lengthscale, options):
        # Super-class constructor call. We do not specify core_fn
        # but we must specify the hyperparameter of this kernel (lengthscale)
        super().__init__("diffrentiable_FALKON_PGP", options,core_fn=None,
                         lengthscale=lengthscale)
        # super().__init__("diffrentiable_FALKON_GPGP", options)
        # self.lengthscale = lengthscale

    def compute(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor, diag: bool):
        ls = self.lengthscale
        xa,xb = torch.chunk(X1,dim=1,chunks=2)
        xc,xd = torch.chunk(X2,dim=1,chunks=2)
        xa_ = xa.div(ls)
        xb_ = xb.div(ls)
        xc_ = xc.div(ls)
        xd_ = xd.div(ls)

        if diag:
            K = (-(xa_-xc_)**2/2).sum(-1).exp() + (-(xb_-xd_)**2/2).sum(-1).exp() - (-(xa_-xd_)**2/2).sum(-1).exp() - (-(xb_-xc_)**2/2).sum(-1).exp()

            out.copy_(K)
        else:
            K = (-pairwise_distances(xa_,xc_)/2).exp()+(-pairwise_distances(xb_,xd_)/2).exp()-(-pairwise_distances(xa_,xd_)/2).exp()-(-pairwise_distances(xb_,xc_)/2).exp()

            out.copy_(K)
        return out

    def compute_diff(self, X1: torch.Tensor, X2: torch.Tensor, diag: bool):
        ls = self.lengthscale
        xa,xb = torch.chunk(X1,dim=1,chunks=2)
        xc,xd = torch.chunk(X2,dim=1,chunks=2)
        xa_ = xa.div(ls)
        xb_ = xb.div(ls)
        xc_ = xc.div(ls)
        xd_ = xd.div(ls)
        if diag:
            K = (-(xa_-xc_)**2/2).sum(-1).exp() + (-(xb_-xd_)**2/2).sum(-1).exp() - (-(xa_-xd_)**2/2).sum(-1).exp() - (-(xb_-xc_)**2/2).sum(-1).exp()
            return K
        K = (-pairwise_distances(xa_, xc_) / 2).exp() + (-pairwise_distances(xb_, xd_) / 2).exp() - (
                    -pairwise_distances(xa_, xd_) / 2).exp() - (-pairwise_distances(xb_, xc_) / 2).exp()

        return K

    def detach(self):
        # Clones the class with detached hyperparameters
        return diffrentiable_FALKON_PGP(
            lengthscale=self.lengthscale.detach(),
            options=self.params
        )

    def compute_sparse(self, X1, X2, out, diag, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Sparse not implemented")

