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

def learn_with_kernel(Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,kernel,pen=1e-5):
    M = int(round(Xtrain.shape[0]**0.5))
    flk_opt = FalkonOptions(use_cpu=False)
    model = falkon.Falkon(
        kernel=kernel, penalty=pen, M=M, options=flk_opt,
        error_every=1, error_fn=rmse)
    model.fit(Xtrain, Ytrain)
    val_err = calc_2(Yval, model.predict(Xval))
    ts_err = calc_2(Ytest, model.predict(Xtest))
    print("Val r2: %.2f" % (val_err))
    print("Test r2: %.2f" % (ts_err))
    return model,val_err,ts_err

def SGD_KRR(Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,ls,pen=1e-5):
    lengthscale_init = torch.tensor([ls]*(Xtrain.shape[1]//2)).requires_grad_()
    kernel = diffrentiable_FALKON_GPGP(lengthscale=lengthscale_init, options=falkon.FalkonOptions())
    penalty_init = torch.tensor(pen, dtype=torch.float32)
    M = int(round(Xtrain.shape[0] ** 0.5))
    centers_init = Xtrain[np.random.choice(Xtrain.shape[0], size=(M,), replace=False)].clone()
    model = NystromCompRegSHAP(
        kernel=kernel, penalty_init=penalty_init, centers_init=centers_init,  # The initial hp values
        opt_penalty=True, opt_centers=True,  # Whether the various hps are to be optimized
    )
    Xtrain, Ytrain =Xtrain.to('cuda:0'), Ytrain.unsqueeze(-1).to('cuda:0')
    Xval, Yval =Xval.to('cuda:0'), Yval.unsqueeze(-1).to('cuda:0')
    Xtest, Ytest =Xtest.to('cuda:0'), Ytest.unsqueeze(-1).to('cuda:0')
    model = model.to('cuda:0')
    opt_hp = torch.optim.Adam(model.parameters(), lr=1e-2)
    patience = 50
    best_val = -np.inf
    best_test =-np.inf
    best_model = copy.deepcopy(model)
    counter=0
    pbar = tqdm.tqdm(list(range(500)))
    for i, j in enumerate(pbar):
        opt_hp.zero_grad()
        loss = model(Xtrain, Ytrain)
        loss.backward()
        opt_hp.step()
        val_err = auc(Yval, model.predict(Xval))
        ts_err = auc(Ytest, model.predict(Xtest))
        if val_err>best_val:
            best_model = copy.deepcopy(model)
            best_val= val_err
            best_test= ts_err
            counter=0
        pbar.set_description(f"Val acc: {val_err} Test acc: {ts_err}")
        counter+=1
        if counter>patience:
            break
    return best_model,best_val,best_test

def SGD_UKRR(Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,dat,pen=1e-5):
    ls_u, ls_i, user_dim = dat
    lengthscale_i = torch.tensor([ls_i]*((Xtrain.shape[1]-user_dim)//2)).requires_grad_()
    lengthscale_u = torch.tensor([ls_u]*user_dim).requires_grad_()
    kernel = diffrentiable_FALKON_UGPGP(lengthscale_items=lengthscale_i,lengthscale_users=lengthscale_u,user_dim=user_dim, options=falkon.FalkonOptions())
    penalty_init = torch.tensor(pen, dtype=torch.float32)
    M = int(round(Xtrain.shape[0] ** 0.5))
    centers_init = Xtrain[np.random.choice(Xtrain.shape[0], size=(M,), replace=False)].clone()
    model = NystromCompRegSHAP(
        kernel=kernel, penalty_init=penalty_init, centers_init=centers_init,  # The initial hp values
        opt_penalty=True, opt_centers=True,  # Whether the various hps are to be optimized
    )
    Xtrain, Ytrain =Xtrain.to('cuda:0'), Ytrain.unsqueeze(-1).to('cuda:0')
    Xval, Yval =Xval.to('cuda:0'), Yval.unsqueeze(-1).to('cuda:0')
    Xtest, Ytest =Xtest.to('cuda:0'), Ytest.unsqueeze(-1).to('cuda:0')
    model = model.to('cuda:0')
    opt_hp = torch.optim.Adam(model.parameters(), lr=1e-2)
    patience = 50
    best_val = -np.inf
    best_test =-np.inf
    best_model = copy.deepcopy(model)
    counter=0
    pbar = tqdm.tqdm(list(range(500)))
    for i, j in enumerate(pbar):
        opt_hp.zero_grad()
        loss = model(Xtrain, Ytrain)
        loss.backward()
        opt_hp.step()
        val_err = auc(Yval, model.predict(Xval))
        ts_err = auc(Ytest, model.predict(Xtest))
        if val_err>best_val:
            best_model = copy.deepcopy(model)
            best_val= val_err
            best_test= ts_err
            counter=0
        pbar.set_description(f"Val acc: {val_err} Test acc: {ts_err}")
        counter+=1
        if counter>patience:
            break
    return best_model,best_val,best_test

def train_vanilla_KRR(Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,ls,pen=1e-5):
    lengthscale_init = torch.tensor([ls]*(Xtrain.shape[1])).requires_grad_(False)
    kernel = falkon.kernels.GaussianKernel(lengthscale_init)
    M = int(round(Xtrain.shape[0] ** 0.5))
    print(M)
    flk_opt = FalkonOptions(use_cpu=False)
    model = falkon.Falkon(
        kernel=kernel, penalty=pen, M=M, options=flk_opt,
        error_every=1, error_fn=rmse)
    model.fit(Xtrain, Ytrain)
    val_err = auc(Yval, model.predict(Xval))
    ts_err = auc(Ytest, model.predict(Xtest))
    print("Val auc: %.2f" % (val_err))
    print("Test auc: %.2f" % (ts_err))
    return model, val_err, ts_err

def train_KRR(Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,ls,pen=1e-5):
    lengthscale_init = torch.tensor([ls]*(Xtrain.shape[1]//2)).requires_grad_(False)
    kernel = diffrentiable_FALKON_GPGP(lengthscale_init, options=falkon.FalkonOptions())
    M = int(round(Xtrain.shape[0] ** 0.5))
    print(M)
    flk_opt = FalkonOptions(use_cpu=False)
    model = falkon.Falkon(
        kernel=kernel, penalty=pen, M=M, options=flk_opt,
        error_every=1, error_fn=rmse)
    model.fit(Xtrain, Ytrain)
    val_err = auc(Yval, model.predict(Xval))
    ts_err = auc(Ytest, model.predict(Xtest))
    print("Val auc: %.2f" % (val_err))
    print("Test auc: %.2f" % (ts_err))
    return model, val_err, ts_err


def train_KRR_user(Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,dat,pen=1e-5):
    ls_u, ls_i, user_dim=dat
    lengthscale_i = torch.tensor([ls_i]*((Xtrain.shape[1])//2-user_dim)).requires_grad_(False)
    lengthscale_u = torch.tensor([ls_u]*user_dim).requires_grad_(False)

    kernel = diffrentiable_FALKON_UGPGP(lengthscale_items=lengthscale_i,lengthscale_users=lengthscale_u,user_dim=user_dim, options=falkon.FalkonOptions())
    M = int(round(Xtrain.shape[0] ** 0.5))
    print(M)
    flk_opt = FalkonOptions(use_cpu=False)
    model = falkon.Falkon(
        kernel=kernel, penalty=pen, M=M, options=flk_opt,
        error_every=1, error_fn=rmse)
    model.fit(Xtrain, Ytrain)
    val_err = auc(Yval, model.predict(Xval))
    ts_err = auc(Ytest, model.predict(Xtest))
    print("Val auc: %.2f" % (val_err))
    print("Test auc: %.2f" % (ts_err))
    return model, val_err, ts_err

def train_KRR_PGP(Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,ls,pen=1e-5):
    lengthscale_init = torch.tensor([ls]*(Xtrain.shape[1]//2)).requires_grad_(False)
    kernel = diffrentiable_FALKON_PGP(lengthscale_init, options=falkon.FalkonOptions())
    M = int(round(Xtrain.shape[0] ** 0.5))
    print(M)
    flk_opt = FalkonOptions(use_cpu=False)
    model = falkon.Falkon(
        kernel=kernel, penalty=pen, M=M, options=flk_opt,
        error_every=1, error_fn=rmse)
    model.fit(Xtrain, Ytrain)
    val_err = auc(Yval, model.predict(Xval))
    ts_err = auc(Ytest, model.predict(Xtest))
    print("Val auc: %.2f" % (val_err))
    print("Test auc: %.2f" % (ts_err))
    return model, val_err, ts_err

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

class diffrentiable_FALKON_PGP(Kernel):
    def __init__(self, lengthscale, options):
        # Super-class constructor call. We do not specify core_fn
        # but we must specify the hyperparameter of this kernel (lengthscale)
        super().__init__("diffrentiable_FALKON_GPGP", options)
        self.lengthscale = lengthscale

    def compute(self, X1: torch.Tensor, X2: torch.Tensor, out: torch.Tensor, diag: bool):
        ls = self.lengthscale.to(device=X1.device, dtype=X1.dtype)
        xa,xb = torch.chunk(X1,dim=1,chunks=2)
        xc,xd = torch.chunk(X2,dim=1,chunks=2)
        xa_ = xa.div(ls)
        xb_ = xb.div(ls)
        xc_ = xc.div(ls)
        xd_ = xd.div(ls)
        if diag:
            xa_ = xa_.unsqueeze(1)
            xb_ = xb_.unsqueeze(1)
            xc_ = xc_.unsqueeze(1)
            xd_ = xd_.unsqueeze(1)
            K = (-pairwise_distances(xa_,xc_)/2).exp()+(-pairwise_distances(xb_,xd_)/2).exp()-(-pairwise_distances(xa_,xd_)/2).exp()-(-pairwise_distances(xb_,xc_)/2).exp()

            out.copy_(K)
        else:
            K = (-pairwise_distances(xa_,xc_)/2).exp()+(-pairwise_distances(xb_,xd_)/2).exp()-(-pairwise_distances(xa_,xd_)/2).exp()-(-pairwise_distances(xb_,xc_)/2).exp()

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
            xa_ = xa_.unsqueeze(1)
            xb_ = xb_.unsqueeze(1)
            xc_ = xc_.unsqueeze(1)
            xd_ = xd_.unsqueeze(1)
            K = (-pairwise_distances(xa_,xc_)/2).exp()+(-pairwise_distances(xb_,xd_)/2).exp()-(-pairwise_distances(xa_,xd_)/2).exp()-(-pairwise_distances(xb_,xc_)/2).exp()

            return K
        K = (-pairwise_distances(xa_, xc_) / 2).exp() + (-pairwise_distances(xb_, xd_) / 2).exp() - (
                    -pairwise_distances(xa_, xd_) / 2).exp() - (-pairwise_distances(xb_, xc_) / 2).exp()

        return K

    def detach(self):
        # Clones the class with detached hyperparameters
        return diffrentiable_FALKON_GPGP(
            lengthscale=self.lengthscale.detach(),
            options=self.params
        )

    def compute_sparse(self, X1, X2, out, diag, **kwargs) -> torch.Tensor:
        raise NotImplementedError("Sparse not implemented")

if __name__ == '__main__':
    X, Y = datasets.fetch_california_housing(return_X_y=True)
    num_train = int(X.shape[0] * 0.8)
    num_test = X.shape[0] - num_train
    shuffle_idx = np.arange(X.shape[0])
    np.random.shuffle(shuffle_idx)
    train_idx = shuffle_idx[:num_train]
    test_idx = shuffle_idx[num_train:]
    Xtrain, Ytrain = X[train_idx], Y[train_idx]
    Xtest, Ytest = X[test_idx], Y[test_idx]

    Xtrain = torch.from_numpy(Xtrain).to(dtype=torch.float32)
    Xtest = torch.from_numpy(Xtest).to(dtype=torch.float32)
    Ytrain = torch.from_numpy(Ytrain).to(dtype=torch.float32)
    Ytest = torch.from_numpy(Ytest).to(dtype=torch.float32)

    train_mean = Xtrain.mean(0, keepdim=True)
    train_std = Xtrain.std(0, keepdim=True)
    Xtrain -= train_mean
    Xtrain /= train_std
    Xtest -= train_mean
    Xtest /= train_std

    lengthscale_init = torch.tensor([2.0]*(Xtrain.shape[1]//2)).requires_grad_(False)
    k = diffrentiable_FALKON_GPGP(lengthscale_init, options=falkon.FalkonOptions())
    model, val_err, ts_err = learn_with_kernel(Xtrain,Ytrain,Xtest,Ytest,Xtest,Ytest,k)
