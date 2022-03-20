#Implement SUPER FALKON with sparse kernel implementation!
from sklearn import datasets
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import falkon
from falkon import FalkonOptions
from falkon.kernels import Kernel, DiffKernel, KeopsKernelMixin
from sklearn.metrics import roc_auc_score
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

def train_KRR(Xtrain,Ytrain,Xval,Yval,Xtest,Ytest,ls,pen=1e-5):
    lengthscale_init = torch.tensor([ls]*(Xtrain.shape[1]//2)).requires_grad_(False)
    kernel = diffrentiable_FALKON_GPGP(lengthscale_init, options=falkon.FalkonOptions())
    M = int(round(Xtrain.shape[0] ** 0.5))
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


class diffrentiable_FALKON_GPGP(Kernel):
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
            xa_ = xa.unsqueeze(1)
            xb_ = xb.unsqueeze(1)
            xc_ = xc.unsqueeze(1)
            xd_ = xd.unsqueeze(1)
            K = (-((xa_ - xc_) ** 2 + (xb_ - xd_) ** 2) / 2).exp() - (
                    -((xa_ - xd_) ** 2 + (xb_ - xc_) ** 2) / 2).exp()
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
            xa_ = xa.unsqueeze(1)
            xb_ = xb.unsqueeze(1)
            xc_ = xc.unsqueeze(1)
            xd_ = xd.unsqueeze(1)
            K = (-((xa_ - xc_) ** 2 + (xb_ - xd_) ** 2) / 2).exp() - (
                    -((xa_ - xd_) ** 2 + (xb_ - xc_) ** 2) / 2).exp()
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



class diffrentiable_FALKON_PGP():
    pass


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
