

import gpytorch
from gpytorch.kernels.keops import RBFKernel

import torch
from gpytorch import settings
from gpytorch.kernels.keops.keops_kernel import KEOLazyTensor
from gpytorch.lazy import KeOpsLazyTensor

class skewedRBF(RBFKernel):

    def forward(self, x1, x2, diag=False, **params):
        # x1_ = x1.div(self.lengthscale)
        # x2_ = x2.div(self.lengthscale)

        covar_func = lambda x1, x2, diag=diag: self.covar_func(x1, x2, diag)

        if diag:
            return covar_func(x1, x2, diag=True)

        return KeOpsLazyTensor(x1, x2, covar_func)

    def covar_func(self, x1, x2, diag=False):
        xa,xb = torch.chunk(x1,dim=1,chunks=2)
        xc,xd = torch.chunk(x2,dim=1,chunks=2)
        ls,_ = torch.chunk(self.lengthscale,dim=1,chunks=2)
        xa_ = xa.div(ls)
        xb_ = xb.div(ls)
        xc_ = xc.div(ls)
        xd_ = xd.div(ls)
        if (
                diag
                or xa_.size(-2) < settings.max_cholesky_size.value()
                or xb_.size(-2) < settings.max_cholesky_size.value()
        ):
            return self._nonkeops_covar_func(x1, x2, diag=diag)

        with torch.autograd.enable_grad():
            xa__ = KEOLazyTensor(xa_[..., :, None, :])
            xb__ = KEOLazyTensor(xb_[..., None, :, :])

            xc__ = KEOLazyTensor(xc_[..., :, None, :])
            xd__ = KEOLazyTensor(xd_[..., None, :, :])

            K = (-((xa__ - xc__) ** 2+(xb__ - xd__) ** 2).sum(-1) / 2).exp() - (-((xa__ - xd__) ** 2+(xb__ - xc__) ** 2).sum(-1) / 2).exp()

            return K




