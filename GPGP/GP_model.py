import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class ExactGPGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPGP, self).__init__(train_x, train_y, likelihood)
        # self.register_buffer('X_train',train_x)
        # self.register_buffer('Y_train',train_y)
        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(ard_num_dims=train_x.shape[1]//2))

    def skew_symmetric_forward(self,a,b,c,d):
        return self.kernel(a,c)*self.kernel(b,d)-self.kernel(a,d)*self.kernel(b,c)

    def covar_wrapper(self,u,u_prime):
        return self.skew_symmetric_forward(u,u_prime,u,u_prime)

    def forward(self,X):
        mean_x = self.mean_module(X)
        u,u_prime = torch.chunk(X,dim=1,chunks=2)
        covar_x = self.covar_wrapper(u,u_prime)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactPGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactPGP, self).__init__(train_x, train_y, likelihood)
        self.register_buffer('X_train',train_x)
        self.register_buffer('Y_train',train_y)
        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(ard_num_dims=train_x.shape[1]//2))

    def skew_symmetric_forward(self,a,b,c,d):
        return self.kernel(a,c)*self.kernel(b,d)-self.kernel(a,d)*self.kernel(b,c)

    def covar_wrapper(self,u,u_prime):
        return self.skew_symmetric_forward(u,u_prime,u,u_prime)

    def forward(self):
        mean_x = self.mean_module(self.X_train)
        u,u_prime = torch.chunk(self.X_train,dim=1,chunks=1)
        covar_x = self.covar_wrapper(u,u_prime)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactVanilla(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactVanilla, self).__init__(train_x, train_y, likelihood)
        self.register_buffer('X_train',train_x)
        self.register_buffer('Y_train',train_y)
        self.mean_module = gpytorch.means.ConstantMean()
        self.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.keops.RBFKernel(ard_num_dims=train_x.shape[1]//2))

    def skew_symmetric_forward(self,a,b,c,d):
        return self.kernel(a,c)*self.kernel(b,d)-self.kernel(a,d)*self.kernel(b,c)

    def covar_wrapper(self,u,u_prime):
        return self.skew_symmetric_forward(u,u_prime,u,u_prime)

    def forward(self):
        mean_x = self.mean_module(self.X_train)
        covar_x = self.kernel(self.X_train)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
# likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
# model = ExactGPModel(train_x, train_y, likelihood).cuda()


#
class ApproximateGPGP(ApproximateGP):
    def __init__(self, inducing_points,dim):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size[0])
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(ApproximateGPGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dim))

    def skew_symmetric_forward(self,a,b,c,d):
        return self.kernel(a,c)*self.kernel(b,d)-self.kernel(a,d)*self.kernel(b,c)

    def covar_wrapper(self,u,u_prime):
        return self.skew_symmetric_forward(u,u_prime,u,u_prime)

    def forward(self, x):
        mean_x = self.mean_module(x)
        u,u_prime = torch.chunk(x,dim=1,chunks=1)
        covar_x = self.covar_wrapper(u,u_prime)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ApproximatePGP(ApproximateGP):
    def __init__(self, inducing_points,dim):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size[0])
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(ApproximatePGP, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dim))

    def skew_symmetric_forward(self,a,b,c,d):
        return self.kernel(a,c)*self.kernel(b,d)-self.kernel(a,d)*self.kernel(b,c)

    def covar_wrapper(self,u,u_prime):
        return self.skew_symmetric_forward(u,u_prime,u,u_prime)

    def forward(self, x):
        mean_x = self.mean_module(x)
        u,u_prime = torch.chunk(x,dim=1,chunks=1)
        covar_x = self.covar_wrapper(u,u_prime)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ApproximateVanilla(ApproximateGP):
    def __init__(self, inducing_points,dim):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size[0])
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(ApproximateVanilla, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=dim))

    def skew_symmetric_forward(self,a,b,c,d):
        return self.kernel(a,c)*self.kernel(b,d)-self.kernel(a,d)*self.kernel(b,c)

    def covar_wrapper(self,u,u_prime):
        return self.skew_symmetric_forward(u,u_prime,u,u_prime)

    def forward(self, x):
        mean_x = self.mean_module(x)
        u,u_prime = torch.chunk(x,dim=1,chunks=1)
        covar_x = self.covar_wrapper(u,u_prime)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

