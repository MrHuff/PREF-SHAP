import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class GPGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        u, u_prime, v, v_prime = x
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(u,v)*self.covar_module(u_prime,v_prime)-self.covar_module(u,v_prime)*self.covar_module(u_prime,v)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)