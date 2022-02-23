import torch

class tensor_CG():
    def __init__(self,precond,reg, eps=1e-3,maxits=10,device='cuda:0'):
        self.precond=precond
        self.eps = eps
        self.reg = reg
        self.maxits=maxits
        self.device = device

    def linop(self,tens_K,p,):
        return torch.bmm(tens_K,p)+p*self.reg

    def solve(self,tens_K,b):
        r = torch.clone(b)
        z = self.precond@b
        p = torch.clone(z)
        rz = (r * z).flatten(1).sum(1,keepdim=True)
        a = 0
        k = 0
        for m in range(self.maxits):
            tmp = self.linop(tens_K,p)
            alp = rz / (p * tmp).flatten(1).sum(1,keepdim=True)
            a += alp.unsqueeze(-1) * p
            r -= alp.unsqueeze(-1) * tmp
            if (r**2).flatten(1).sum(1).mean() < self.eps:
                break
            z =  self.precond@r
            rznew = (r * z).flatten(1).sum(1,keepdim=True)
            p = z + (rznew.unsqueeze(-1) / rz.unsqueeze(-1)) * p
            rz = rznew
            k += 1
        return a





