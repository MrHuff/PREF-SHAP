
from GPGP.kernel import *

class base_KRR:
    def __init__(self,K,y,lambd=1e-2,device='cuda: 0'):
        self.K = K.float().to(device)
        self.y = y.float().to(device)
        self.n = K.shape[0]
        self.eye = (torch.eye(self.n)*lambd).to(device)

    def fit(self):
        self.alpha,LU = torch.solve(self.y,self.K+self.eye)
        return self.alpha



