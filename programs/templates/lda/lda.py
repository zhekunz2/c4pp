import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
doc= np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 17, 17, 17, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25, 25], dtype=np.int64).reshape(262,1)
doc=torch.tensor(doc)
K=2
K=torch.tensor(K)
M=25
M=torch.tensor(M)
N=262
N=torch.tensor(N)
beta= np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32).reshape(5,1)
beta=torch.tensor(beta)
w= np.array([4, 3, 5, 4, 3, 3, 3, 3, 3, 4, 5, 3, 4, 4, 5, 3, 4, 4, 4, 3, 5, 4, 5, 2, 3, 3, 1, 5, 5, 1, 4, 3, 1, 2, 5, 4, 4, 3, 5, 4, 2, 4, 5, 3, 4, 1, 4, 4, 3, 2, 1, 2, 1, 2, 2, 2, 1, 2, 2, 3, 1, 2, 2, 4, 4, 5, 4, 5, 5, 4, 3, 5, 4, 4, 4, 2, 2, 1, 1, 2, 1, 3, 1, 2, 1, 1, 1, 3, 2, 3, 3, 5, 4, 5, 4, 3, 5, 4, 2, 2, 2, 1, 3, 2, 1, 3, 1, 3, 1, 1, 2, 1, 2, 2, 4, 4, 4, 5, 5, 4, 4, 5, 4, 3, 3, 3, 1, 3, 3, 4, 2, 1, 3, 4, 4, 5, 4, 4, 4, 3, 4, 3, 4, 5, 1, 2, 1, 3, 2, 1, 1, 2, 3, 3, 3, 3, 4, 1, 4, 4, 4, 4, 3, 4, 4, 1, 2, 2, 3, 3, 1, 1, 4, 1, 3, 1, 5, 3, 2, 2, 1, 1, 2, 3, 3, 4, 4, 5, 3, 4, 3, 1, 5, 5, 5, 3, 3, 4, 5, 3, 3, 3, 2, 3, 1, 3, 3, 1, 3, 1, 5, 5, 5, 2, 2, 3, 3, 3, 1, 1, 5, 5, 5, 3, 1, 5, 4, 1, 3, 3, 3, 3, 4, 2, 5, 1, 3, 5, 2, 5, 5, 2, 1, 3, 3, 5, 3, 5, 3, 3, 5, 1, 2, 2, 1, 1, 2, 1, 2, 3, 1, 1], dtype=np.int64).reshape(262,1)
w=torch.tensor(w)
V=5
V=torch.tensor(V)
alpha= np.array([0.5, 0.5], dtype=np.float32).reshape(2,1)
alpha=torch.tensor(alpha)
def model(doc,K,M,N,beta,w,V,alpha):
    theta = torch.zeros([amb(M),amb(K)])
    phi = torch.zeros([amb(K),amb(V)])
    for m in range(1, M+1):
        with pyro.iarange('theta_range_{0}'.format(m)):
            theta[m-1] = pyro.sample('theta{0}'.format(m-1), dist.Dirichlet(alpha*torch.ones([amb(K)])))
    for k in range(1, K+1):
        with pyro.iarange('phi_range_{0}'.format(k)):
            phi[k-1] = pyro.sample('phi{0}'.format(k-1), dist.Dirichlet(beta*torch.ones([amb(V)])))
    for n in range(1, N+1):
        gamma = torch.zeros([amb(K)])
        for k in range(1, K+1):
            gamma[k-1]=torch.log(theta[doc[n-1]-1,k-1])+torch.log(phi[k-1,w[n-1]-1])
    gamma = torch.zeros([amb(K)])
    
def guide(doc,K,M,N,beta,w,V,alpha):
    theta = torch.zeros([amb(M),amb(K)])
    phi = torch.zeros([amb(K),amb(V)])
    for m in range(1, M+1):
        arg_1 = pyro.param('arg_1', torch.ones((amb(K))), constraint=constraints.positive)
        with pyro.iarange('theta_prange'):
            theta[m-1] = pyro.sample('theta{0}'.format(m-1), dist.Dirichlet(arg_1))
        pass
    for k in range(1, K+1):
        arg_2 = pyro.param('arg_2', torch.ones((amb(V))), constraint=constraints.positive)
        with pyro.iarange('phi_prange'):
            phi[k-1] = pyro.sample('phi{0}'.format(k-1), dist.Dirichlet(arg_2))
        pass
    for n in range(1, N+1):
        for k in range(1, K+1):
            pass
        pass
    
    pass
    return { "theta": theta,"phi": phi, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(doc,K,M,N,beta,w,V,alpha)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('theta_mean', np.array2string(dist.Dirichlet(pyro.param('arg_1')).mean.detach().numpy(), separator=','))
print('phi_mean', np.array2string(dist.Dirichlet(pyro.param('arg_2')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('theta:')
    samplefile.write(np.array2string(np.array([guide(doc,K,M,N,beta,w,V,alpha)['theta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('phi:')
    samplefile.write(np.array2string(np.array([guide(doc,K,M,N,beta,w,V,alpha)['phi'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
