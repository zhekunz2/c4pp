import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
grade= np.array([[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2], [1, 2, -1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, -1, 2, 2, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 2], [2, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, -1, -1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1], [-1, -1, 1, 1, -1, 2, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 2, 2, 1, 1, 1, -1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 1], [1, 1, 1, 1, 2, 1, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, -1, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, -1, 2, 2, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, -1, -1, 2, 1, -1, 1, 2, 2, 1, 1, 1, 1], [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 3, 2, 3, 3, 3, 3], [3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, -1, -1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1], [1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 4, 2, 3, 4, 4, 5, 5, 5, 5, 5, 5], [5, 5, 5, 1, 1, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 3, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 3, 3, 4], [5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 1, 1, 2, 3, 3, 3, 4, 5, 5, 5, 5, 1, 1, 1, 1, 3, 3, 3, 4, 4, 5, 5, 5, 5]], dtype=np.int64)
grade=torch.tensor(grade)
ncat= np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5, 5, 5, 5], dtype=np.int64).reshape(34,1)
ncat=torch.tensor(ncat)
nInd=34
nInd=torch.tensor(nInd)
nChild=13
nChild=torch.tensor(nChild)
delta= np.array([2.9541, 0.6603, 0.7965, 1.0495, 5.7874, 3.8376, 0.6324, 0.8272, 0.6968, 0.8747, 0.8136, 0.8246, 0.6711, 0.978, 1.1528, 1.6923, 1.0331, 0.5381, 1.0688, 8.1123, 0.9974, 1.2656, 1.1802, 1.368, 1.5435, 1.5006, 1.6766, 1.4297, 3.385, 3.3085, 3.4007, 2.0906, 1.0954, 1.5329], dtype=np.float32).reshape(34,1)
delta=torch.tensor(delta)
gamma= np.array([[0.7425, 10.267, 10.5215, 9.3877], [0.2593, -0.5998, 10.5891, 6.6701], [8.8921, 12.4275, 12.4788, 13.7778], [5.8374, 6.9485, 13.7184, 14.3476], [4.8066, 9.1037, 10.7483, 0.3887], [3.2573, 11.6273, 15.8842, 14.8926], [15.5487, 15.4091, 3.9216, 15.475], [0.4927, 1.3059, 1.5012, 0.8021], [5.0022, 4.0168, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, 1.0153, 7.0421, 14.4242], [17.4685, 16.7409, 16.872, 17.0061], [5.2099, 16.9406, 1.3556, 1.8793], [1.8902, 2.3873, 6.3704, 5.1537], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, 17.4944], [2.3016, 2.497, 2.3689, 3.9525], [8.2832, 7.1053, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, 3.2535, 3.2306], [2.9495, 5.3198, 10.4988, 10.3038]], dtype=np.float32)
gamma=torch.tensor(gamma)
def model(grade,ncat,nInd,nChild,delta,gamma):
    p = torch.zeros([amb(nChild),amb(nInd),amb(5)])
    Q = torch.zeros([amb(nChild),amb(nInd),amb(4)])
    with pyro.iarange('theta_range_'.format(''), nChild):
        theta = pyro.sample('theta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(nChild)]),torch.tensor(36.0)*torch.ones([amb(nChild)])))
    for i in range(1, nChild+1):
        for j in range(1, nInd+1):
            for k in range(1, (ncat[j-1]-1)+1):
                Q[i-1,j-1,k-1]=(1/(1+torch.exp(-delta[j-1]*(theta[i-1]-gamma[j-1,k-1]))))
            p[i-1,j-1,1-1]=1-Q[i-1,j-1,1-1]
            for k in range(2, (ncat[j-1]-1)+1):
                p[i-1,j-1,k-1]=Q[i-1,j-1,k-1-1]-Q[i-1,j-1,k-1]
            p[i-1,j-1,ncat[j-1]-1]=Q[i-1,j-1,ncat[j-1]-1-1]
            if grade[i-1,j-1]!=-1:
    
def guide(grade,ncat,nInd,nChild,delta,gamma):
    arg_1 = pyro.param('arg_1', torch.ones((amb(nChild))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(nChild))))
    arg_3 = pyro.param('arg_3', torch.ones((amb(nChild))), constraint=constraints.positive)
    with pyro.iarange('theta_prange'):
        theta = pyro.sample('theta'.format(''), dist.StudentT(df=arg_1,loc=arg_2,scale=arg_3))
    for i in range(1, nChild+1):
        for j in range(1, nInd+1):
            for k in range(1, (ncat[j-1]-1)+1):
                pass
            for k in range(2, (ncat[j-1]-1)+1):
                pass
            pass
        pass
    
    pass
    return { "theta": theta, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(grade,ncat,nInd,nChild,delta,gamma)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('theta_mean', np.array2string(dist.StudentT(pyro.param('arg_1')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('theta:')
    samplefile.write(np.array2string(np.array([guide(grade,ncat,nInd,nChild,delta,gamma)['theta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
