import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
I=183
I=torch.tensor(I)
n10=43
n10=torch.tensor(n10)
n11=12
n11=torch.tensor(n11)
n01=7
n01=torch.tensor(n01)
J=2
J=torch.tensor(J)
def model(I,n10,n11,n01,J):
    Y = torch.zeros([amb(I),amb(2)])
    est = torch.zeros([amb(I),amb(2)])
    for i in range(1, I+1):
        Y[i-1,1-1]=1
        Y[i-1,2-1]=0
    for i in range(1, n10+1):
        est[i-1,1-1]=1
        est[i-1,2-1]=0
    for i in range((n10+1), (n10+n01)+1):
        est[i-1,1-1]=0
        est[i-1,2-1]=1
    for i in range((n10+n01+1), (n10+n01+n11)+1):
        est[i-1,1-1]=1
        est[i-1,2-1]=1
    for i in range((n10+n01+n11+1), I+1):
        est[i-1,1-1]=0
        est[i-1,2-1]=0
    p = torch.zeros([amb(I),amb(2)])
    beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1000.0)*torch.ones([amb(1)])))
    for i in range(1, I+1):
        p[i-1,1-1]=torch.exp(beta*est[i-1,1-1])
        p[i-1,2-1]=torch.exp(beta*est[i-1,2-1])
        p[i-1,1-1]=p[i-1,1-1]/(p[i-1,1-1]+p[i-1,2-1])
        p[i-1,2-1]=1-p[i-1,1-1]
    
def guide(I,n10,n11,n01,J):
    for i in range(1, I+1):
        pass
    for i in range(1, n10+1):
        pass
    for i in range((n10+1), (n10+n01)+1):
        pass
    for i in range((n10+n01+1), (n10+n01+n11)+1):
        pass
    for i in range((n10+n01+n11+1), I+1):
        pass
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    beta = pyro.sample('beta'.format(''), dist.Exponential(arg_1))
    for i in range(1, I+1):
        pass
    
    pass
    return { "beta": beta, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(I,n10,n11,n01,J)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('beta_mean', np.array2string(dist.Exponential(pyro.param('arg_1')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(I,n10,n11,n01,J)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
