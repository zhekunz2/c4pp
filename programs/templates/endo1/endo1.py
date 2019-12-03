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
def model(I,n10,n11,n01):
    J = torch.zeros([amb(1)])
    Y = torch.zeros([amb(2),amb(I)])
    est = torch.zeros([amb(2),amb( I)])
    est1m2 = torch.zeros([amb(I)])
    J=2
    for i in range(1, I+1):
        Y[1-1,i-1]=1
        Y[2-1,i-1]=0
    for i in range(1, n10+1):
        est[1-1,i-1]=1
        est[2-1,i-1]=0
    for i in range((n10+1), (n10+n01)+1):
        est[1-1,i-1]=0
        est[2-1,i-1]=1
    for i in range((n10+n01+1), (n10+n01+n11)+1):
        est[1-1,i-1]=1
        est[2-1,i-1]=1
    for i in range((n10+n01+n11+1), I+1):
        est[1-1,i-1]=0
        est[2-1,i-1]=0
    est1m2=est[1-1]-est[2-1]
    beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1000.0)*torch.ones([amb(1)])))
    pyro.sample('obs__100'.format(), dist.Binomial(total_count=1.0,logits=beta*est1m2), obs=Y[1-1])
    
def guide(I,n10,n11,n01):
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
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    beta = pyro.sample('beta'.format(''), dist.StudentT(df=arg_1,loc=arg_2,scale=arg_3))
    
    pass
    return { "beta": beta, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(I,n10,n11,n01)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('beta_mean', np.array2string(dist.StudentT(pyro.param('arg_1')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(I,n10,n11,n01)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
