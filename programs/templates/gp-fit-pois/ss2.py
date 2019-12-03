import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
history= np.array([0, 0, 0, 176, 239, 11, 74], dtype=np.float32).reshape(7,1)
history=torch.tensor(history)
def model(history):
    with pyro.iarange('p_range_'.format(''), 3):
        p = pyro.sample('p'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(3)]),torch.tensor(1234.0)*torch.ones([amb(3)])))
    with pyro.iarange('phi_range_'.format(''), 2):
        phi = pyro.sample('phi'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(2)]),torch.tensor(1234.0)*torch.ones([amb(2)])))
    chi = torch.zeros([amb(3)])
    chi[3-1]=1
    chi[2-1]=(1-phi[2-1])+phi[2-1]*(1-p[3-1])
    chi[1-1]=(1-phi[1-1])+phi[1-1]*(1-p[2-1])*chi[2-1]
    
def guide(history):
    arg_1 = pyro.param('arg_1', torch.ones((amb(3))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(3))), constraint=constraints.positive)
    with pyro.iarange('p_prange'):
        p = pyro.sample('p'.format(''), dist.Normal(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(2))))
    arg_4 = pyro.param('arg_4', torch.ones((amb(2))), constraint=constraints.positive)
    with pyro.iarange('phi_prange'):
        phi = pyro.sample('phi'.format(''), dist.Normal(arg_3,arg_4))
    
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(history)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('p_mean', np.array2string(dist.Normal(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('phi_mean', np.array2string(dist.Normal(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
with open('pyro_out', 'w') as outputfile:
