import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
y= np.array([28.0, 26.0, 33.0, 24.0, 34.0, -44.0, 27.0, 16.0, 40.0, -2.0, 29.0, 22.0, 24.0, 21.0, 25.0, 30.0, 23.0, 29.0, 31.0, 19.0, 24.0, 20.0, 36.0, 32.0, 36.0, 28.0, 25.0, 21.0, 28.0, 29.0, 37.0, 25.0, 28.0, 26.0, 30.0, 32.0, 36.0, 26.0, 30.0, 22.0, 36.0, 23.0, 27.0, 27.0, 28.0, 27.0, 31.0, 27.0, 26.0, 33.0, 26.0, 32.0, 32.0, 24.0, 39.0, 28.0, 24.0, 25.0, 32.0, 25.0, 29.0, 27.0, 28.0, 29.0, 16.0, 23.0], dtype=np.float32).reshape(66,1)
y=torch.tensor(y)
N=66
N=torch.tensor(N)
def model(y,N):
    with pyro.iarange('beta_range_'.format('')):
        beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    sigma = pyro.sample('sigma'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    pyro.sample('obs__100'.format(), dist.Normal(beta[1-1],sigma), obs=y)
    
def guide(y,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    with pyro.iarange('beta_prange'):
        beta = pyro.sample('beta'.format(''), dist.Gamma(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))))
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    sigma = pyro.sample('sigma'.format(''), dist.Normal(arg_3,arg_4))
    
    pass
    return { "beta": beta,"sigma": sigma, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(y,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('beta_mean', np.array2string(dist.Gamma(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('sigma_mean', np.array2string(dist.Normal(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(y,N)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma:')
    samplefile.write(np.array2string(np.array([guide(y,N)['sigma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
