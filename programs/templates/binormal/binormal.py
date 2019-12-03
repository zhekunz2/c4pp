import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
dummy=0
dummy=torch.tensor(dummy)
def model(dummy):
    Sigma = torch.zeros([amb(2),amb(2)])
    mu = torch.zeros([amb(2)])
    mu[1-1]=0.0
    mu[2-1]=0.0
    Sigma[1-1,1-1]=1.0
    Sigma[2-1,2-1]=1.0
    Sigma[1-1,2-1]=0.10
    Sigma[2-1,1-1]=0.10
    with pyro.iarange('y_range_'.format('')):
        y = pyro.sample('y'.format(''), dist.MultivariateNormal(loc=mu*torch.ones([amb(2)]),covariance_matrix=Sigma*torch.ones([amb(2)])))
    
def guide(dummy):
    arg_1 = pyro.param('arg_1', torch.ones((amb(2))))
    arg_2 = pyro.param('arg_2', torch.eye((1)))
    with pyro.iarange('y_prange'):
        y = pyro.sample('y'.format(''), dist.MultivariateNormal(loc=arg_1,covariance_matrix=arg_2))
    
    pass
    return { "y": y, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(dummy)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('y_mean', np.array2string(dist.MultivariateNormal(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('y:')
    samplefile.write(np.array2string(np.array([guide(dummy)['y'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
