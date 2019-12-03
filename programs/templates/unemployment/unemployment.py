import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
y= np.array([3.8, 5.9, 5.3, 3.3, 3.0, 2.9, 5.5, 4.4, 4.1, 4.3, 6.8, 5.5, 5.5, 6.7, 5.5, 5.7, 5.2, 4.5, 3.8, 3.8, 3.6, 3.5, 4.9, 5.9, 5.6, 4.9, 5.6, 8.5, 7.7, 7.1, 6.1, 5.8, 7.1, 7.6, 9.7, 9.6, 7.5, 7.2, 7.0, 6.2, 5.5, 5.3, 5.6, 6.8, 7.5, 6.9, 6.1, 5.6, 5.4, 4.9, 4.5, 4.2, 4.0, 4.7, 5.8, 6.0, 5.5], dtype=np.float32).reshape(57,1)
y=torch.tensor(y)
y_lag= np.array([3.9, 3.8, 5.9, 5.3, 3.3, 3.0, 2.9, 5.5, 4.4, 4.1, 4.3, 6.8, 5.5, 5.5, 6.7, 5.5, 5.7, 5.2, 4.5, 3.8, 3.8, 3.6, 3.5, 4.9, 5.9, 5.6, 4.9, 5.6, 8.5, 7.7, 7.1, 6.1, 5.8, 7.1, 7.6, 9.7, 9.6, 7.5, 7.2, 7.0, 6.2, 5.5, 5.3, 5.6, 6.8, 7.5, 6.9, 6.1, 5.6, 5.4, 4.9, 4.5, 4.2, 4.0, 4.7, 5.8, 6.0], dtype=np.float32).reshape(57,1)
y_lag=torch.tensor(y_lag)
N=57
N=torch.tensor(N)
def model(y,y_lag,N):
    with pyro.iarange('beta_range_'.format('')):
        beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(2)]),torch.tensor(1234.0)*torch.ones([amb(2)])))
    sigma = pyro.sample('sigma'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    pyro.sample('obs__100'.format(), dist.Normal(beta[0-1]+beta[2-1]*y_lag,sigma), obs=y)
    
def guide(y,y_lag,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(2))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(2))), constraint=constraints.positive)
    with pyro.iarange('beta_prange'):
        beta = pyro.sample('beta'.format(''), dist.Pareto(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    sigma = pyro.sample('sigma'.format(''), dist.Pareto(arg_3,arg_4))
    
    pass
    return { "beta": beta,"sigma": sigma, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(y,y_lag,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('beta_mean', np.array2string(dist.Pareto(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('sigma_mean', np.array2string(dist.Pareto(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(y,y_lag,N)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma:')
    samplefile.write(np.array2string(np.array([guide(y,y_lag,N)['sigma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
