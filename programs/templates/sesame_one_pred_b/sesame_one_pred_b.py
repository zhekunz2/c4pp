import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
y= np.array([30.0, 37.0, 46.0, 14.0, 63.0, 36.0, 45.0, 47.0, 50.0, 52.0, 52.0, 29.0, 16.0, 28.0, 21.0, 45.0, 24.0, 16.0, 46.0, 50.0, 48.0, 42.0, 23.0, 27.0, 13.0, 18.0, 27.0, 17.0, 20.0, 15.0, 11.0, 17.0, 43.0, 27.0, 41.0, 23.0, 39.0, 12.0, 17.0, 16.0, 22.0, 19.0, 11.0, 6.0, 8.0, 48.0, 48.0, 36.0, 20.0, 13.0, 10.0, 47.0, 13.0, 32.0, 15.0, 14.0, 21.0, 16.0, 15.0, 20.0, 48.0, 19.0, 31.0, 28.0, 40.0, 46.0, 43.0, 47.0, 38.0, 42.0, 49.0, 17.0, 23.0, 42.0, 43.0, 48.0, 51.0, 45.0, 50.0, 30.0, 45.0, 53.0, 45.0, 21.0, 43.0, 37.0, 51.0, 20.0, 32.0, 51.0, 16.0, 36.0, 40.0, 36.0, 46.0, 43.0, 42.0, 33.0, 23.0, 19.0, 47.0, 36.0, 48.0, 36.0, 42.0, 29.0, 45.0, 37.0, 48.0, 48.0, 35.0, 21.0, 26.0, 32.0, 15.0, 11.0, 14.0, 19.0, 16.0, 18.0, 44.0, 16.0, 35.0, 15.0, 12.0, 18.0, 15.0, 13.0, 17.0, 13.0, 31.0, 13.0, 13.0, 13.0, 36.0, 15.0, 22.0, 26.0, 13.0, 14.0, 46.0, 25.0, 26.0, 42.0, 15.0, 25.0, 43.0, 27.0, 14.0, 17.0, 15.0, 16.0, 15.0, 23.0, 7.0, 14.0, 24.0, 17.0, 16.0, 37.0, 8.0, 32.0, 22.0, 22.0, 28.0, 20.0, 6.0, 14.0, 15.0, 11.0, 20.0, 16.0, 29.0, 22.0, 13.0, 20.0, 12.0, 16.0, 19.0, 16.0, 11.0, 16.0, 19.0, 11.0, 15.0, 14.0, 16.0, 13.0, 13.0, 24.0, 13.0, 25.0, 25.0, 43.0, 44.0, 13.0, 15.0, 0.0, 18.0, 10.0, 15.0, 33.0, 19.0, 15.0, 41.0, 30.0, 24.0, 45.0, 17.0, 14.0, 15.0, 14.0, 19.0, 17.0, 16.0, 32.0, 18.0, 40.0, 23.0, 23.0, 23.0, 46.0, 11.0, 20.0, 44.0, 19.0, 13.0, 15.0, 13.0, 16.0, 49.0, 13.0, 54.0, 34.0, 44.0, 33.0, 26.0, 19.0, 35.0, 32.0], dtype=np.float32).reshape(240,1)
y=torch.tensor(y)
encouraged= np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape(240,1)
encouraged=torch.tensor(encouraged)
N=240
N=torch.tensor(N)
def model(y,encouraged,N):
    with pyro.iarange('beta_range_'.format('')):
        beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(2)]),torch.tensor(1234.0)*torch.ones([amb(2)])))
    sigma = pyro.sample('sigma'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    pyro.sample('obs__100'.format(), dist.Normal(beta[0-1]+beta[2-1]*encouraged,sigma), obs=y)
    
def guide(y,encouraged,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(2))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(2))), constraint=constraints.positive)
    with pyro.iarange('beta_prange'):
        beta = pyro.sample('beta'.format(''), dist.LogNormal(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    sigma = pyro.sample('sigma'.format(''), dist.Weibull(arg_3,arg_4))
    
    pass
    return { "beta": beta,"sigma": sigma, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(y,encouraged,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('beta_mean', np.array2string(dist.LogNormal(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('sigma_mean', np.array2string(dist.Weibull(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(y,encouraged,N)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma:')
    samplefile.write(np.array2string(np.array([guide(y,encouraged,N)['sigma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
