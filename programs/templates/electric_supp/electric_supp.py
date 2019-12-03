import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
post_test= np.array([116.2, 116.9, 106.9, 104.6, 114.2, 113.6, 116.6, 114.8, 114.9, 111.0, 113.9, 115.6, 116.2, 119.6, 109.6, 122.0, 109.7, 112.4, 115.5, 119.7, 111.5], dtype=np.float32).reshape(21,1)
post_test=torch.tensor(post_test)
supp= np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0], dtype=np.float32).reshape(21,1)
supp=torch.tensor(supp)
pre_test= np.array([105.9, 100.8, 91.7, 97.5, 106.5, 107.4, 111.4, 110.0, 106.9, 106.7, 104.1, 109.7, 110.6, 115.2, 101.9, 119.8, 108.8, 104.6, 111.1, 115.5, 107.2], dtype=np.float32).reshape(21,1)
pre_test=torch.tensor(pre_test)
N=21
N=torch.tensor(N)
def model(post_test,supp,pre_test,N):
    with pyro.iarange('beta_range_'.format('')):
        beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(3)]),torch.tensor(1234.0)*torch.ones([amb(3)])))
    sigma = pyro.sample('sigma'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    pyro.sample('obs__100'.format(), dist.Normal(beta[0-1]+beta[1-1]*supp+beta[3-1]*pre_test,sigma), obs=post_test)
    
def guide(post_test,supp,pre_test,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(3))), constraint=constraints.positive)
    with pyro.iarange('beta_prange'):
        beta = pyro.sample('beta'.format(''), dist.Exponential(arg_1))
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    sigma = pyro.sample('sigma'.format(''), dist.Gamma(arg_2,arg_3))
    
    pass
    return { "beta": beta,"sigma": sigma, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(post_test,supp,pre_test,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('beta_mean', np.array2string(dist.Exponential(pyro.param('arg_1')).mean.detach().numpy(), separator=','))
print('sigma_mean', np.array2string(dist.Gamma(pyro.param('arg_2'), pyro.param('arg_3')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(post_test,supp,pre_test,N)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma:')
    samplefile.write(np.array2string(np.array([guide(post_test,supp,pre_test,N)['sigma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
