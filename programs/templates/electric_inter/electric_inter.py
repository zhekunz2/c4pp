import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
post_test= np.array([116.2, 116.9, 106.9, 104.6, 114.2, 113.6, 116.6, 114.8, 114.9, 111.0, 113.9, 115.6, 116.2, 119.6, 109.6, 122.0, 109.7, 112.4, 115.5, 119.7, 111.5, 95.5, 109.4, 94.0, 101.0, 115.3, 110.6, 96.0, 111.7, 115.9, 113.9, 114.4, 116.2, 115.7, 118.0, 110.9, 113.4, 111.2, 113.3, 115.9, 115.2, 110.0], dtype=np.float32).reshape(42,1)
post_test=torch.tensor(post_test)
treatment= np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(42,1)
treatment=torch.tensor(treatment)
pre_test= np.array([105.9, 100.8, 91.7, 97.5, 106.5, 107.4, 111.4, 110.0, 106.9, 106.7, 104.1, 109.7, 110.6, 115.2, 101.9, 119.8, 108.8, 104.6, 111.1, 115.5, 107.2, 81.2, 102.5, 78.4, 91.9, 110.3, 104.9, 89.7, 106.2, 106.2, 111.9, 109.4, 111.8, 112.1, 113.9, 106.3, 110.8, 107.5, 110.0, 111.6, 109.8, 102.6], dtype=np.float32).reshape(42,1)
pre_test=torch.tensor(pre_test)
N=42
N=torch.tensor(N)
def model(post_test,treatment,pre_test,N):
    inter = torch.zeros([amb(N)])
    inter=treatment*pre_test
    with pyro.iarange('beta_range_'.format('')):
        beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(4)]),torch.tensor(1234.0)*torch.ones([amb(4)])))
    sigma = pyro.sample('sigma'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    pyro.sample('obs__100'.format(), dist.Normal(beta[0-1]+beta[1-1]*treatment+beta[2-1]*pre_test+beta[4-1]*inter,sigma), obs=post_test)
    
def guide(post_test,treatment,pre_test,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(4))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(4))), constraint=constraints.positive)
    with pyro.iarange('beta_prange'):
        beta = pyro.sample('beta'.format(''), dist.Gamma(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    sigma = pyro.sample('sigma'.format(''), dist.Gamma(arg_3,arg_4))
    
    pass
    return { "beta": beta,"sigma": sigma, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(post_test,treatment,pre_test,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('beta_mean', np.array2string(dist.Gamma(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('sigma_mean', np.array2string(dist.Gamma(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(post_test,treatment,pre_test,N)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma:')
    samplefile.write(np.array2string(np.array([guide(post_test,treatment,pre_test,N)['sigma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
