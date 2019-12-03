import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
midterm= np.array([80.0, 53.0, 91.0, 63.0, 91.0, 73.0, 59.0, 69.0, 78.0, 91.0, 79.0, 81.0, 71.0, 74.0, 76.0, 80.0, 89.0, 91.0, 90.0, 93.0, 96.0, 88.0, 90.0, 79.0, 81.0, 83.0, 94.0, 70.0, 79.0, 74.0, 84.0, 79.0, 91.0, 77.0, 65.0, 76.0, 70.0, 80.0, 77.0, 89.0, 70.0, 88.0, 71.0, 66.0, 81.0, 67.0, 81.0, 79.0, 68.0, 77.0, 70.0, 75.0], dtype=np.float32).reshape(52,1)
midterm=torch.tensor(midterm)
final= np.array([103.0, 79.0, 122.0, 78.0, 135.0, 117.0, 135.0, 123.0, 109.0, 126.0, 124.0, 126.0, 130.0, 117.0, 133.0, 118.0, 127.0, 118.0, 138.0, 143.0, 135.0, 129.0, 91.0, 95.0, 121.0, 131.0, 144.0, 112.0, 132.0, 121.0, 121.0, 126.0, 116.0, 112.0, 114.0, 121.0, 126.0, 122.0, 118.0, 123.0, 87.0, 136.0, 120.0, 140.0, 125.0, 86.0, 145.0, 128.0, 121.0, 98.0, 134.0, 99.0], dtype=np.float32).reshape(52,1)
final=torch.tensor(final)
N=52
N=torch.tensor(N)
def model(midterm,final,N):
    with pyro.iarange('beta_range_'.format('')):
        beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(2)]),torch.tensor(1234.0)*torch.ones([amb(2)])))
    sigma = pyro.sample('sigma'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    pyro.sample('obs__100'.format(), dist.Normal(beta[0-1]+beta[2-1]*midterm,sigma), obs=final)
    
def guide(midterm,final,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(2))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(2))), constraint=constraints.positive)
    with pyro.iarange('beta_prange'):
        beta = pyro.sample('beta'.format(''), dist.Normal(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))))
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    sigma = pyro.sample('sigma'.format(''), dist.Normal(arg_3,arg_4))
    
    pass
    return { "beta": beta,"sigma": sigma, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(midterm,final,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('beta_mean', np.array2string(dist.Normal(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('sigma_mean', np.array2string(dist.Normal(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(midterm,final,N)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma:')
    samplefile.write(np.array2string(np.array([guide(midterm,final,N)['sigma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
