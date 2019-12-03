import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
y= np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32).reshape(10,1)
y=torch.tensor(y)
N=10
N=torch.tensor(N)
def model(y,N):
    theta = pyro.sample('theta'.format(''), dist.Beta(torch.tensor(1.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    for n in range(1, N+1):
        pyro.sample('obs_{0}_100'.format(n), dist.Bernoulli(theta), obs=y[n-1])
    
def guide(y,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    theta = pyro.sample('theta'.format(''), dist.Beta(arg_1,arg_2))
    for n in range(1, N+1):
        pass
    
    pass
    return { "theta": theta, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(y,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('theta_mean', np.array2string(dist.Beta(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('theta:')
    samplefile.write(np.array2string(np.array([guide(y,N)['theta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
