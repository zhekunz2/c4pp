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
    y = torch.zeros([amb(5)])
    y[1-1]=2.0
    y[2-1]=1.0
    y[3-1]=-0.5
    y[4-1]=3.0
    y[5-1]=0.25
    mu = pyro.sample('mu'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    for n in range(1, 5+1):
        pyro.sample('obs_{0}_100'.format(n), dist.Normal(mu,1.0), obs=y[n-1])
    
def guide(dummy):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    mu = pyro.sample('mu'.format(''), dist.Weibull(arg_1,arg_2))
    for n in range(1, 5+1):
        pass
    
    pass
    return { "mu": mu, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(dummy)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('mu_mean', np.array2string(dist.Weibull(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('mu:')
    samplefile.write(np.array2string(np.array([guide(dummy)['mu'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
