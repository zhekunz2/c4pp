import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
C=15
C=torch.tensor(C)
R=5
R=torch.tensor(R)
M=10
M=torch.tensor(M)
def model(C,R,M):
    N = pyro.sample('N'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    pyro.sample('obs__100'.format(), dist.Binomial(C,M/N), obs=R)
    
def guide(C,R,M):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    N = pyro.sample('N'.format(''), dist.Pareto(arg_1,arg_2))
    
    pass
    return { "N": N, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(C,R,M)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('N_mean', np.array2string(dist.Pareto(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('N:')
    samplefile.write(np.array2string(np.array([guide(C,R,M)['N'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
