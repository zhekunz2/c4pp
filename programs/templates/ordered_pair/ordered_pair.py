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
    L = torch.zeros([amb(1)])
    H = torch.zeros([amb(1)])
    L=-5.0
    H=5.0
    a = pyro.sample('a'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    b = pyro.sample('b'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    
def guide(dummy):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    a = pyro.sample('a'.format(''), dist.Exponential(arg_1))
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    b = pyro.sample('b'.format(''), dist.Beta(arg_2,arg_3))
    
    pass
    return { "a": a,"b": b, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(dummy)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('a_mean', np.array2string(dist.Exponential(pyro.param('arg_1')).mean.detach().numpy(), separator=','))
print('b_mean', np.array2string(dist.Beta(pyro.param('arg_2'), pyro.param('arg_3')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('a:')
    samplefile.write(np.array2string(np.array([guide(dummy)['a'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('b:')
    samplefile.write(np.array2string(np.array([guide(dummy)['b'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
