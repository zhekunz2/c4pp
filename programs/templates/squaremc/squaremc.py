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
    x_raw = pyro.sample('x_raw'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    y_raw = pyro.sample('y_raw'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    x = torch.zeros([amb(1)])
    y = torch.zeros([amb(1)])
    x=(1 if x_raw>0 else -1)-x_raw
    y=(1 if y_raw>0 else -1)-y_raw
    
def guide(dummy):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    x_raw = pyro.sample('x_raw'.format(''), dist.LogNormal(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    y_raw = pyro.sample('y_raw'.format(''), dist.Weibull(arg_3,arg_4))
    
    pass
    return { "y_raw": y_raw,"x_raw": x_raw, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(dummy)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('y_raw_mean', np.array2string(dist.Weibull(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('x_raw_mean', np.array2string(dist.LogNormal(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('y_raw:')
    samplefile.write(np.array2string(np.array([guide(dummy)['y_raw'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('x_raw:')
    samplefile.write(np.array2string(np.array([guide(dummy)['x_raw'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
