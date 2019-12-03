import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
n1=10
n1=torch.tensor(n1)
n2=10
n2=torch.tensor(n2)
k1=5
k1=torch.tensor(k1)
k2=7
k2=torch.tensor(k2)
def model(n1,n2,k1,k2):
    theta1 = pyro.sample('theta1'.format(''), dist.Beta(torch.tensor(1.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    theta2 = pyro.sample('theta2'.format(''), dist.Beta(torch.tensor(1.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    delta = torch.zeros([amb(1)])
    delta=theta1-theta2
    pyro.sample('obs__100'.format(), dist.Binomial(n1,theta1), obs=k1)
    pyro.sample('obs__101'.format(), dist.Binomial(n2,theta2), obs=k2)
    
def guide(n1,n2,k1,k2):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    theta1 = pyro.sample('theta1'.format(''), dist.Beta(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    theta2 = pyro.sample('theta2'.format(''), dist.Beta(arg_3,arg_4))
    
    pass
    return { "theta2": theta2,"theta1": theta1, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(n1,n2,k1,k2)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('theta2_mean', np.array2string(dist.Beta(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('theta1_mean', np.array2string(dist.Beta(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('theta2:')
    samplefile.write(np.array2string(np.array([guide(n1,n2,k1,k2)['theta2'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('theta1:')
    samplefile.write(np.array2string(np.array([guide(n1,n2,k1,k2)['theta1'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
