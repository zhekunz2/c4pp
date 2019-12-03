import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
x= np.array([5, 1, 5, 14, 3, 19, 1, 1, 4, 22], dtype=np.float32).reshape(10,1)
x=torch.tensor(x)
t= np.array([94.3, 15.7, 62.9, 126.0, 5.24, 31.4, 1.05, 1.05, 2.1, 10.5], dtype=np.float32).reshape(10,1)
t=torch.tensor(t)
N=10
N=torch.tensor(N)
def model(x,t,N):
    alpha = pyro.sample('alpha'.format(''), dist.Exponential(torch.tensor(1.0)*torch.ones([amb(1)])))
    beta = pyro.sample('beta'.format(''), dist.Gamma(torch.tensor(0.1)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    with pyro.iarange('theta_range_'.format('')):
        theta = pyro.sample('theta'.format(''), dist.Gamma(alpha*torch.ones([amb(N)]),beta*torch.ones([amb(N)])))
    pyro.sample('obs__100'.format(), dist.Poisson(theta*t), obs=x)
    
def guide(x,t,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    alpha = pyro.sample('alpha'.format(''), dist.Exponential(arg_1))
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    beta = pyro.sample('beta'.format(''), dist.Pareto(arg_2,arg_3))
    arg_4 = pyro.param('arg_4', torch.ones((amb(N))), constraint=constraints.positive)
    arg_5 = pyro.param('arg_5', torch.ones((amb(N))), constraint=constraints.positive)
    with pyro.iarange('theta_prange'):
        theta = pyro.sample('theta'.format(''), dist.Gamma(arg_4,arg_5))
    
    pass
    return { "alpha": alpha,"beta": beta,"theta": theta, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(x,t,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('alpha_mean', np.array2string(dist.Exponential(pyro.param('arg_1')).mean.detach().numpy(), separator=','))
print('beta_mean', np.array2string(dist.Pareto(pyro.param('arg_2'), pyro.param('arg_3')).mean.detach().numpy(), separator=','))
print('theta_mean', np.array2string(dist.Gamma(pyro.param('arg_4'), pyro.param('arg_5')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('alpha:')
    samplefile.write(np.array2string(np.array([guide(x,t,N)['alpha'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(x,t,N)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('theta:')
    samplefile.write(np.array2string(np.array([guide(x,t,N)['theta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
