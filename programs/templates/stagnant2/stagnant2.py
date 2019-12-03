import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
Y= np.array([1.12, 1.12, 0.99, 1.03, 0.92, 0.9, 0.81, 0.83, 0.65, 0.67, 0.6, 0.59, 0.51, 0.44, 0.43, 0.43, 0.33, 0.3, 0.25, 0.24, 0.13, -0.01, -0.13, -0.14, -0.3, -0.33, -0.46, -0.43, -0.65], dtype=np.float32).reshape(29,1)
Y=torch.tensor(Y)
x= np.array([-1.39, -1.39, -1.08, -1.08, -0.94, -0.8, -0.63, -0.63, -0.25, -0.25, -0.12, -0.12, 0.01, 0.11, 0.11, 0.11, 0.25, 0.25, 0.34, 0.34, 0.44, 0.59, 0.7, 0.7, 0.85, 0.85, 0.99, 0.99, 1.19], dtype=np.float32).reshape(29,1)
x=torch.tensor(x)
N=29
N=torch.tensor(N)
def model(Y,x,N):
    x_change = pyro.sample('x_change'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    mu = torch.zeros([amb(N)])
    alpha = pyro.sample('alpha'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(5.0)*torch.ones([amb(1)])))
    with pyro.iarange('beta_range_'.format(''), 2):
        beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(2)]),torch.tensor(5.0)*torch.ones([amb(2)])))
    sigma = pyro.sample('sigma'.format(''), dist.Cauchy(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(5.0)*torch.ones([amb(1)])))
    for n in range(1, N+1):
        mu[n-1]=alpha+(beta[1-1] if x[n-1]<x_change else beta[2-1])*(x[n-1]-x_change)
    pyro.sample('obs__100'.format(), dist.Normal(mu,sigma), obs=Y)
    
def guide(Y,x,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    x_change = pyro.sample('x_change'.format(''), dist.Beta(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    alpha = pyro.sample('alpha'.format(''), dist.Gamma(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(2))))
    arg_6 = pyro.param('arg_6', torch.ones((amb(2))), constraint=constraints.positive)
    with pyro.iarange('beta_prange'):
        beta = pyro.sample('beta'.format(''), dist.Cauchy(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))))
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))), constraint=constraints.positive)
    sigma = pyro.sample('sigma'.format(''), dist.Normal(arg_7,arg_8))
    for n in range(1, N+1):
        pass
    
    pass
    return { "alpha": alpha,"beta": beta,"sigma": sigma,"x_change": x_change, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(Y,x,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('alpha_mean', np.array2string(dist.Gamma(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('beta_mean', np.array2string(dist.Cauchy(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('sigma_mean', np.array2string(dist.Normal(pyro.param('arg_7'), pyro.param('arg_8')).mean.detach().numpy(), separator=','))
print('x_change_mean', np.array2string(dist.Beta(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('alpha:')
    samplefile.write(np.array2string(np.array([guide(Y,x,N)['alpha'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(Y,x,N)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma:')
    samplefile.write(np.array2string(np.array([guide(Y,x,N)['sigma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('x_change:')
    samplefile.write(np.array2string(np.array([guide(Y,x,N)['x_change'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
