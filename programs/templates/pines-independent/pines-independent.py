import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
y= np.array([3040.0, 2470.0, 3610.0, 3480.0, 3810.0, 2330.0, 1800.0, 3110.0, 3160.0, 2310.0, 4360.0, 1880.0, 3670.0, 1740.0, 2250.0, 2650.0, 4970.0, 2620.0, 2900.0, 1670.0, 2540.0, 3840.0, 3800.0, 4600.0, 1900.0, 2530.0, 2920.0, 4990.0, 1670.0, 3310.0, 3450.0, 3600.0, 2850.0, 1590.0, 3770.0, 3850.0, 2480.0, 3570.0, 2620.0, 1890.0, 3030.0, 3030.0], dtype=np.float32).reshape(42,1)
y=torch.tensor(y)
x= np.array([29.2, 24.7, 32.3, 31.3, 31.5, 24.5, 19.9, 27.3, 27.1, 24.0, 33.8, 21.5, 32.2, 22.5, 27.5, 25.6, 34.5, 26.2, 26.7, 21.1, 24.1, 30.7, 32.7, 32.6, 22.1, 25.3, 30.8, 38.9, 22.1, 29.2, 30.1, 31.4, 26.7, 22.1, 30.3, 32.0, 23.2, 30.3, 29.9, 20.8, 33.2, 28.2], dtype=np.float32).reshape(42,1)
x=torch.tensor(x)
z= np.array([25.4, 22.2, 32.2, 31.0, 30.9, 23.9, 19.2, 27.2, 26.3, 23.9, 33.2, 21.0, 29.0, 22.0, 23.8, 25.3, 34.2, 25.7, 26.4, 20.0, 23.9, 30.7, 32.6, 32.5, 20.8, 23.1, 29.8, 38.1, 21.3, 28.5, 29.2, 31.4, 25.9, 21.4, 29.8, 30.6, 22.6, 30.3, 23.8, 18.4, 29.4, 28.2], dtype=np.float32).reshape(42,1)
z=torch.tensor(z)
N=42
N=torch.tensor(N)
def model(y,x,z,N):
    y_std = torch.zeros([amb(N)])
    x_std = torch.zeros([amb(N)])
    z_std = torch.zeros([amb(N)])
    y_std=(y-torch.mean(y))/torch.std(y)
    x_std=(x-torch.mean(x))/torch.std(x)
    z_std=(z-torch.mean(z))/torch.std(z)
    mu = torch.zeros([amb(2),amb( N)])
    alpha = pyro.sample('alpha'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(10.0)*torch.ones([amb(1)])))
    beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(5.0)*torch.ones([amb(1)])))
    mu[1-1]=alpha+beta*x_std
    gamma = pyro.sample('gamma'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(10.0)*torch.ones([amb(1)])))
    delta = pyro.sample('delta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(5.0)*torch.ones([amb(1)])))
    mu[2-1]=gamma+delta*z_std
    with pyro.iarange('sigma_range_'.format('')):
        sigma = pyro.sample('sigma'.format(''), dist.Cauchy(torch.tensor(0.0)*torch.ones([amb(2)]),torch.tensor(5.0)*torch.ones([amb(2)])))
    y_std=dist.Normal(mu[0-1],sigma[1-1])
    y_std=dist.Normal(mu[1-1],sigma[2-1])
    
def guide(y,x,z,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    alpha = pyro.sample('alpha'.format(''), dist.Gamma(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    beta = pyro.sample('beta'.format(''), dist.Weibull(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))), constraint=constraints.positive)
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))), constraint=constraints.positive)
    gamma = pyro.sample('gamma'.format(''), dist.Gamma(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))), constraint=constraints.positive)
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))), constraint=constraints.positive)
    delta = pyro.sample('delta'.format(''), dist.Weibull(arg_7,arg_8))
    arg_9 = pyro.param('arg_9', torch.ones((amb(2))))
    arg_10 = pyro.param('arg_10', torch.ones((amb(2))), constraint=constraints.positive)
    with pyro.iarange('sigma_prange'):
        sigma = pyro.sample('sigma'.format(''), dist.Cauchy(arg_9,arg_10))
    
    pass
    return { "alpha": alpha,"beta": beta,"sigma": sigma,"gamma": gamma,"delta": delta, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(y,x,z,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('alpha_mean', np.array2string(dist.Gamma(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('beta_mean', np.array2string(dist.Weibull(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('sigma_mean', np.array2string(dist.Cauchy(pyro.param('arg_9'), pyro.param('arg_10')).mean.detach().numpy(), separator=','))
print('gamma_mean', np.array2string(dist.Gamma(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('delta_mean', np.array2string(dist.Weibull(pyro.param('arg_7'), pyro.param('arg_8')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('alpha:')
    samplefile.write(np.array2string(np.array([guide(y,x,z,N)['alpha'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(y,x,z,N)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma:')
    samplefile.write(np.array2string(np.array([guide(y,x,z,N)['sigma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('gamma:')
    samplefile.write(np.array2string(np.array([guide(y,x,z,N)['gamma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('delta:')
    samplefile.write(np.array2string(np.array([guide(y,x,z,N)['delta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
