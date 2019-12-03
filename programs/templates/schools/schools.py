import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
y= np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0], dtype=np.float32).reshape(8,1)
y=torch.tensor(y)
sigma_y= np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0], dtype=np.float32).reshape(8,1)
sigma_y=torch.tensor(sigma_y)
N=8
N=torch.tensor(N)
def model(y,sigma_y,N):
    mu_theta = pyro.sample('mu_theta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    sigma_eta = pyro.sample('sigma_eta'.format(''), dist.Gamma(torch.tensor(1.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    with pyro.iarange('eta_range_'.format('')):
        eta = pyro.sample('eta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(N)]),sigma_eta*torch.ones([amb(N)])))
    xi = pyro.sample('xi'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(5.0)*torch.ones([amb(1)])))
    sigma_theta = torch.zeros([amb(1)])
    theta = torch.zeros([amb(N)])
    theta=mu_theta+xi*eta
    sigma_theta=torch.abs(xi)/sigma_eta
    pyro.sample('obs__100'.format(), dist.Normal(theta,sigma_y), obs=y)
    
def guide(y,sigma_y,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    mu_theta = pyro.sample('mu_theta'.format(''), dist.Gamma(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    sigma_eta = pyro.sample('sigma_eta'.format(''), dist.Gamma(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(N))))
    arg_6 = pyro.param('arg_6', torch.ones((amb(N))), constraint=constraints.positive)
    with pyro.iarange('eta_prange'):
        eta = pyro.sample('eta'.format(''), dist.Cauchy(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))), constraint=constraints.positive)
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))), constraint=constraints.positive)
    xi = pyro.sample('xi'.format(''), dist.Beta(arg_7,arg_8))
    
    pass
    return { "xi": xi,"eta": eta,"sigma_eta": sigma_eta,"mu_theta": mu_theta, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(y,sigma_y,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('xi_mean', np.array2string(dist.Beta(pyro.param('arg_7'), pyro.param('arg_8')).mean.detach().numpy(), separator=','))
print('mu_theta_mean', np.array2string(dist.Gamma(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('sigma_eta_mean', np.array2string(dist.Gamma(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('eta_mean', np.array2string(dist.Cauchy(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('xi:')
    samplefile.write(np.array2string(np.array([guide(y,sigma_y,N)['xi'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('eta:')
    samplefile.write(np.array2string(np.array([guide(y,sigma_y,N)['eta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_eta:')
    samplefile.write(np.array2string(np.array([guide(y,sigma_y,N)['sigma_eta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('mu_theta:')
    samplefile.write(np.array2string(np.array([guide(y,sigma_y,N)['mu_theta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
