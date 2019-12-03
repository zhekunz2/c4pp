import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
T=500
T=torch.tensor(T)
def model(T):
    phi = pyro.sample('phi'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    sigma = pyro.sample('sigma'.format(''), dist.Cauchy(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(5.0)*torch.ones([amb(1)])))
    mu = pyro.sample('mu'.format(''), dist.Cauchy(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(10.0)*torch.ones([amb(1)])))
    with pyro.iarange('h_std_range_'.format('')):
        h_std = pyro.sample('h_std'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(T)]),torch.tensor(1.0)*torch.ones([amb(T)])))
    h = torch.zeros([amb(T)])
    h=h_std*sigma
    h[1-1]=h[1-1]/torch.sqrt(1-phi*phi)
    h=h+mu
    for t in range(2, T+1):
        h[t-1]=h[t-1]+phi*(h[t-1-1]-mu)
    
def guide(T):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    phi = pyro.sample('phi'.format(''), dist.Normal(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))))
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    sigma = pyro.sample('sigma'.format(''), dist.Cauchy(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))))
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))), constraint=constraints.positive)
    mu = pyro.sample('mu'.format(''), dist.Cauchy(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(T))), constraint=constraints.positive)
    arg_8 = pyro.param('arg_8', torch.ones((amb(T))), constraint=constraints.positive)
    with pyro.iarange('h_std_prange'):
        h_std = pyro.sample('h_std'.format(''), dist.Gamma(arg_7,arg_8))
    for t in range(2, T+1):
        pass
    
    pass
    return { "mu": mu,"phi": phi,"h_std": h_std,"sigma": sigma, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(T)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('mu_mean', np.array2string(dist.Cauchy(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('phi_mean', np.array2string(dist.Normal(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('h_std_mean', np.array2string(dist.Gamma(pyro.param('arg_7'), pyro.param('arg_8')).mean.detach().numpy(), separator=','))
print('sigma_mean', np.array2string(dist.Cauchy(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('mu:')
    samplefile.write(np.array2string(np.array([guide(T)['mu'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('phi:')
    samplefile.write(np.array2string(np.array([guide(T)['phi'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('h_std:')
    samplefile.write(np.array2string(np.array([guide(T)['h_std'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma:')
    samplefile.write(np.array2string(np.array([guide(T)['sigma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
