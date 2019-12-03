import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
Y= np.array([[30.0, 33.0, 30.0, 32.0, 30.0, 58.0, 69.0], [51.0, 62.0, 49.0, 87.0, 111.0, 75.0, 112.0], [81.0, 115.0, 156.0, 108.0, 167.0, 125.0, 120.0], [172.0, 115.0, 179.0, 142.0, 142.0, 203.0, 139.0], [209.0, 174.0, 145.0, 203.0, 140.0, 214.0, 177.0]], dtype=np.float32)
Y=torch.tensor(Y)
x= np.array([118, 484, 664, 1004, 1231, 1372, 1582], dtype=np.float32).reshape(7,1)
x=torch.tensor(x)
K=5
K=torch.tensor(K)
N=7
N=torch.tensor(N)
def model(Y,x,K,N):
    tau = torch.zeros([amb(3)])
    tau_C = pyro.sample('tau_C'.format(''), dist.Gamma(torch.tensor(0.001)*torch.ones([amb(1)]),torch.tensor(0.001)*torch.ones([amb(1)])))
    with pyro.iarange('mu_range_'.format(''), 3):
        mu = pyro.sample('mu'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(3)]),torch.tensor(100.0)*torch.ones([amb(3)])))
    for j in range(1, 3+1):
        with pyro.iarange('tau_range_{0}'.format(j), 3):
            tau[j-1] = pyro.sample('tau{0}'.format(j-1), dist.Gamma(torch.tensor(0.001)*torch.ones([]),torch.tensor(0.001)*torch.ones([])))
    
def guide(Y,x,K,N):
    tau = torch.zeros([amb(3)])
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    tau_C = pyro.sample('tau_C'.format(''), dist.Gamma(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(3))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(3))), constraint=constraints.positive)
    with pyro.iarange('mu_prange'):
        mu = pyro.sample('mu'.format(''), dist.Pareto(arg_3,arg_4))
    for j in range(1, 3+1):
        arg_5 = pyro.param('arg_5', torch.ones(()), constraint=constraints.positive)
        arg_6 = pyro.param('arg_6', torch.ones(()), constraint=constraints.positive)
        with pyro.iarange('tau_prange'):
            tau[j-1] = pyro.sample('tau{0}'.format(j-1), dist.Gamma(arg_5,arg_6))
        pass
    
    pass
    return { "mu": mu,"tau": tau,"tau_C": tau_C, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(Y,x,K,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('mu_mean', np.array2string(dist.Pareto(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('tau_C_mean', np.array2string(dist.Gamma(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('tau_mean', np.array2string(dist.Gamma(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('mu:')
    samplefile.write(np.array2string(np.array([guide(Y,x,K,N)['mu'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('tau:')
    samplefile.write(np.array2string(np.array([guide(Y,x,K,N)['tau'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('tau_C:')
    samplefile.write(np.array2string(np.array([guide(Y,x,K,N)['tau_C'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
