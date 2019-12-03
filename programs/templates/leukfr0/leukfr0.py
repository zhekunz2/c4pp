import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
Npair=21
Npair=torch.tensor(Npair)
N=42
N=torch.tensor(N)
pair= np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 19, 18, 8, 1, 20, 6, 2, 10, 3, 14, 4, 11, 7, 9, 12, 16, 17, 5, 13, 15, 21], dtype=np.int64).reshape(42,1)
pair=torch.tensor(pair)
t= np.array([1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17, 22, 23, 35], dtype=np.float32).reshape(18,1)
t=torch.tensor(t)
fail= np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0], dtype=np.float32).reshape(42,1)
fail=torch.tensor(fail)
Z= np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5], dtype=np.float32).reshape(42,1)
Z=torch.tensor(Z)
NT=17
NT=torch.tensor(NT)
obs_t= np.array([1, 1, 2, 2, 3, 4, 4, 5, 5, 8, 8, 8, 8, 11, 11, 12, 12, 15, 17, 22, 23, 6, 6, 6, 6, 7, 9, 10, 10, 11, 13, 16, 17, 19, 20, 22, 23, 25, 32, 32, 34, 35], dtype=np.float32).reshape(42,1)
obs_t=torch.tensor(obs_t)
def model(Npair,N,pair,t,fail,Z,NT,obs_t):
    dL0 = torch.zeros([amb(NT)])
    b = torch.zeros([amb(Npair)])
    Y = torch.zeros([amb(N),amb(NT)])
    dN = torch.zeros([amb(N),amb(NT)])
    c = torch.zeros([amb(1)])
    r = torch.zeros([amb(1)])
    for i in range(1, N+1):
        for j in range(1, NT+1):
            Y[i-1,j-1]=(0 if obs_t[i-1]-t[j-1]+.000000001 <= 0 else 1)
            dN[i-1,j-1]=Y[i-1,j-1]*fail[i-1]*(0 if t[j+1-1]-obs_t[i-1]-.000000001 <= 0 else 1)
    c=0.001
    r=0.1
    beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1000.0)*torch.ones([amb(1)])))
    tau = pyro.sample('tau'.format(''), dist.Gamma(torch.tensor(0.001)*torch.ones([amb(1)]),torch.tensor(0.001)*torch.ones([amb(1)])))
    sigma = torch.zeros([amb(1)])
    sigma=1/torch.sqrt(tau)
    for k in range(1, Npair+1):
        with pyro.iarange('b_range_{0}'.format(k), Npair):
            b[k-1] = pyro.sample('b{0}'.format(k-1), dist.Normal(torch.tensor(0.0)*torch.ones([]),sigma*torch.ones([])))
    for j in range(1, NT+1):
        with pyro.iarange('dL0_range_{0}'.format(j), NT):
            dL0[j-1] = pyro.sample('dL0{0}'.format(j-1), dist.Gamma(r*(t[j+1-1]-t[j-1])*c*torch.ones([]),c*torch.ones([])))
        for i in range(1, N+1):
            pyro.sample('obs_{0}_{1}_100'.format(j,i), dist.Poisson(Y[i-1,j-1]*torch.exp(beta*Z[i-1]+b[pair[i-1]-1])*dL0[j-1]), obs=dN[i-1,j-1])
    
def guide(Npair,N,pair,t,fail,Z,NT,obs_t):
    dL0 = torch.zeros([amb(NT)])
    b = torch.zeros([amb(Npair)])
    for i in range(1, N+1):
        for j in range(1, NT+1):
            pass
        pass
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    beta = pyro.sample('beta'.format(''), dist.LogNormal(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    tau = pyro.sample('tau'.format(''), dist.Pareto(arg_3,arg_4))
    for k in range(1, Npair+1):
        arg_5 = pyro.param('arg_5', torch.ones(()), constraint=constraints.positive)
        arg_6 = pyro.param('arg_6', torch.ones(()), constraint=constraints.positive)
        with pyro.iarange('b_prange'):
            b[k-1] = pyro.sample('b{0}'.format(k-1), dist.Beta(arg_5,arg_6))
        pass
    for j in range(1, NT+1):
        arg_7 = pyro.param('arg_7', torch.ones(()), constraint=constraints.positive)
        arg_8 = pyro.param('arg_8', torch.ones(()), constraint=constraints.positive)
        with pyro.iarange('dL0_prange'):
            dL0[j-1] = pyro.sample('dL0{0}'.format(j-1), dist.Gamma(arg_7,arg_8))
        for i in range(1, N+1):
            pass
        pass
    
    pass
    return { "dL0": dL0,"tau": tau,"beta": beta,"b": b, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(Npair,N,pair,t,fail,Z,NT,obs_t)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('tau_mean', np.array2string(dist.Pareto(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('beta_mean', np.array2string(dist.LogNormal(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('dL0_mean', np.array2string(dist.Gamma(pyro.param('arg_7'), pyro.param('arg_8')).mean.detach().numpy(), separator=','))
print('b_mean', np.array2string(dist.Beta(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('dL0:')
    samplefile.write(np.array2string(np.array([guide(Npair,N,pair,t,fail,Z,NT,obs_t)['dL0'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('tau:')
    samplefile.write(np.array2string(np.array([guide(Npair,N,pair,t,fail,Z,NT,obs_t)['tau'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(Npair,N,pair,t,fail,Z,NT,obs_t)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('b:')
    samplefile.write(np.array2string(np.array([guide(Npair,N,pair,t,fail,Z,NT,obs_t)['b'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
