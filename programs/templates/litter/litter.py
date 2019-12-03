import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
n= np.array([[13, 12, 12, 11, 9, 10, 9, 9, 8, 11, 8, 10, 13, 10, 12, 9], [10, 9, 10, 5, 9, 9, 13, 7, 5, 10, 7, 6, 10, 10, 10, 7]], dtype=np.float32)
n=torch.tensor(n)
r= np.array([[13, 12, 12, 11, 9, 10, 9, 9, 8, 10, 8, 9, 12, 9, 11, 8], [9, 8, 9, 4, 8, 7, 11, 4, 4, 5, 5, 3, 7, 3, 7, 0]], dtype=np.float32)
r=torch.tensor(r)
G=2
G=torch.tensor(G)
N=16
N=torch.tensor(N)
def model(n,r,G,N):
    p = torch.zeros([amb(G),amb(N)])
    with pyro.iarange('mu_range_'.format('')):
        mu = pyro.sample('mu'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(2)]),torch.tensor(1234.0)*torch.ones([amb(2)])))
    with pyro.iarange('a_plus_b_range_'.format('')):
        a_plus_b = pyro.sample('a_plus_b'.format(''), dist.Pareto(torch.tensor(0.1)*torch.ones([amb(G)]),torch.tensor(1.5)*torch.ones([amb(G)])))
    a = torch.zeros([amb(G)])
    b = torch.zeros([amb(G)])
    a=mu*a_plus_b
    b=(1-mu)*a_plus_b
    for g in range(1, G+1):
        for i in range(1, N+1):
            with pyro.iarange('p_range_{0}_{1}'.format(g,i)):
                p[g-1,i-1] = pyro.sample('p{0}_{1}'.format(g-1,i-1), dist.Beta(a[g-1]*torch.ones([]),b[g-1]*torch.ones([])))
            pyro.sample('obs_{0}_{1}_100'.format(g,i), dist.Binomial(n[g-1,i-1],p[g-1,i-1]), obs=r[g-1,i-1])
    theta = torch.zeros([amb(G)])
    for g in range(1, G+1):
        theta[g-1]=1/(a[g-1]+b[g-1])
    
def guide(n,r,G,N):
    p = torch.zeros([amb(G),amb(N)])
    arg_1 = pyro.param('arg_1', torch.ones((amb(2))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(2))), constraint=constraints.positive)
    with pyro.iarange('mu_prange'):
        mu = pyro.sample('mu'.format(''), dist.Normal(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(G))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(G))), constraint=constraints.positive)
    with pyro.iarange('a_plus_b_prange'):
        a_plus_b = pyro.sample('a_plus_b'.format(''), dist.Gamma(arg_3,arg_4))
    for g in range(1, G+1):
        for i in range(1, N+1):
            arg_5 = pyro.param('arg_5', torch.ones(()), constraint=constraints.positive)
            arg_6 = pyro.param('arg_6', torch.ones(()), constraint=constraints.positive)
            with pyro.iarange('p_prange'):
                p[g-1,i-1] = pyro.sample('p{0}_{1}'.format(g-1,i-1), dist.Beta(arg_5,arg_6))
            pass
        pass
    for g in range(1, G+1):
        pass
    
    pass
    return { "mu": mu,"p": p,"a_plus_b": a_plus_b, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(n,r,G,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('mu_mean', np.array2string(dist.Normal(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('p_mean', np.array2string(dist.Beta(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('a_plus_b_mean', np.array2string(dist.Gamma(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('mu:')
    samplefile.write(np.array2string(np.array([guide(n,r,G,N)['mu'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('p:')
    samplefile.write(np.array2string(np.array([guide(n,r,G,N)['p'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('a_plus_b:')
    samplefile.write(np.array2string(np.array([guide(n,r,G,N)['a_plus_b'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
