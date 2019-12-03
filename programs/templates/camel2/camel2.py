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
    Y = torch.zeros([amb(4),amb( 2)])
    Y1 = torch.zeros([amb(4)])
    Y2 = torch.zeros([amb(4)])
    mu = torch.zeros([amb(2)])
    S = torch.zeros([amb(2),amb(2)])
    mu[1-1]=0
    mu[2-1]=0
    S[1-1,1-1]=1000
    S[1-1,2-1]=0
    S[2-1,1-1]=0
    S[2-1,2-1]=1000
    Y[1-1,1-1]=1.0
    Y[1-1,2-1]=1.0
    Y[2-1,1-1]=1.0
    Y[2-1,2-1]=-1.0
    Y[3-1,1-1]=-1.0
    Y[3-1,2-1]=1.0
    Y[4-1,1-1]=-1.0
    Y[4-1,2-1]=-1.0
    Y1[1-1]=2.0
    Y1[2-1]=2.0
    Y1[3-1]=-2.0
    Y1[4-1]=-2.0
    Y2[1-1]=2.0
    Y2[2-1]=2.0
    Y2[3-1]=-2.0
    Y2[4-1]=-2.0
    with pyro.iarange('Sigma_range_'.format('')):
        Sigma = pyro.sample('Sigma'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(2),amb(2)]),torch.tensor(1234.0)*torch.ones([amb(2),amb(2)])))
    rho = torch.zeros([amb(1)])
    rho=Sigma[1-1,2-1]/torch.sqrt(Sigma[1-1,1-1]*Sigma[2-1,2-1])
    for n in range(1, 4+1):
        pyro.sample('obs_{0}_100'.format(n), dist.MultivariateNormal(loc=mu,covariance_matrix=Sigma), obs=Y[n-1])
    Y1=dist.Normal(0.0,torch.sqrt(Sigma[1-1,1-1]))
    Y2=dist.Normal(0.0,torch.sqrt(Sigma[2-1,2-1]))
    
def guide(dummy):
    arg_1 = pyro.param('arg_1', torch.ones((amb(2),amb(2))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(2),amb(2))), constraint=constraints.positive)
    with pyro.iarange('Sigma_prange'):
        Sigma = pyro.sample('Sigma'.format(''), dist.Beta(arg_1,arg_2))
    for n in range(1, 4+1):
        pass
    
    pass
    return { "Sigma": Sigma, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(dummy)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('Sigma_mean', np.array2string(dist.Beta(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('Sigma:')
    samplefile.write(np.array2string(np.array([guide(dummy)['Sigma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
