import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
x= np.array([1.6907, 1.7242, 1.7552, 1.7842, 1.8113, 1.8369, 1.861, 1.8839], dtype=np.float32).reshape(8,1)
x=torch.tensor(x)
r= np.array([6, 13, 18, 28, 52, 53, 61, 60], dtype=np.float32).reshape(8,1)
r=torch.tensor(r)
n= np.array([59, 60, 62, 56, 63, 59, 62, 60], dtype=np.float32).reshape(8,1)
n=torch.tensor(n)
N=8
N=torch.tensor(N)
def model(x,r,n,N):
    centered_x = torch.zeros([amb(N)])
    mean_x = torch.zeros([amb(1)])
    mean_x=torch.mean(x)
    centered_x=x-mean_x
    alpha_star = pyro.sample('alpha_star'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(10000.0)*torch.ones([amb(1)])))
    beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(10000.0)*torch.ones([amb(1)])))
    p = torch.zeros([amb(N)])
    for i in range(1, N+1):
        p[i-1]=(1-torch.exp(-torch.exp(alpha_star+beta*centered_x[i-1])))
    pyro.sample('obs__100'.format(), dist.Binomial(n,p), obs=r)
    alpha = torch.zeros([amb(1)])
    llike = torch.zeros([amb(N)])
    rhat = torch.zeros([amb(N)])
    alpha=alpha_star-beta*mean_x
    for i in range(1, N+1):
        llike[i-1]=r[i-1]*torch.log(p[i-1])+(n[i-1]-r[i-1])*torch.log(1-p[i-1])
        rhat[i-1]=p[i-1]*n[i-1]
    
def guide(x,r,n,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    alpha_star = pyro.sample('alpha_star'.format(''), dist.Normal(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    beta = pyro.sample('beta'.format(''), dist.Beta(arg_3,arg_4))
    for i in range(1, N+1):
        pass
    for i in range(1, N+1):
        pass
    
    pass
    return { "beta": beta,"alpha_star": alpha_star, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(x,r,n,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('beta_mean', np.array2string(dist.Beta(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('alpha_star_mean', np.array2string(dist.Normal(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(x,r,n,N)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('alpha_star:')
    samplefile.write(np.array2string(np.array([guide(x,r,n,N)['alpha_star'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
