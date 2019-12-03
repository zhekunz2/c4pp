import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
J=3
J=torch.tensor(J)
n= np.array([48, 34, 21], dtype=np.float32).reshape(3,1)
n=torch.tensor(n)
beta=0.76
beta=torch.tensor(beta)
y= np.array([21, 20, 15], dtype=np.float32).reshape(3,1)
y=torch.tensor(y)
alpha=4.48
alpha=torch.tensor(alpha)
Z= np.array([10.0, 30.0, 50.0], dtype=np.float32).reshape(3,1)
Z=torch.tensor(Z)
sigma2=81.14
sigma2=torch.tensor(sigma2)
def model(J,n,beta,y,alpha,Z,sigma2):
    sigma = torch.zeros([amb(1)])
    sigma=torch.sqrt(sigma2)
    p = torch.zeros([amb(J)])
    theta1 = pyro.sample('theta1'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(32.0)*torch.ones([amb(1)])))
    theta2 = pyro.sample('theta2'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(32.0)*torch.ones([amb(1)])))
    with pyro.iarange('X_range_'.format('')):
        X = pyro.sample('X'.format(''), dist.Normal(alpha+beta*Z*torch.ones([amb(J)]),sigma*torch.ones([amb(J)])))
    pyro.sample('obs__100'.format(), dist.Binomial(total_count=n,logits=theta1+theta2*X), obs=y)
    
def guide(J,n,beta,y,alpha,Z,sigma2):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    theta1 = pyro.sample('theta1'.format(''), dist.Beta(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    theta2 = pyro.sample('theta2'.format(''), dist.Exponential(arg_3))
    arg_4 = pyro.param('arg_4', torch.ones((amb(J))))
    arg_5 = pyro.param('arg_5', torch.ones((amb(J))), constraint=constraints.positive)
    with pyro.iarange('X_prange'):
        X = pyro.sample('X'.format(''), dist.Normal(arg_4,arg_5))
    
    pass
    return { "theta2": theta2,"X": X,"theta1": theta1, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(J,n,beta,y,alpha,Z,sigma2)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('theta2_mean', np.array2string(dist.Exponential(pyro.param('arg_3')).mean.detach().numpy(), separator=','))
print('X_mean', np.array2string(dist.Normal(pyro.param('arg_4'), pyro.param('arg_5')).mean.detach().numpy(), separator=','))
print('theta1_mean', np.array2string(dist.Beta(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('theta2:')
    samplefile.write(np.array2string(np.array([guide(J,n,beta,y,alpha,Z,sigma2)['theta2'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('X:')
    samplefile.write(np.array2string(np.array([guide(J,n,beta,y,alpha,Z,sigma2)['X'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('theta1:')
    samplefile.write(np.array2string(np.array([guide(J,n,beta,y,alpha,Z,sigma2)['theta1'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
