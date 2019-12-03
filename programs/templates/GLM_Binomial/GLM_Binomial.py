import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
nyears=40
nyears=torch.tensor(nyears)
year= np.array([-0.95, -0.9, -0.85, -0.8, -0.75, -0.7, -0.65, -0.6, -0.55, -0.5, -0.45, -0.4, -0.35, -0.3, -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0], dtype=np.float32).reshape(40,1)
year=torch.tensor(year)
C= np.array([27, 42, 35, 55, 61, 19, 41, 74, 43, 42, 73, 37, 48, 49, 19, 72, 30, 18, 31, 71, 63, 51, 48, 73, 49, 54, 43, 59, 30, 24, 62, 55, 51, 47, 14, 27, 45, 20, 26, 19], dtype=np.float32).reshape(40,1)
C=torch.tensor(C)
N= np.array([43, 83, 53, 91, 95, 24, 62, 91, 64, 57, 97, 56, 74, 66, 28, 92, 40, 23, 46, 96, 91, 75, 71, 100, 72, 77, 64, 68, 43, 32, 97, 92, 75, 84, 22, 58, 81, 37, 45, 39], dtype=np.float32).reshape(40,1)
N=torch.tensor(N)
def model(nyears,year,C,N):
    year_squared = torch.zeros([amb(nyears)])
    year_squared=year*year
    alpha = pyro.sample('alpha'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    beta1 = pyro.sample('beta1'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    beta2 = pyro.sample('beta2'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    logit_p = torch.zeros([amb(nyears)])
    logit_p=alpha+beta1*year+beta2*year_squared
    pyro.sample('obs__100'.format(), dist.Binomial(total_count=N,logits=logit_p), obs=C)
    p = torch.zeros([amb(nyears)])
    for i in range(1, nyears+1):
        p[i-1]=(1/(1+torch.exp(-logit_p[i-1])))
    
def guide(nyears,year,C,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    alpha = pyro.sample('alpha'.format(''), dist.Pareto(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))))
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    beta1 = pyro.sample('beta1'.format(''), dist.LogNormal(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))), constraint=constraints.positive)
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))), constraint=constraints.positive)
    beta2 = pyro.sample('beta2'.format(''), dist.Beta(arg_5,arg_6))
    for i in range(1, nyears+1):
        pass
    
    pass
    return { "alpha": alpha,"beta2": beta2,"beta1": beta1, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(nyears,year,C,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('alpha_mean', np.array2string(dist.Pareto(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('beta2_mean', np.array2string(dist.Beta(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('beta1_mean', np.array2string(dist.LogNormal(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('alpha:')
    samplefile.write(np.array2string(np.array([guide(nyears,year,C,N)['alpha'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta2:')
    samplefile.write(np.array2string(np.array([guide(nyears,year,C,N)['beta2'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta1:')
    samplefile.write(np.array2string(np.array([guide(nyears,year,C,N)['beta1'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
