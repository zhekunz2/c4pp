import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
I=21
I=torch.tensor(I)
x2= np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape(21,1)
x2=torch.tensor(x2)
N= np.array([39, 62, 81, 51, 39, 6, 74, 72, 51, 79, 13, 16, 30, 28, 45, 4, 12, 41, 30, 51, 7], dtype=np.float32).reshape(21,1)
N=torch.tensor(N)
x1= np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape(21,1)
x1=torch.tensor(x1)
n= np.array([10, 23, 23, 26, 17, 5, 53, 55, 32, 46, 10, 8, 10, 8, 23, 0, 3, 22, 15, 32, 3], dtype=np.float32).reshape(21,1)
n=torch.tensor(n)
def model(I,x2,N,x1,n):
    x1x2 = torch.zeros([amb(I)])
    x1x2=x1*x2
    alpha0 = pyro.sample('alpha0'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    alpha1 = pyro.sample('alpha1'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    alpha2 = pyro.sample('alpha2'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    alpha12 = pyro.sample('alpha12'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    sigma = pyro.sample('sigma'.format(''), dist.Cauchy(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    with pyro.iarange('c_range_'.format('')):
        c = pyro.sample('c'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(I)]),sigma*torch.ones([amb(I)])))
    b = torch.zeros([amb(I)])
    b=c-torch.mean(c)
    pyro.sample('obs__100'.format(), dist.Binomial(total_count=N,logits=alpha0+alpha1*x1+alpha2*x2+alpha12*x1x2+b), obs=n)
    
def guide(I,x2,N,x1,n):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    alpha0 = pyro.sample('alpha0'.format(''), dist.Exponential(arg_1))
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    alpha1 = pyro.sample('alpha1'.format(''), dist.Exponential(arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))))
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    alpha2 = pyro.sample('alpha2'.format(''), dist.LogNormal(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))))
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))), constraint=constraints.positive)
    alpha12 = pyro.sample('alpha12'.format(''), dist.Normal(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))), constraint=constraints.positive)
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))), constraint=constraints.positive)
    sigma = pyro.sample('sigma'.format(''), dist.Beta(arg_7,arg_8))
    arg_9 = pyro.param('arg_9', torch.ones((amb(I))))
    arg_10 = pyro.param('arg_10', torch.ones((amb(I))), constraint=constraints.positive)
    with pyro.iarange('c_prange'):
        c = pyro.sample('c'.format(''), dist.LogNormal(arg_9,arg_10))
    
    pass
    return { "alpha2": alpha2,"c": c,"alpha1": alpha1,"alpha0": alpha0,"alpha12": alpha12,"sigma": sigma, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(I,x2,N,x1,n)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('alpha2_mean', np.array2string(dist.LogNormal(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('c_mean', np.array2string(dist.LogNormal(pyro.param('arg_9'), pyro.param('arg_10')).mean.detach().numpy(), separator=','))
print('alpha1_mean', np.array2string(dist.Exponential(pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('alpha0_mean', np.array2string(dist.Exponential(pyro.param('arg_1')).mean.detach().numpy(), separator=','))
print('alpha12_mean', np.array2string(dist.Normal(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('sigma_mean', np.array2string(dist.Beta(pyro.param('arg_7'), pyro.param('arg_8')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('alpha2:')
    samplefile.write(np.array2string(np.array([guide(I,x2,N,x1,n)['alpha2'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('c:')
    samplefile.write(np.array2string(np.array([guide(I,x2,N,x1,n)['c'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('alpha1:')
    samplefile.write(np.array2string(np.array([guide(I,x2,N,x1,n)['alpha1'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('alpha0:')
    samplefile.write(np.array2string(np.array([guide(I,x2,N,x1,n)['alpha0'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('alpha12:')
    samplefile.write(np.array2string(np.array([guide(I,x2,N,x1,n)['alpha12'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma:')
    samplefile.write(np.array2string(np.array([guide(I,x2,N,x1,n)['sigma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
