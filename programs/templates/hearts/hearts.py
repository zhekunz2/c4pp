import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
y= np.array([5, 2, 0, 0, 2, 1, 0, 0, 0, 0, 13, 0], dtype=np.float32).reshape(12,1)
y=torch.tensor(y)
x= np.array([6, 9, 17, 22, 7, 5, 5, 14, 9, 7, 9, 51], dtype=np.float32).reshape(12,1)
x=torch.tensor(x)
t= np.array([11, 11, 17, 22, 9, 6, 5, 14, 9, 7, 22, 51], dtype=np.float32).reshape(12,1)
t=torch.tensor(t)
N=12
N=torch.tensor(N)
def model(y,x,t,N):
    p = torch.zeros([amb(1)])
    log1m_theta = torch.zeros([amb(1)])
    alpha = pyro.sample('alpha'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    p=(1/(1+torch.exp(-alpha)))
    delta = pyro.sample('delta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    theta = torch.zeros([amb(1)])
    theta=(1/(1+torch.exp(-delta)))
    log1m_theta=torch.log(1-theta)
    for i in range(1, N+1):
        if y[i-1]==0:
        else:
    beta = torch.zeros([amb(1)])
    beta=torch.exp(alpha)
    
def guide(y,x,t,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    alpha = pyro.sample('alpha'.format(''), dist.StudentT(df=arg_1,loc=arg_2,scale=arg_3))
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))), constraint=constraints.positive)
    delta = pyro.sample('delta'.format(''), dist.Weibull(arg_4,arg_5))
    for i in range(1, N+1):
        pass
    
    pass
    return { "alpha": alpha,"delta": delta, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(y,x,t,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('alpha_mean', np.array2string(dist.StudentT(pyro.param('arg_1')).mean.detach().numpy(), separator=','))
print('delta_mean', np.array2string(dist.Weibull(pyro.param('arg_4'), pyro.param('arg_5')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('alpha:')
    samplefile.write(np.array2string(np.array([guide(y,x,t,N)['alpha'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('delta:')
    samplefile.write(np.array2string(np.array([guide(y,x,t,N)['delta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
