import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
Y= np.array([1.8, 1.85, 1.87, 1.77, 2.02, 2.27, 2.15, 2.26, 2.47, 2.19, 2.26, 2.4, 2.39, 2.41, 2.5, 2.32, 2.32, 2.43, 2.47, 2.56, 2.65, 2.47, 2.64, 2.56, 2.7, 2.72, 2.57], dtype=np.float32).reshape(27,1)
Y=torch.tensor(Y)
x= np.array([1.0, 1.5, 1.5, 1.5, 2.5, 4.0, 5.0, 5.0, 7.0, 8.0, 8.5, 9.0, 9.5, 9.5, 10.0, 12.0, 12.0, 13.0, 13.0, 14.5, 15.5, 15.5, 16.5, 17.0, 22.5, 29.0, 31.5], dtype=np.float32).reshape(27,1)
x=torch.tensor(x)
N=27
N=torch.tensor(N)
def model(Y,x,N):
    m = torch.zeros([amb(N)])
    alpha = pyro.sample('alpha'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1000.0)*torch.ones([amb(1)])))
    beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1000.0)*torch.ones([amb(1)])))
    lambdax = pyro.sample('lambdax'.format(''), dist.Uniform(torch.tensor(0.5)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    for i in range(1, N+1):
        m[i-1]=alpha-beta*torch.pow(lambdax, x[i-1])
    tau = pyro.sample('tau'.format(''), dist.Gamma(torch.tensor(0.0001)*torch.ones([amb(1)]),torch.tensor(0.0001)*torch.ones([amb(1)])))
    sigma = torch.zeros([amb(1)])
    U3 = torch.zeros([amb(1)])
    sigma=1/torch.sqrt(tau)
    U3=torch.log(lambdax/(1+lambdax))
    
def guide(Y,x,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    alpha = pyro.sample('alpha'.format(''), dist.Pareto(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))))
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    beta = pyro.sample('beta'.format(''), dist.LogNormal(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))))
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))))
    lambdax = pyro.sample('lambdax'.format(''), dist.Uniform(arg_5,arg_6))
    for i in range(1, N+1):
        pass
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))), constraint=constraints.positive)
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))), constraint=constraints.positive)
    tau = pyro.sample('tau'.format(''), dist.Gamma(arg_7,arg_8))
    
    pass
    return { "alpha": alpha,"beta": beta,"lambdax": lambdax,"tau": tau, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(Y,x,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('tau_mean', np.array2string(dist.Gamma(pyro.param('arg_7'), pyro.param('arg_8')).mean.detach().numpy(), separator=','))
print('beta_mean', np.array2string(dist.LogNormal(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('lambdax_mean', np.array2string(dist.Uniform(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('alpha_mean', np.array2string(dist.Pareto(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('alpha:')
    samplefile.write(np.array2string(np.array([guide(Y,x,N)['alpha'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(Y,x,N)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('lambdax:')
    samplefile.write(np.array2string(np.array([guide(Y,x,N)['lambdax'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('tau:')
    samplefile.write(np.array2string(np.array([guide(Y,x,N)['tau'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
