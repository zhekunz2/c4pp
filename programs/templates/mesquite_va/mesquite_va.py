import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
diam1= np.array([1.8, 1.7, 2.8, 1.3, 3.3, 1.4, 1.5, 3.9, 1.8, 2.1, 0.8, 1.3, 1.2, 1.5, 2.8, 1.4, 1.5, 2.4, 1.9, 2.3, 2.1, 2.4, 1.0, 1.3, 1.1, 1.3, 2.5, 5.2, 2.0, 1.6, 1.4, 3.2, 1.9, 2.4, 2.5, 2.1, 2.4, 2.4, 1.9, 2.7, 1.3, 2.9, 2.1, 4.1, 2.8, 1.27], dtype=np.float32).reshape(46,1)
diam1=torch.tensor(diam1)
diam2= np.array([1.15, 1.35, 2.55, 0.85, 1.9, 1.4, 0.5, 2.3, 1.35, 1.6, 0.63, 0.95, 0.9, 0.7, 1.7, 0.85, 0.6, 2.4, 1.55, 1.6, 1.7, 1.3, 0.4, 0.6, 0.7, 1.2, 2.3, 4.0, 1.6, 1.6, 1.0, 1.9, 1.8, 2.4, 1.8, 1.5, 2.2, 1.7, 1.2, 2.5, 1.1, 2.7, 1.0, 3.8, 2.5, 1.0], dtype=np.float32).reshape(46,1)
diam2=torch.tensor(diam2)
group= np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape(46,1)
group=torch.tensor(group)
weight= np.array([401.3, 513.7, 1179.2, 308.0, 855.2, 268.7, 155.5, 1253.2, 328.0, 614.6, 60.2, 269.6, 448.4, 120.4, 378.7, 266.4, 138.9, 1020.8, 635.7, 621.8, 579.8, 326.8, 66.7, 68.0, 153.1, 256.4, 723.0, 4052.0, 345.0, 330.9, 163.5, 1160.0, 386.6, 693.5, 674.4, 217.5, 771.3, 341.7, 125.7, 462.5, 64.5, 850.6, 226.0, 1745.1, 908.0, 213.5], dtype=np.float32).reshape(46,1)
weight=torch.tensor(weight)
canopy_height= np.array([1.0, 1.33, 0.6, 1.2, 1.05, 1.0, 0.9, 1.3, 0.6, 0.8, 0.6, 0.95, 1.2, 0.7, 1.2, 1.1, 0.64, 1.2, 1.2, 1.3, 1.0, 0.9, 1.0, 0.5, 0.9, 0.6, 1.4, 2.5, 1.4, 1.3, 1.1, 1.5, 0.8, 1.1, 1.3, 0.85, 1.5, 1.2, 1.15, 1.5, 0.7, 1.9, 1.5, 1.5, 1.5, 0.62], dtype=np.float32).reshape(46,1)
canopy_height=torch.tensor(canopy_height)
N=46
N=torch.tensor(N)
def model(diam1,diam2,group,weight,canopy_height,N):
    log_weight = torch.zeros([amb(N)])
    log_canopy_volume = torch.zeros([amb(N)])
    log_canopy_area = torch.zeros([amb(N)])
    log_weight=torch.log(weight)
    log_canopy_volume=torch.log(diam1*diam2*canopy_height)
    log_canopy_area=torch.log(diam1*diam2)
    with pyro.iarange('beta_range_'.format('')):
        beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(4)]),torch.tensor(1234.0)*torch.ones([amb(4)])))
    sigma = pyro.sample('sigma'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    log_weight=dist.Normal(beta[0-1]+beta[1-1]*log_canopy_volume+beta[2-1]*log_canopy_area+beta[4-1]*group,sigma)
    
def guide(diam1,diam2,group,weight,canopy_height,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(4))), constraint=constraints.positive)
    with pyro.iarange('beta_prange'):
        beta = pyro.sample('beta'.format(''), dist.Exponential(arg_1))
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    sigma = pyro.sample('sigma'.format(''), dist.Cauchy(arg_2,arg_3))
    
    pass
    return { "beta": beta,"sigma": sigma, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(diam1,diam2,group,weight,canopy_height,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('beta_mean', np.array2string(dist.Exponential(pyro.param('arg_1')).mean.detach().numpy(), separator=','))
print('sigma_mean', np.array2string(dist.Cauchy(pyro.param('arg_2'), pyro.param('arg_3')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(diam1,diam2,group,weight,canopy_height,N)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma:')
    samplefile.write(np.array2string(np.array([guide(diam1,diam2,group,weight,canopy_height,N)['sigma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
