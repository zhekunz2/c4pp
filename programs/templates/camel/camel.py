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
    mu = torch.zeros([amb(2)])
    S = torch.zeros([amb(2),amb(2)])
    mu[1-1]=0
    mu[2-1]=0
    S[1-1,1-1]=1000
    S[1-1,2-1]=0
    S[2-1,1-1]=0
    S[2-1,2-1]=1000
    y121 = pyro.sample('y121'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    y91 = pyro.sample('y91'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    y72 = pyro.sample('y72'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    y62 = pyro.sample('y62'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    y82 = pyro.sample('y82'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    y52 = pyro.sample('y52'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    y101 = pyro.sample('y101'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    y111 = pyro.sample('y111'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    with pyro.iarange('Sigma_range_'.format('')):
        Sigma = pyro.sample('Sigma'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(2),amb(2)]),torch.tensor(1234.0)*torch.ones([amb(2),amb(2)])))
    Y = torch.zeros([amb(12),amb( 2)])
    Y[1-1,1-1]=1
    Y[1-1,2-1]=1
    Y[2-1,1-1]=1
    Y[2-1,2-1]=-1
    Y[3-1,1-1]=-1
    Y[3-1,2-1]=1
    Y[4-1,1-1]=-1
    Y[4-1,2-1]=-1
    Y[5-1,1-1]=2
    Y[6-1,1-1]=2
    Y[7-1,1-1]=-2
    Y[8-1,1-1]=-2
    Y[5-1,2-1]=y52
    Y[6-1,2-1]=y62
    Y[7-1,2-1]=y72
    Y[8-1,2-1]=y82
    Y[9-1,1-1]=y91
    Y[10-1,1-1]=y101
    Y[11-1,1-1]=y111
    Y[12-1,1-1]=y121
    Y[9-1,2-1]=2
    Y[10-1,2-1]=2
    Y[11-1,2-1]=-2
    Y[12-1,2-1]=-2
    for n in range(1, 12+1):
        pyro.sample('obs_{0}_100'.format(n), dist.MultivariateNormal(loc=mu,covariance_matrix=Sigma), obs=Y[n-1])
    rho = torch.zeros([amb(1)])
    rho=Sigma[1-1,2-1]/torch.sqrt(Sigma[1-1,1-1]*Sigma[2-1,2-1])
    
def guide(dummy):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    y121 = pyro.sample('y121'.format(''), dist.Gamma(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))))
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    y91 = pyro.sample('y91'.format(''), dist.LogNormal(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))))
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))), constraint=constraints.positive)
    y72 = pyro.sample('y72'.format(''), dist.LogNormal(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))), constraint=constraints.positive)
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))))
    arg_9 = pyro.param('arg_9', torch.ones((amb(1))), constraint=constraints.positive)
    y62 = pyro.sample('y62'.format(''), dist.StudentT(df=arg_7,loc=arg_8,scale=arg_9))
    arg_10 = pyro.param('arg_10', torch.ones((amb(1))), constraint=constraints.positive)
    arg_11 = pyro.param('arg_11', torch.ones((amb(1))), constraint=constraints.positive)
    y82 = pyro.sample('y82'.format(''), dist.Beta(arg_10,arg_11))
    arg_12 = pyro.param('arg_12', torch.ones((amb(1))), constraint=constraints.positive)
    y52 = pyro.sample('y52'.format(''), dist.Exponential(arg_12))
    arg_13 = pyro.param('arg_13', torch.ones((amb(1))), constraint=constraints.positive)
    arg_14 = pyro.param('arg_14', torch.ones((amb(1))))
    arg_15 = pyro.param('arg_15', torch.ones((amb(1))), constraint=constraints.positive)
    y101 = pyro.sample('y101'.format(''), dist.StudentT(df=arg_13,loc=arg_14,scale=arg_15))
    arg_16 = pyro.param('arg_16', torch.ones((amb(1))))
    arg_17 = pyro.param('arg_17', torch.ones((amb(1))), constraint=constraints.positive)
    y111 = pyro.sample('y111'.format(''), dist.Normal(arg_16,arg_17))
    arg_18 = pyro.param('arg_18', torch.ones((amb(2),amb(2))), constraint=constraints.positive)
    with pyro.iarange('Sigma_prange'):
        Sigma = pyro.sample('Sigma'.format(''), dist.Exponential(arg_18))
    for n in range(1, 12+1):
        pass
    
    pass
    return { "y121": y121,"y91": y91,"y72": y72,"y62": y62,"y82": y82,"y52": y52,"y101": y101,"y111": y111,"Sigma": Sigma, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(dummy)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('y121_mean', np.array2string(dist.Gamma(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('y91_mean', np.array2string(dist.LogNormal(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('y72_mean', np.array2string(dist.LogNormal(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('y62_mean', np.array2string(dist.StudentT(pyro.param('arg_7')).mean.detach().numpy(), separator=','))
print('y82_mean', np.array2string(dist.Beta(pyro.param('arg_10'), pyro.param('arg_11')).mean.detach().numpy(), separator=','))
print('y52_mean', np.array2string(dist.Exponential(pyro.param('arg_12')).mean.detach().numpy(), separator=','))
print('y101_mean', np.array2string(dist.StudentT(pyro.param('arg_13')).mean.detach().numpy(), separator=','))
print('y111_mean', np.array2string(dist.Normal(pyro.param('arg_16'), pyro.param('arg_17')).mean.detach().numpy(), separator=','))
print('Sigma_mean', np.array2string(dist.Exponential(pyro.param('arg_18')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('y121:')
    samplefile.write(np.array2string(np.array([guide(dummy)['y121'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('y91:')
    samplefile.write(np.array2string(np.array([guide(dummy)['y91'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('y72:')
    samplefile.write(np.array2string(np.array([guide(dummy)['y72'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('y62:')
    samplefile.write(np.array2string(np.array([guide(dummy)['y62'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('y82:')
    samplefile.write(np.array2string(np.array([guide(dummy)['y82'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('y52:')
    samplefile.write(np.array2string(np.array([guide(dummy)['y52'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('y101:')
    samplefile.write(np.array2string(np.array([guide(dummy)['y101'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('y111:')
    samplefile.write(np.array2string(np.array([guide(dummy)['y111'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('Sigma:')
    samplefile.write(np.array2string(np.array([guide(dummy)['Sigma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
