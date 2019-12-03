import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
r0= np.array([0, 2, 2, 1, 2, 0, 1, 1, 1, 2, 4, 4, 2, 1, 7, 4, 3, 5, 3, 2, 4, 1, 4, 5, 2, 7, 5, 8, 2, 3, 5, 4, 1, 6, 5, 11, 5, 2, 5, 8, 5, 6, 6, 10, 7, 5, 5, 2, 8, 1, 13, 9, 11, 9, 4, 4, 8, 6, 8, 6, 8, 14, 6, 5, 5, 2, 4, 2, 9, 5, 6, 7, 5, 10, 3, 2, 1, 7, 9, 13, 9, 11, 4, 8, 2, 3, 7, 4, 7, 5, 6, 6, 5, 6, 9, 7, 7, 7, 4, 2, 3, 4, 10, 3, 4, 2, 10, 5, 4, 5, 4, 6, 5, 3, 2, 2, 4, 6, 4, 1], dtype=np.float32).reshape(120,1)
r0=torch.tensor(r0)
r1= np.array([3, 5, 2, 7, 7, 2, 5, 3, 5, 11, 6, 6, 11, 4, 4, 2, 8, 8, 6, 5, 15, 4, 9, 9, 4, 12, 8, 8, 6, 8, 12, 4, 7, 16, 12, 9, 4, 7, 8, 11, 5, 12, 8, 17, 9, 3, 2, 7, 6, 5, 11, 14, 13, 8, 6, 4, 8, 4, 8, 7, 15, 15, 9, 9, 5, 6, 3, 9, 12, 14, 16, 17, 8, 8, 9, 5, 9, 11, 6, 14, 21, 16, 6, 9, 8, 9, 8, 4, 11, 11, 6, 9, 4, 4, 9, 9, 10, 14, 6, 3, 4, 6, 10, 4, 3, 3, 10, 4, 10, 5, 4, 3, 13, 1, 7, 5, 7, 6, 3, 7], dtype=np.float32).reshape(120,1)
r1=torch.tensor(r1)
K=120
K=torch.tensor(K)
year= np.array([-10.0, -9.0, -9.0, -8.0, -8.0, -8.0, -7.0, -7.0, -7.0, -7.0, -6.0, -6.0, -6.0, -6.0, -6.0, -5.0, -5.0, -5.0, -5.0, -5.0, -5.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 9.0, 9.0, 10.0], dtype=np.float32).reshape(120,1)
year=torch.tensor(year)
n0= np.array([28, 21, 32, 35, 35, 38, 30, 43, 49, 53, 31, 35, 46, 53, 61, 40, 29, 44, 52, 55, 61, 31, 48, 44, 42, 53, 56, 71, 43, 43, 43, 40, 44, 70, 75, 71, 37, 31, 42, 46, 47, 55, 63, 91, 43, 39, 35, 32, 53, 49, 75, 64, 69, 64, 49, 29, 40, 27, 48, 43, 61, 77, 55, 60, 46, 28, 33, 32, 46, 57, 56, 78, 58, 52, 31, 28, 46, 42, 45, 63, 71, 69, 43, 50, 31, 34, 54, 46, 58, 62, 52, 41, 34, 52, 63, 59, 88, 62, 47, 53, 57, 74, 68, 61, 45, 45, 62, 73, 53, 39, 45, 51, 55, 41, 53, 51, 42, 46, 54, 32], dtype=np.float32).reshape(120,1)
n0=torch.tensor(n0)
n1= np.array([28, 21, 32, 35, 35, 38, 30, 43, 49, 53, 31, 35, 46, 53, 61, 40, 29, 44, 52, 55, 61, 31, 48, 44, 42, 53, 56, 71, 43, 43, 43, 40, 44, 70, 75, 71, 37, 31, 42, 46, 47, 55, 63, 91, 43, 39, 35, 32, 53, 49, 75, 64, 69, 64, 49, 29, 40, 27, 48, 43, 61, 77, 55, 60, 46, 28, 33, 32, 46, 57, 56, 78, 58, 52, 31, 28, 46, 42, 45, 63, 71, 69, 43, 50, 31, 34, 54, 46, 58, 62, 52, 41, 34, 52, 63, 59, 88, 62, 47, 53, 57, 74, 68, 61, 45, 45, 62, 73, 53, 39, 45, 51, 55, 41, 53, 51, 42, 46, 54, 32], dtype=np.float32).reshape(120,1)
n1=torch.tensor(n1)
def model(r0,r1,K,year,n0,n1):
    yearsq = torch.zeros([amb(K)])
    yearsq=year*year
    with pyro.iarange('b_range_'.format('')):
        b = pyro.sample('b'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(K)]),torch.tensor(1.0)*torch.ones([amb(K)])))
    with pyro.iarange('mu_range_'.format('')):
        mu = pyro.sample('mu'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(K)]),torch.tensor(1000.0)*torch.ones([amb(K)])))
    pyro.sample('obs__100'.format(), dist.Binomial(total_count=n0,logits=mu), obs=r0)
    alpha = pyro.sample('alpha'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1000.0)*torch.ones([amb(1)])))
    beta1 = pyro.sample('beta1'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1000.0)*torch.ones([amb(1)])))
    beta2 = pyro.sample('beta2'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1000.0)*torch.ones([amb(1)])))
    sigma_sq = pyro.sample('sigma_sq'.format(''), dist.Gamma(torch.tensor(0.001)*torch.ones([amb(1)]),torch.tensor(0.001)*torch.ones([amb(1)])))
    sigma = torch.zeros([amb(1)])
    sigma=torch.sqrt(sigma_sq)
    pyro.sample('obs__101'.format(), dist.Binomial(total_count=n1,logits=alpha+mu+beta1*year+beta2*(yearsq-22)+b*sigma), obs=r1)
    
def guide(r0,r1,K,year,n0,n1):
    arg_1 = pyro.param('arg_1', torch.ones((amb(K))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(K))), constraint=constraints.positive)
    with pyro.iarange('b_prange'):
        b = pyro.sample('b'.format(''), dist.Normal(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(K))))
    arg_4 = pyro.param('arg_4', torch.ones((amb(K))), constraint=constraints.positive)
    with pyro.iarange('mu_prange'):
        mu = pyro.sample('mu'.format(''), dist.LogNormal(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))))
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))), constraint=constraints.positive)
    alpha = pyro.sample('alpha'.format(''), dist.LogNormal(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))), constraint=constraints.positive)
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))), constraint=constraints.positive)
    beta1 = pyro.sample('beta1'.format(''), dist.Beta(arg_7,arg_8))
    arg_9 = pyro.param('arg_9', torch.ones((amb(1))), constraint=constraints.positive)
    arg_10 = pyro.param('arg_10', torch.ones((amb(1))), constraint=constraints.positive)
    beta2 = pyro.sample('beta2'.format(''), dist.Gamma(arg_9,arg_10))
    arg_11 = pyro.param('arg_11', torch.ones((amb(1))), constraint=constraints.positive)
    arg_12 = pyro.param('arg_12', torch.ones((amb(1))), constraint=constraints.positive)
    sigma_sq = pyro.sample('sigma_sq'.format(''), dist.Beta(arg_11,arg_12))
    
    pass
    return { "b": b,"mu": mu,"sigma_sq": sigma_sq,"beta2": beta2,"beta1": beta1,"alpha": alpha, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(r0,r1,K,year,n0,n1)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('b_mean', np.array2string(dist.Normal(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('mu_mean', np.array2string(dist.LogNormal(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('sigma_sq_mean', np.array2string(dist.Beta(pyro.param('arg_11'), pyro.param('arg_12')).mean.detach().numpy(), separator=','))
print('beta2_mean', np.array2string(dist.Gamma(pyro.param('arg_9'), pyro.param('arg_10')).mean.detach().numpy(), separator=','))
print('beta1_mean', np.array2string(dist.Beta(pyro.param('arg_7'), pyro.param('arg_8')).mean.detach().numpy(), separator=','))
print('alpha_mean', np.array2string(dist.LogNormal(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('b:')
    samplefile.write(np.array2string(np.array([guide(r0,r1,K,year,n0,n1)['b'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('mu:')
    samplefile.write(np.array2string(np.array([guide(r0,r1,K,year,n0,n1)['mu'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_sq:')
    samplefile.write(np.array2string(np.array([guide(r0,r1,K,year,n0,n1)['sigma_sq'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta2:')
    samplefile.write(np.array2string(np.array([guide(r0,r1,K,year,n0,n1)['beta2'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta1:')
    samplefile.write(np.array2string(np.array([guide(r0,r1,K,year,n0,n1)['beta1'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('alpha:')
    samplefile.write(np.array2string(np.array([guide(r0,r1,K,year,n0,n1)['alpha'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
