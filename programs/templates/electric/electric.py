import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
pair= np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96], dtype=np.int64).reshape(192,1)
pair=torch.tensor(pair)
y= np.array([48.9, 70.5, 89.7, 44.2, 77.5, 84.7, 78.9, 86.8, 60.8, 75.7, 95.1, 81.6, 101.2, 66.4, 96.2, 101.3, 108.9, 114.6, 111.7, 97.3, 96.5, 94.9, 104.8, 104.5, 108.3, 92.8, 104.9, 88.9, 109.9, 98.9, 113.1, 114.1, 114.0, 114.0, 92.4, 116.2, 116.9, 106.9, 104.6, 114.2, 113.6, 116.6, 114.8, 114.9, 111.0, 113.9, 60.6, 55.5, 84.8, 84.9, 101.9, 70.6, 78.4, 84.2, 108.6, 76.6, 101.9, 100.1, 91.7, 92.5, 94.4, 101.3, 102.2, 100.6, 113.8, 114.3, 87.9, 105.6, 102.5, 93.6, 109.2, 111.2, 106.4, 110.6, 112.1, 113.3, 111.1, 98.9, 112.2, 114.5, 110.3, 103.7, 110.8, 108.9, 109.6, 107.2, 115.6, 116.2, 119.6, 109.6, 122.0, 109.7, 112.4, 115.5, 119.7, 111.5, 52.3, 55.0, 80.4, 47.0, 69.7, 74.1, 72.7, 97.3, 74.1, 76.3, 84.5, 69.1, 77.0, 72.9, 94.4, 98.0, 82.4, 104.9, 102.4, 95.3, 89.5, 80.0, 96.9, 102.9, 68.9, 110.6, 107.6, 90.5, 105.8, 110.6, 111.3, 107.1, 105.8, 111.0, 91.7, 95.5, 109.4, 94.0, 101.0, 115.3, 110.6, 96.0, 111.7, 115.9, 113.9, 114.4, 54.6, 56.5, 75.2, 71.1, 75.6, 55.3, 59.3, 87.0, 73.7, 52.9, 110.3, 98.9, 97.2, 97.2, 67.6, 103.9, 103.8, 93.4, 103.1, 101.2, 83.6, 103.0, 88.5, 97.8, 103.6, 105.8, 99.6, 104.8, 86.0, 85.3, 102.9, 110.4, 115.3, 114.7, 110.2, 98.3, 101.9, 100.8, 111.6, 105.4, 116.2, 115.7, 118.0, 110.9, 113.4, 111.2, 113.3, 115.9, 115.2, 110.0], dtype=np.float32).reshape(192,1)
y=torch.tensor(y)
n_pair=96
n_pair=torch.tensor(n_pair)
treatment= np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(192,1)
treatment=torch.tensor(treatment)
N=192
N=torch.tensor(N)
def model(pair,y,n_pair,treatment,N):
    mu_a = pyro.sample('mu_a'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    sigma_a = pyro.sample('sigma_a'.format(''), dist.Uniform(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    sigma_y = pyro.sample('sigma_y'.format(''), dist.Uniform(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    with pyro.iarange('a_range_'.format('')):
        a = pyro.sample('a'.format(''), dist.Normal(100*mu_a*torch.ones([amb(n_pair)]),sigma_a*torch.ones([amb(n_pair)])))
    beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    y_hat = torch.zeros([amb(N)])
    for i in range(1, N+1):
        y_hat[i-1]=a[pair[i-1]-1]+beta*treatment[i-1]
    pyro.sample('obs__100'.format(), dist.Normal(y_hat,sigma_y), obs=y)
    
def guide(pair,y,n_pair,treatment,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    mu_a = pyro.sample('mu_a'.format(''), dist.Normal(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))))
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))))
    sigma_a = pyro.sample('sigma_a'.format(''), dist.Uniform(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))))
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))))
    sigma_y = pyro.sample('sigma_y'.format(''), dist.Uniform(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(n_pair))))
    arg_8 = pyro.param('arg_8', torch.ones((amb(n_pair))), constraint=constraints.positive)
    with pyro.iarange('a_prange'):
        a = pyro.sample('a'.format(''), dist.Normal(arg_7,arg_8))
    arg_9 = pyro.param('arg_9', torch.ones((amb(1))), constraint=constraints.positive)
    arg_10 = pyro.param('arg_10', torch.ones((amb(1))))
    arg_11 = pyro.param('arg_11', torch.ones((amb(1))), constraint=constraints.positive)
    beta = pyro.sample('beta'.format(''), dist.StudentT(df=arg_9,loc=arg_10,scale=arg_11))
    for i in range(1, N+1):
        pass
    
    pass
    return { "a": a,"sigma_a": sigma_a,"beta": beta,"sigma_y": sigma_y,"mu_a": mu_a, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(pair,y,n_pair,treatment,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('a_mean', np.array2string(dist.Normal(pyro.param('arg_7'), pyro.param('arg_8')).mean.detach().numpy(), separator=','))
print('beta_mean', np.array2string(dist.StudentT(pyro.param('arg_9')).mean.detach().numpy(), separator=','))
print('sigma_a_mean', np.array2string(dist.Uniform(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('sigma_y_mean', np.array2string(dist.Uniform(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('mu_a_mean', np.array2string(dist.Normal(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('a:')
    samplefile.write(np.array2string(np.array([guide(pair,y,n_pair,treatment,N)['a'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_a:')
    samplefile.write(np.array2string(np.array([guide(pair,y,n_pair,treatment,N)['sigma_a'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(pair,y,n_pair,treatment,N)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_y:')
    samplefile.write(np.array2string(np.array([guide(pair,y,n_pair,treatment,N)['sigma_y'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('mu_a:')
    samplefile.write(np.array2string(np.array([guide(pair,y,n_pair,treatment,N)['mu_a'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')