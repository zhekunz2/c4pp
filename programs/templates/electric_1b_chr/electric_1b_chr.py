import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
pre_test= np.array([13.8, 16.5, 18.5, 8.8, 15.3, 15.0, 19.4, 15.0, 11.8, 16.4, 16.2, 55.1, 73.1, 51.4, 62.3, 69.7, 70.9, 82.5, 83.9, 69.8, 69.2, 75.1, 68.2, 83.5, 71.7, 77.6, 96.5, 64.2, 88.3, 88.6, 102.4, 105.0, 101.4, 102.6, 76.4, 105.9, 100.8, 91.7, 97.5, 106.5, 107.4, 111.4, 110.0, 106.9, 106.7, 104.1, 12.0, 12.3, 17.2, 14.6, 18.9, 15.3, 16.6, 16.0, 20.1, 16.4, 80.3, 76.6, 70.1, 60.9, 72.3, 78.6, 77.4, 71.8, 93.0, 97.7, 65.7, 80.4, 75.6, 68.4, 83.3, 93.7, 87.3, 95.2, 80.4, 85.4, 98.0, 83.3, 100.8, 103.8, 100.7, 97.0, 97.3, 94.5, 97.7, 93.9, 109.7, 110.6, 115.2, 101.9, 119.8, 108.8, 104.6, 111.1, 115.5, 107.2, 12.3, 14.4, 17.7, 11.5, 16.4, 16.8, 18.7, 18.2, 15.4, 18.7, 17.1, 50.3, 63.3, 50.6, 66.4, 73.7, 46.0, 88.6, 78.4, 61.1, 68.9, 66.6, 67.5, 87.3, 40.8, 103.7, 95.4, 68.6, 91.1, 97.8, 100.6, 99.6, 98.7, 104.0, 78.0, 81.2, 102.5, 78.4, 91.9, 110.3, 104.9, 89.7, 106.2, 106.2, 111.9, 109.4, 11.9, 15.1, 16.8, 15.8, 15.8, 13.9, 14.5, 17.0, 15.8, 14.3, 90.0, 81.3, 74.9, 72.0, 52.2, 81.2, 82.0, 73.8, 81.1, 78.8, 63.2, 77.6, 65.9, 68.9, 78.8, 88.4, 86.5, 89.5, 55.9, 52.6, 95.3, 103.6, 108.1, 109.1, 98.9, 87.4, 92.5, 89.0, 100.6, 94.0, 111.8, 112.1, 113.9, 106.3, 110.8, 107.5, 110.0, 111.6, 109.8, 102.6], dtype=np.float32).reshape(192,1)
pre_test=torch.tensor(pre_test)
N=192
N=torch.tensor(N)
y= np.array([48.9, 70.5, 89.7, 44.2, 77.5, 84.7, 78.9, 86.8, 60.8, 75.7, 95.1, 81.6, 101.2, 66.4, 96.2, 101.3, 108.9, 114.6, 111.7, 97.3, 96.5, 94.9, 104.8, 104.5, 108.3, 92.8, 104.9, 88.9, 109.9, 98.9, 113.1, 114.1, 114.0, 114.0, 92.4, 116.2, 116.9, 106.9, 104.6, 114.2, 113.6, 116.6, 114.8, 114.9, 111.0, 113.9, 60.6, 55.5, 84.8, 84.9, 101.9, 70.6, 78.4, 84.2, 108.6, 76.6, 101.9, 100.1, 91.7, 92.5, 94.4, 101.3, 102.2, 100.6, 113.8, 114.3, 87.9, 105.6, 102.5, 93.6, 109.2, 111.2, 106.4, 110.6, 112.1, 113.3, 111.1, 98.9, 112.2, 114.5, 110.3, 103.7, 110.8, 108.9, 109.6, 107.2, 115.6, 116.2, 119.6, 109.6, 122.0, 109.7, 112.4, 115.5, 119.7, 111.5, 52.3, 55.0, 80.4, 47.0, 69.7, 74.1, 72.7, 97.3, 74.1, 76.3, 84.5, 69.1, 77.0, 72.9, 94.4, 98.0, 82.4, 104.9, 102.4, 95.3, 89.5, 80.0, 96.9, 102.9, 68.9, 110.6, 107.6, 90.5, 105.8, 110.6, 111.3, 107.1, 105.8, 111.0, 91.7, 95.5, 109.4, 94.0, 101.0, 115.3, 110.6, 96.0, 111.7, 115.9, 113.9, 114.4, 54.6, 56.5, 75.2, 71.1, 75.6, 55.3, 59.3, 87.0, 73.7, 52.9, 110.3, 98.9, 97.2, 97.2, 67.6, 103.9, 103.8, 93.4, 103.1, 101.2, 83.6, 103.0, 88.5, 97.8, 103.6, 105.8, 99.6, 104.8, 86.0, 85.3, 102.9, 110.4, 115.3, 114.7, 110.2, 98.3, 101.9, 100.8, 111.6, 105.4, 116.2, 115.7, 118.0, 110.9, 113.4, 111.2, 113.3, 115.9, 115.2, 110.0], dtype=np.float32).reshape(192,1)
y=torch.tensor(y)
n_pair=96
n_pair=torch.tensor(n_pair)
treatment= np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32).reshape(192,1)
treatment=torch.tensor(treatment)
pair= np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96], dtype=np.int64).reshape(192,1)
pair=torch.tensor(pair)
def model(pre_test,N,y,n_pair,treatment,pair):
    sigma_a = pyro.sample('sigma_a'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    sigma_y = pyro.sample('sigma_y'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    mu_a = pyro.sample('mu_a'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    with pyro.iarange('beta_range_'.format('')):
        beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(2)]),torch.tensor(100.0)*torch.ones([amb(2)])))
    with pyro.iarange('eta_range_'.format('')):
        eta = pyro.sample('eta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(n_pair)]),torch.tensor(1.0)*torch.ones([amb(n_pair)])))
    y_hat = torch.zeros([amb(N)])
    a = torch.zeros([amb(n_pair)])
    a=100*mu_a+sigma_a*eta
    for i in range(1, N+1):
        y_hat[i-1]=a[pair[i-1]-1]+beta[1-1]*treatment[i-1]+beta[2-1]*pre_test[i-1]
    pyro.sample('obs__100'.format(), dist.Normal(y_hat,sigma_y), obs=y)
    
def guide(pre_test,N,y,n_pair,treatment,pair):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    sigma_a = pyro.sample('sigma_a'.format(''), dist.Gamma(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    sigma_y = pyro.sample('sigma_y'.format(''), dist.Gamma(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))))
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))), constraint=constraints.positive)
    mu_a = pyro.sample('mu_a'.format(''), dist.Cauchy(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(2))))
    arg_8 = pyro.param('arg_8', torch.ones((amb(2))), constraint=constraints.positive)
    with pyro.iarange('beta_prange'):
        beta = pyro.sample('beta'.format(''), dist.Normal(arg_7,arg_8))
    arg_9 = pyro.param('arg_9', torch.ones((amb(n_pair))), constraint=constraints.positive)
    arg_10 = pyro.param('arg_10', torch.ones((amb(n_pair))), constraint=constraints.positive)
    with pyro.iarange('eta_prange'):
        eta = pyro.sample('eta'.format(''), dist.Gamma(arg_9,arg_10))
    for i in range(1, N+1):
        pass
    
    pass
    return { "beta": beta,"sigma_a": sigma_a,"eta": eta,"sigma_y": sigma_y,"mu_a": mu_a, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(pre_test,N,y,n_pair,treatment,pair)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('sigma_a_mean', np.array2string(dist.Gamma(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('sigma_y_mean', np.array2string(dist.Gamma(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('eta_mean', np.array2string(dist.Gamma(pyro.param('arg_9'), pyro.param('arg_10')).mean.detach().numpy(), separator=','))
print('beta_mean', np.array2string(dist.Normal(pyro.param('arg_7'), pyro.param('arg_8')).mean.detach().numpy(), separator=','))
print('mu_a_mean', np.array2string(dist.Cauchy(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(pre_test,N,y,n_pair,treatment,pair)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_a:')
    samplefile.write(np.array2string(np.array([guide(pre_test,N,y,n_pair,treatment,pair)['sigma_a'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('eta:')
    samplefile.write(np.array2string(np.array([guide(pre_test,N,y,n_pair,treatment,pair)['eta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_y:')
    samplefile.write(np.array2string(np.array([guide(pre_test,N,y,n_pair,treatment,pair)['sigma_y'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('mu_a:')
    samplefile.write(np.array2string(np.array([guide(pre_test,N,y,n_pair,treatment,pair)['mu_a'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
