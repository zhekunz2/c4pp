import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
n_airport=8
n_airport=torch.tensor(n_airport)
y= np.array([0.375, 0.0, 0.375, 0.0, 0.333333333333, 1.0, 0.125, 1.0, 0.25, 0.0, 0.5, 0.125, 0.5, 1.0, 0.125, 0.857142857143, 0.5, 0.666666666667, 0.333333333333, 0.0, 0.142857142857, 1.0, 0.0, 1.0, 0.142857142857, 0.0, 0.714285714286, 0.0, 0.285714285714, 1.0, 0.142857142857, 1.0, 0.428571428571, 0.0, 0.285714285714, 0.857142857143, 0.857142857143, 0.857142857143, 0.142857142857, 0.75], dtype=np.float32).reshape(40,1)
y=torch.tensor(y)
airport= np.array([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64).reshape(40,1)
airport=torch.tensor(airport)
treatment= np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.int64).reshape(40,1)
treatment=torch.tensor(treatment)
n_treatment=5
n_treatment=torch.tensor(n_treatment)
N=40
N=torch.tensor(N)
def model(n_airport,y,airport,treatment,n_treatment,N):
    sigma_y = pyro.sample('sigma_y'.format(''), dist.Uniform(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    sigma_d_raw = pyro.sample('sigma_d_raw'.format(''), dist.Uniform(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    sigma_g_raw = pyro.sample('sigma_g_raw'.format(''), dist.Uniform(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    xi_g = pyro.sample('xi_g'.format(''), dist.Uniform(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    xi_d = pyro.sample('xi_d'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    mu = pyro.sample('mu'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    mu_g_raw = pyro.sample('mu_g_raw'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    mu_d_raw = pyro.sample('mu_d_raw'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    with pyro.iarange('g_raw_range_'.format('')):
        g_raw = pyro.sample('g_raw'.format(''), dist.Normal(100*mu_g_raw*torch.ones([amb(n_treatment)]),sigma_g_raw*torch.ones([amb(n_treatment)])))
    with pyro.iarange('d_raw_range_'.format('')):
        d_raw = pyro.sample('d_raw'.format(''), dist.Normal(100*mu_d_raw*torch.ones([amb(n_airport)]),sigma_d_raw*torch.ones([amb(n_airport)])))
    d = torch.zeros([amb(n_airport)])
    g = torch.zeros([amb(n_treatment)])
    sigma_d = torch.zeros([amb(1)])
    sigma_g = torch.zeros([amb(1)])
    y_hat = torch.zeros([amb(N)])
    g=xi_g*(g_raw-torch.mean(g_raw))
    d=xi_d*(d_raw-torch.mean(d_raw))
    sigma_g=xi_g*sigma_g_raw
    sigma_d=torch.abs(xi_d)*sigma_d_raw
    for i in range(1, N+1):
        y_hat[i-1]=mu+g[treatment[i-1]-1]+d[airport[i-1]-1]
    pyro.sample('obs__100'.format(), dist.Normal(y_hat,sigma_y), obs=y)
    
def guide(n_airport,y,airport,treatment,n_treatment,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))))
    sigma_y = pyro.sample('sigma_y'.format(''), dist.Uniform(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))))
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))))
    sigma_d_raw = pyro.sample('sigma_d_raw'.format(''), dist.Uniform(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))))
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))))
    sigma_g_raw = pyro.sample('sigma_g_raw'.format(''), dist.Uniform(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))))
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))))
    xi_g = pyro.sample('xi_g'.format(''), dist.Uniform(arg_7,arg_8))
    arg_9 = pyro.param('arg_9', torch.ones((amb(1))), constraint=constraints.positive)
    arg_10 = pyro.param('arg_10', torch.ones((amb(1))), constraint=constraints.positive)
    xi_d = pyro.sample('xi_d'.format(''), dist.Pareto(arg_9,arg_10))
    arg_11 = pyro.param('arg_11', torch.ones((amb(1))), constraint=constraints.positive)
    arg_12 = pyro.param('arg_12', torch.ones((amb(1))), constraint=constraints.positive)
    mu = pyro.sample('mu'.format(''), dist.Gamma(arg_11,arg_12))
    arg_13 = pyro.param('arg_13', torch.ones((amb(1))), constraint=constraints.positive)
    mu_g_raw = pyro.sample('mu_g_raw'.format(''), dist.Exponential(arg_13))
    arg_14 = pyro.param('arg_14', torch.ones((amb(1))), constraint=constraints.positive)
    arg_15 = pyro.param('arg_15', torch.ones((amb(1))), constraint=constraints.positive)
    mu_d_raw = pyro.sample('mu_d_raw'.format(''), dist.Pareto(arg_14,arg_15))
    arg_16 = pyro.param('arg_16', torch.ones((amb(n_treatment))), constraint=constraints.positive)
    with pyro.iarange('g_raw_prange'):
        g_raw = pyro.sample('g_raw'.format(''), dist.Exponential(arg_16))
    arg_17 = pyro.param('arg_17', torch.ones((amb(n_airport))))
    arg_18 = pyro.param('arg_18', torch.ones((amb(n_airport))), constraint=constraints.positive)
    with pyro.iarange('d_raw_prange'):
        d_raw = pyro.sample('d_raw'.format(''), dist.Normal(arg_17,arg_18))
    for i in range(1, N+1):
        pass
    
    pass
    return { "mu": mu,"sigma_y": sigma_y,"sigma_g_raw": sigma_g_raw,"mu_d_raw": mu_d_raw,"mu_g_raw": mu_g_raw,"sigma_d_raw": sigma_d_raw,"d_raw": d_raw,"xi_d": xi_d,"xi_g": xi_g,"g_raw": g_raw, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(n_airport,y,airport,treatment,n_treatment,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('mu_mean', np.array2string(dist.Gamma(pyro.param('arg_11'), pyro.param('arg_12')).mean.detach().numpy(), separator=','))
print('sigma_y_mean', np.array2string(dist.Uniform(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('sigma_g_raw_mean', np.array2string(dist.Uniform(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('mu_d_raw_mean', np.array2string(dist.Pareto(pyro.param('arg_14'), pyro.param('arg_15')).mean.detach().numpy(), separator=','))
print('mu_g_raw_mean', np.array2string(dist.Exponential(pyro.param('arg_13')).mean.detach().numpy(), separator=','))
print('sigma_d_raw_mean', np.array2string(dist.Uniform(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('d_raw_mean', np.array2string(dist.Normal(pyro.param('arg_17'), pyro.param('arg_18')).mean.detach().numpy(), separator=','))
print('xi_d_mean', np.array2string(dist.Pareto(pyro.param('arg_9'), pyro.param('arg_10')).mean.detach().numpy(), separator=','))
print('xi_g_mean', np.array2string(dist.Uniform(pyro.param('arg_7'), pyro.param('arg_8')).mean.detach().numpy(), separator=','))
print('g_raw_mean', np.array2string(dist.Exponential(pyro.param('arg_16')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('mu:')
    samplefile.write(np.array2string(np.array([guide(n_airport,y,airport,treatment,n_treatment,N)['mu'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_y:')
    samplefile.write(np.array2string(np.array([guide(n_airport,y,airport,treatment,n_treatment,N)['sigma_y'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_g_raw:')
    samplefile.write(np.array2string(np.array([guide(n_airport,y,airport,treatment,n_treatment,N)['sigma_g_raw'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('mu_d_raw:')
    samplefile.write(np.array2string(np.array([guide(n_airport,y,airport,treatment,n_treatment,N)['mu_d_raw'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('mu_g_raw:')
    samplefile.write(np.array2string(np.array([guide(n_airport,y,airport,treatment,n_treatment,N)['mu_g_raw'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_d_raw:')
    samplefile.write(np.array2string(np.array([guide(n_airport,y,airport,treatment,n_treatment,N)['sigma_d_raw'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('d_raw:')
    samplefile.write(np.array2string(np.array([guide(n_airport,y,airport,treatment,n_treatment,N)['d_raw'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('xi_d:')
    samplefile.write(np.array2string(np.array([guide(n_airport,y,airport,treatment,n_treatment,N)['xi_d'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('xi_g:')
    samplefile.write(np.array2string(np.array([guide(n_airport,y,airport,treatment,n_treatment,N)['xi_g'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('g_raw:')
    samplefile.write(np.array2string(np.array([guide(n_airport,y,airport,treatment,n_treatment,N)['g_raw'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
