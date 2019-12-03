import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
group_uncensored= np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=np.int64).reshape(65,1)
group_uncensored=torch.tensor(group_uncensored)
N_uncensored=65
N_uncensored=torch.tensor(N_uncensored)
M=4
M=torch.tensor(M)
t_uncensored= np.array([12.0, 17.0, 21.0, 25.0, 11.0, 26.0, 27.0, 30.0, 13.0, 12.0, 21.0, 20.0, 23.0, 25.0, 23.0, 29.0, 35.0, 31.0, 36.0, 32.0, 27.0, 23.0, 12.0, 18.0, 38.0, 29.0, 30.0, 32.0, 25.0, 30.0, 37.0, 27.0, 22.0, 26.0, 28.0, 19.0, 15.0, 12.0, 35.0, 35.0, 10.0, 22.0, 18.0, 12.0, 31.0, 24.0, 37.0, 29.0, 27.0, 18.0, 22.0, 13.0, 18.0, 29.0, 28.0, 16.0, 22.0, 26.0, 19.0, 17.0, 28.0, 26.0, 12.0, 17.0, 26.0], dtype=np.float32).reshape(65,1)
t_uncensored=torch.tensor(t_uncensored)
N_censored=15
N_censored=torch.tensor(N_censored)
censor_time= np.array([40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 10.0, 24.0, 40.0, 40.0, 20.0, 29.0, 10.0], dtype=np.float32).reshape(15,1)
censor_time=torch.tensor(censor_time)
group_censored= np.array([1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4], dtype=np.int64).reshape(15,1)
group_censored=torch.tensor(group_censored)
def model(group_uncensored,N_uncensored,M,t_uncensored,N_censored,censor_time,group_censored):
    t2_censored = torch.zeros([amb(N_censored)])
    r = pyro.sample('r'.format(''), dist.Exponential(torch.tensor(0.001)*torch.ones([amb(1)])))
    with pyro.iarange('beta_range_'.format(''), M):
        beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(M)]),torch.tensor(100.0)*torch.ones([amb(M)])))
    for n in range(1, N_uncensored+1):
        pyro.sample('obs_{0}_100'.format(n), dist.Weibull(r,torch.exp(-beta[group_uncensored[n-1]-1]/r)), obs=t_uncensored[n-1])
    for n in range(1, N_censored+1):
        with pyro.iarange('t2_censored_range_{0}'.format(n), N_censored):
            t2_censored[n-1] = pyro.sample('t2_censored{0}'.format(n-1), dist.Weibull(r*torch.ones([]),torch.exp(-beta[group_censored[n-1]-1]/r)/censor_time[n-1]*torch.ones([])))
    median = torch.zeros([amb(M)])
    pos_control = torch.zeros([amb(1)])
    test_sub = torch.zeros([amb(1)])
    veh_control = torch.zeros([amb(1)])
    for m in range(1, M+1):
        median[m-1]=torch.pow(torch.log(2)*torch.exp(-beta[m-1]), 1/r)
    veh_control=beta[2-1]-beta[1-1]
    test_sub=beta[3-1]-beta[1-1]
    pos_control=beta[4-1]-beta[1-1]
    
def guide(group_uncensored,N_uncensored,M,t_uncensored,N_censored,censor_time,group_censored):
    t2_censored = torch.zeros([amb(N_censored)])
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    r = pyro.sample('r'.format(''), dist.Exponential(arg_1))
    arg_2 = pyro.param('arg_2', torch.ones((amb(M))))
    arg_3 = pyro.param('arg_3', torch.ones((amb(M))), constraint=constraints.positive)
    with pyro.iarange('beta_prange'):
        beta = pyro.sample('beta'.format(''), dist.Cauchy(arg_2,arg_3))
    for n in range(1, N_uncensored+1):
        pass
    for n in range(1, N_censored+1):
        arg_4 = pyro.param('arg_4', torch.ones(()), constraint=constraints.positive)
        arg_5 = pyro.param('arg_5', torch.ones(()), constraint=constraints.positive)
        with pyro.iarange('t2_censored_prange'):
            t2_censored[n-1] = pyro.sample('t2_censored{0}'.format(n-1), dist.Weibull(arg_4,arg_5))
        pass
    for m in range(1, M+1):
        pass
    
    pass
    return { "t2_censored": t2_censored,"beta": beta,"r": r, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(group_uncensored,N_uncensored,M,t_uncensored,N_censored,censor_time,group_censored)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('beta_mean', np.array2string(dist.Cauchy(pyro.param('arg_2'), pyro.param('arg_3')).mean.detach().numpy(), separator=','))
print('t2_censored_mean', np.array2string(dist.Weibull(pyro.param('arg_4'), pyro.param('arg_5')).mean.detach().numpy(), separator=','))
print('r_mean', np.array2string(dist.Exponential(pyro.param('arg_1')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('t2_censored:')
    samplefile.write(np.array2string(np.array([guide(group_uncensored,N_uncensored,M,t_uncensored,N_censored,censor_time,group_censored)['t2_censored'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(group_uncensored,N_uncensored,M,t_uncensored,N_censored,censor_time,group_censored)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('r:')
    samplefile.write(np.array2string(np.array([guide(group_uncensored,N_uncensored,M,t_uncensored,N_censored,censor_time,group_censored)['r'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
