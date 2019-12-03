import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
scenario_id= np.array([1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64).reshape(40,1)
scenario_id=torch.tensor(scenario_id)
n_groups=5
n_groups=torch.tensor(n_groups)
N=40
N=torch.tensor(N)
n_scenarios=8
n_scenarios=torch.tensor(n_scenarios)
y= np.array([0.375, 0.0, 0.375, 0.0, 0.333333333333, 1.0, 0.125, 1.0, 0.25, 0.0, 0.5, 0.125, 0.5, 1.0, 0.125, 0.857142857143, 0.5, 0.666666666667, 0.333333333333, 0.0, 0.142857142857, 1.0, 0.0, 1.0, 0.142857142857, 0.0, 0.714285714286, 0.0, 0.285714285714, 1.0, 0.142857142857, 1.0, 0.428571428571, 0.0, 0.285714285714, 0.857142857143, 0.857142857143, 0.857142857143, 0.142857142857, 0.75], dtype=np.float32).reshape(40,1)
y=torch.tensor(y)
group_id= np.array([1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.int64).reshape(40,1)
group_id=torch.tensor(group_id)
def model(scenario_id,n_groups,N,n_scenarios,y,group_id):
    sigma_y = pyro.sample('sigma_y'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    sigma_b = pyro.sample('sigma_b'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    sigma_a = pyro.sample('sigma_a'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    mu_a = pyro.sample('mu_a'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    with pyro.iarange('a_range_'.format('')):
        a = pyro.sample('a'.format(''), dist.Normal(10*mu_a*torch.ones([amb(n_groups)]),sigma_a*torch.ones([amb(n_groups)])))
    mu_b = pyro.sample('mu_b'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    with pyro.iarange('b_range_'.format('')):
        b = pyro.sample('b'.format(''), dist.Normal(10*mu_b*torch.ones([amb(n_scenarios)]),sigma_b*torch.ones([amb(n_scenarios)])))
    y_hat = torch.zeros([amb(N)])
    for i in range(1, N+1):
        y_hat[i-1]=a[group_id[i-1]-1]+b[scenario_id[i-1]-1]
    pyro.sample('obs__100'.format(), dist.Normal(y_hat,sigma_y), obs=y)
    
def guide(scenario_id,n_groups,N,n_scenarios,y,group_id):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    sigma_y = pyro.sample('sigma_y'.format(''), dist.Normal(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    sigma_b = pyro.sample('sigma_b'.format(''), dist.Gamma(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))))
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))), constraint=constraints.positive)
    sigma_a = pyro.sample('sigma_a'.format(''), dist.LogNormal(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))), constraint=constraints.positive)
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))), constraint=constraints.positive)
    mu_a = pyro.sample('mu_a'.format(''), dist.Weibull(arg_7,arg_8))
    arg_9 = pyro.param('arg_9', torch.ones((amb(n_groups))), constraint=constraints.positive)
    arg_10 = pyro.param('arg_10', torch.ones((amb(n_groups))), constraint=constraints.positive)
    with pyro.iarange('a_prange'):
        a = pyro.sample('a'.format(''), dist.Beta(arg_9,arg_10))
    arg_11 = pyro.param('arg_11', torch.ones((amb(1))))
    arg_12 = pyro.param('arg_12', torch.ones((amb(1))), constraint=constraints.positive)
    mu_b = pyro.sample('mu_b'.format(''), dist.Normal(arg_11,arg_12))
    arg_13 = pyro.param('arg_13', torch.ones((amb(n_scenarios))))
    arg_14 = pyro.param('arg_14', torch.ones((amb(n_scenarios))), constraint=constraints.positive)
    with pyro.iarange('b_prange'):
        b = pyro.sample('b'.format(''), dist.Normal(arg_13,arg_14))
    for i in range(1, N+1):
        pass
    
    pass
    return { "a": a,"sigma_y": sigma_y,"b": b,"sigma_b": sigma_b,"sigma_a": sigma_a,"mu_b": mu_b,"mu_a": mu_a, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(scenario_id,n_groups,N,n_scenarios,y,group_id)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('a_mean', np.array2string(dist.Beta(pyro.param('arg_9'), pyro.param('arg_10')).mean.detach().numpy(), separator=','))
print('sigma_y_mean', np.array2string(dist.Normal(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('b_mean', np.array2string(dist.Normal(pyro.param('arg_13'), pyro.param('arg_14')).mean.detach().numpy(), separator=','))
print('sigma_b_mean', np.array2string(dist.Gamma(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('sigma_a_mean', np.array2string(dist.LogNormal(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('mu_b_mean', np.array2string(dist.Normal(pyro.param('arg_11'), pyro.param('arg_12')).mean.detach().numpy(), separator=','))
print('mu_a_mean', np.array2string(dist.Weibull(pyro.param('arg_7'), pyro.param('arg_8')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('a:')
    samplefile.write(np.array2string(np.array([guide(scenario_id,n_groups,N,n_scenarios,y,group_id)['a'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_y:')
    samplefile.write(np.array2string(np.array([guide(scenario_id,n_groups,N,n_scenarios,y,group_id)['sigma_y'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('b:')
    samplefile.write(np.array2string(np.array([guide(scenario_id,n_groups,N,n_scenarios,y,group_id)['b'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_b:')
    samplefile.write(np.array2string(np.array([guide(scenario_id,n_groups,N,n_scenarios,y,group_id)['sigma_b'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_a:')
    samplefile.write(np.array2string(np.array([guide(scenario_id,n_groups,N,n_scenarios,y,group_id)['sigma_a'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('mu_b:')
    samplefile.write(np.array2string(np.array([guide(scenario_id,n_groups,N,n_scenarios,y,group_id)['mu_b'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('mu_a:')
    samplefile.write(np.array2string(np.array([guide(scenario_id,n_groups,N,n_scenarios,y,group_id)['mu_a'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
