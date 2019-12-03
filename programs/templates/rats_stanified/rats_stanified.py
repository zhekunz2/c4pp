import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
xbar=22.0
xbar=torch.tensor(xbar)
N=30
N=torch.tensor(N)
rat= np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], dtype=np.float32).reshape(150,1)
rat=torch.tensor(rat)
x= np.array([8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 22.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 29.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0, 36.0], dtype=np.float32).reshape(150,1)
x=torch.tensor(x)
y= np.array([151.0, 145.0, 147.0, 155.0, 135.0, 159.0, 141.0, 159.0, 177.0, 134.0, 160.0, 143.0, 154.0, 171.0, 163.0, 160.0, 142.0, 156.0, 157.0, 152.0, 154.0, 139.0, 146.0, 157.0, 132.0, 160.0, 169.0, 157.0, 137.0, 153.0, 199.0, 199.0, 214.0, 200.0, 188.0, 210.0, 189.0, 201.0, 236.0, 182.0, 208.0, 188.0, 200.0, 221.0, 216.0, 207.0, 187.0, 203.0, 212.0, 203.0, 205.0, 190.0, 191.0, 211.0, 185.0, 207.0, 216.0, 205.0, 180.0, 200.0, 246.0, 249.0, 263.0, 237.0, 230.0, 252.0, 231.0, 248.0, 285.0, 220.0, 261.0, 220.0, 244.0, 270.0, 242.0, 248.0, 234.0, 243.0, 259.0, 246.0, 253.0, 225.0, 229.0, 250.0, 237.0, 257.0, 261.0, 248.0, 219.0, 244.0, 283.0, 293.0, 312.0, 272.0, 280.0, 298.0, 275.0, 297.0, 350.0, 260.0, 313.0, 273.0, 289.0, 326.0, 281.0, 288.0, 280.0, 283.0, 307.0, 286.0, 298.0, 267.0, 272.0, 285.0, 286.0, 303.0, 295.0, 289.0, 258.0, 286.0, 320.0, 354.0, 328.0, 297.0, 323.0, 331.0, 305.0, 338.0, 376.0, 296.0, 352.0, 314.0, 325.0, 358.0, 312.0, 324.0, 316.0, 317.0, 336.0, 321.0, 334.0, 302.0, 302.0, 323.0, 331.0, 345.0, 333.0, 316.0, 291.0, 324.0], dtype=np.float32).reshape(150,1)
y=torch.tensor(y)
Npts=150
Npts=torch.tensor(Npts)
def model(xbar,N,rat,x,y,Npts):
    sigma_y = pyro.sample('sigma_y'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    sigma_alpha = pyro.sample('sigma_alpha'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    sigma_beta = pyro.sample('sigma_beta'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    mu_alpha = pyro.sample('mu_alpha'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    mu_beta = pyro.sample('mu_beta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    with pyro.iarange('alpha_range_'.format(''), N):
        alpha = pyro.sample('alpha'.format(''), dist.Normal(mu_alpha*torch.ones([amb(N)]),sigma_alpha*torch.ones([amb(N)])))
    with pyro.iarange('beta_range_'.format(''), N):
        beta = pyro.sample('beta'.format(''), dist.Normal(mu_beta*torch.ones([amb(N)]),sigma_beta*torch.ones([amb(N)])))
    for n in range(1, Npts+1):
        irat = torch.zeros([amb(1)])
        irat=rat[n-1]
        pyro.sample('obs_{0}_100'.format(n), dist.Normal(alpha[irat-1]+beta[irat-1]*(x[n-1]-xbar),sigma_y), obs=y[n-1])
    irat = torch.zeros([amb(1)])
    alpha0 = torch.zeros([amb(1)])
    alpha0=mu_alpha-xbar*mu_beta
    
def guide(xbar,N,rat,x,y,Npts):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    sigma_y = pyro.sample('sigma_y'.format(''), dist.Normal(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    sigma_alpha = pyro.sample('sigma_alpha'.format(''), dist.Weibull(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))))
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))), constraint=constraints.positive)
    sigma_beta = pyro.sample('sigma_beta'.format(''), dist.LogNormal(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))), constraint=constraints.positive)
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))))
    arg_9 = pyro.param('arg_9', torch.ones((amb(1))), constraint=constraints.positive)
    mu_alpha = pyro.sample('mu_alpha'.format(''), dist.StudentT(df=arg_7,loc=arg_8,scale=arg_9))
    arg_10 = pyro.param('arg_10', torch.ones((amb(1))))
    arg_11 = pyro.param('arg_11', torch.ones((amb(1))), constraint=constraints.positive)
    mu_beta = pyro.sample('mu_beta'.format(''), dist.Normal(arg_10,arg_11))
    arg_12 = pyro.param('arg_12', torch.ones((amb(N))), constraint=constraints.positive)
    arg_13 = pyro.param('arg_13', torch.ones((amb(N))), constraint=constraints.positive)
    with pyro.iarange('alpha_prange'):
        alpha = pyro.sample('alpha'.format(''), dist.Pareto(arg_12,arg_13))
    arg_14 = pyro.param('arg_14', torch.ones((amb(N))), constraint=constraints.positive)
    arg_15 = pyro.param('arg_15', torch.ones((amb(N))), constraint=constraints.positive)
    with pyro.iarange('beta_prange'):
        beta = pyro.sample('beta'.format(''), dist.Weibull(arg_14,arg_15))
    for n in range(1, Npts+1):
        pass
    
    pass
    return { "sigma_y": sigma_y,"sigma_alpha": sigma_alpha,"sigma_beta": sigma_beta,"beta": beta,"mu_alpha": mu_alpha,"alpha": alpha,"mu_beta": mu_beta, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(xbar,N,rat,x,y,Npts)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('sigma_y_mean', np.array2string(dist.Normal(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('sigma_alpha_mean', np.array2string(dist.Weibull(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('sigma_beta_mean', np.array2string(dist.LogNormal(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('beta_mean', np.array2string(dist.Weibull(pyro.param('arg_14'), pyro.param('arg_15')).mean.detach().numpy(), separator=','))
print('mu_alpha_mean', np.array2string(dist.StudentT(pyro.param('arg_7')).mean.detach().numpy(), separator=','))
print('alpha_mean', np.array2string(dist.Pareto(pyro.param('arg_12'), pyro.param('arg_13')).mean.detach().numpy(), separator=','))
print('mu_beta_mean', np.array2string(dist.Normal(pyro.param('arg_10'), pyro.param('arg_11')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('sigma_y:')
    samplefile.write(np.array2string(np.array([guide(xbar,N,rat,x,y,Npts)['sigma_y'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_alpha:')
    samplefile.write(np.array2string(np.array([guide(xbar,N,rat,x,y,Npts)['sigma_alpha'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_beta:')
    samplefile.write(np.array2string(np.array([guide(xbar,N,rat,x,y,Npts)['sigma_beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(xbar,N,rat,x,y,Npts)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('mu_alpha:')
    samplefile.write(np.array2string(np.array([guide(xbar,N,rat,x,y,Npts)['mu_alpha'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('alpha:')
    samplefile.write(np.array2string(np.array([guide(xbar,N,rat,x,y,Npts)['alpha'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('mu_beta:')
    samplefile.write(np.array2string(np.array([guide(xbar,N,rat,x,y,Npts)['mu_beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
