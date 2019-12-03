import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
J=9
J=torch.tensor(J)
siteset= np.array([6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 2, 2, 7, 7, 7, 7, 2, 2, 7, 7, 2, 2, 2, 7, 7, 2, 2, 7, 7, 7, 7, 2, 7, 7, 2, 7, 7, 2, 2, 2, 2, 2, 2, 2, 2, 2, 7, 2, 2, 2, 2, 7, 7, 2, 2, 2, 7, 2, 2, 2, 2, 3, 3, 3, 3, 8, 8, 8, 8, 3, 3, 3, 3, 8, 8, 8, 8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8, 3, 3, 3, 8, 3, 8, 8, 8, 8, 8, 8, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 8, 3, 3, 3, 3, 3, 3, 3, 3, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 9, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5], dtype=np.float32).reshape(240,1)
siteset=torch.tensor(siteset)
yt= np.array([[30.0, 37.0], [46.0, 14.0], [63.0, 36.0], [45.0, 47.0], [50.0, 52.0], [52.0, 29.0], [16.0, 28.0], [21.0, 45.0], [24.0, 16.0], [46.0, 50.0], [48.0, 42.0], [23.0, 27.0], [13.0, 18.0], [27.0, 17.0], [20.0, 15.0], [11.0, 17.0], [43.0, 27.0], [41.0, 23.0], [39.0, 12.0], [17.0, 16.0], [22.0, 19.0], [11.0, 6.0], [8.0, 48.0], [48.0, 36.0], [20.0, 13.0], [10.0, 47.0], [13.0, 32.0], [15.0, 14.0], [21.0, 16.0], [15.0, 20.0], [48.0, 19.0], [31.0, 28.0], [40.0, 46.0], [43.0, 47.0], [38.0, 42.0], [49.0, 17.0], [23.0, 42.0], [43.0, 48.0], [51.0, 45.0], [50.0, 30.0], [45.0, 53.0], [45.0, 21.0], [43.0, 37.0], [51.0, 20.0], [32.0, 51.0], [16.0, 36.0], [40.0, 36.0], [46.0, 43.0], [42.0, 33.0], [23.0, 19.0], [47.0, 36.0], [48.0, 36.0], [42.0, 29.0], [45.0, 37.0], [48.0, 48.0], [35.0, 21.0], [26.0, 32.0], [15.0, 11.0], [14.0, 19.0], [16.0, 18.0], [44.0, 16.0], [35.0, 15.0], [12.0, 18.0], [15.0, 13.0], [17.0, 13.0], [31.0, 13.0], [13.0, 13.0], [36.0, 15.0], [22.0, 26.0], [13.0, 14.0], [46.0, 25.0], [26.0, 42.0], [15.0, 25.0], [43.0, 27.0], [14.0, 17.0], [15.0, 16.0], [15.0, 23.0], [7.0, 14.0], [24.0, 17.0], [16.0, 37.0], [8.0, 32.0], [22.0, 22.0], [28.0, 20.0], [6.0, 14.0], [15.0, 11.0], [20.0, 16.0], [29.0, 22.0], [13.0, 20.0], [12.0, 16.0], [19.0, 16.0], [11.0, 16.0], [19.0, 11.0], [15.0, 14.0], [16.0, 13.0], [13.0, 24.0], [13.0, 25.0], [25.0, 43.0], [44.0, 13.0], [15.0, 0.0], [18.0, 10.0], [15.0, 33.0], [19.0, 15.0], [41.0, 30.0], [24.0, 45.0], [17.0, 14.0], [15.0, 14.0], [19.0, 17.0], [16.0, 32.0], [18.0, 40.0], [23.0, 23.0], [23.0, 46.0], [11.0, 20.0], [44.0, 19.0], [13.0, 15.0], [13.0, 16.0], [49.0, 13.0], [54.0, 34.0], [44.0, 33.0], [26.0, 19.0], [35.0, 32.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
yt=torch.tensor(yt)
z= np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32).reshape(240,1)
z=torch.tensor(z)
N=240
N=torch.tensor(N)
def model(J,siteset,yt,z,N):
    ag = torch.zeros([amb(J),amb(2)])
    a = torch.zeros([amb(J)])
    g = torch.zeros([amb(J)])
    Sigma_ag = torch.zeros([amb(2),amb(2)])
    Sigma_yt = torch.zeros([amb(2),amb(2)])
    yt_hat = torch.zeros([amb(N),amb( 2)])
    Sigma_yt[2-1,1-1]=Sigma_yt[1-1,2-1]
    Sigma_ag[2-1,1-1]=Sigma_ag[1-1,2-1]
    sigma_y = pyro.sample('sigma_y'.format(''), dist.Uniform(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    Sigma_yt[1-1,1-1]=torch.pow(sigma_y, 2)
    sigma_t = pyro.sample('sigma_t'.format(''), dist.Uniform(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    Sigma_yt[2-1,2-1]=torch.pow(sigma_t, 2)
    rho_yt = pyro.sample('rho_yt'.format(''), dist.Uniform(-1*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    Sigma_yt[1-1,2-1]=rho_yt*sigma_y*sigma_t
    d = pyro.sample('d'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(31.6)*torch.ones([amb(1)])))
    b = pyro.sample('b'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(31.6)*torch.ones([amb(1)])))
    sigma_a = pyro.sample('sigma_a'.format(''), dist.Uniform(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    Sigma_ag[1-1,1-1]=torch.pow(sigma_a, 2)
    sigma_g = pyro.sample('sigma_g'.format(''), dist.Uniform(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    Sigma_ag[2-1,2-1]=torch.pow(sigma_g, 2)
    rho_ag = pyro.sample('rho_ag'.format(''), dist.Uniform(-1*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    Sigma_ag[1-1,2-1]=rho_ag*sigma_a*sigma_g
    with pyro.iarange('mu_ag_range_'.format('')):
        mu_ag = pyro.sample('mu_ag'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(2)]),torch.tensor(31.6)*torch.ones([amb(2)])))
    for j in range(1, J+1):
        with pyro.iarange('ag_range_{0}'.format(j)):
            ag[j-1] = pyro.sample('ag{0}'.format(j-1), dist.MultivariateNormal(loc=mu_ag*torch.ones([amb(2)]),covariance_matrix=Sigma_ag*torch.ones([amb(2)])))
    for j in range(1, J+1):
        a[j-1]=ag[j-1,1-1]
        g[j-1]=ag[j-1,2-1]
    
def guide(J,siteset,yt,z,N):
    ag = torch.zeros([amb(J),amb(2)])
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))))
    sigma_y = pyro.sample('sigma_y'.format(''), dist.Uniform(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))))
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))))
    sigma_t = pyro.sample('sigma_t'.format(''), dist.Uniform(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))))
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))))
    rho_yt = pyro.sample('rho_yt'.format(''), dist.Uniform(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))), constraint=constraints.positive)
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))), constraint=constraints.positive)
    d = pyro.sample('d'.format(''), dist.Beta(arg_7,arg_8))
    arg_9 = pyro.param('arg_9', torch.ones((amb(1))))
    arg_10 = pyro.param('arg_10', torch.ones((amb(1))), constraint=constraints.positive)
    b = pyro.sample('b'.format(''), dist.LogNormal(arg_9,arg_10))
    arg_11 = pyro.param('arg_11', torch.ones((amb(1))))
    arg_12 = pyro.param('arg_12', torch.ones((amb(1))))
    sigma_a = pyro.sample('sigma_a'.format(''), dist.Uniform(arg_11,arg_12))
    arg_13 = pyro.param('arg_13', torch.ones((amb(1))))
    arg_14 = pyro.param('arg_14', torch.ones((amb(1))))
    sigma_g = pyro.sample('sigma_g'.format(''), dist.Uniform(arg_13,arg_14))
    arg_15 = pyro.param('arg_15', torch.ones((amb(1))))
    arg_16 = pyro.param('arg_16', torch.ones((amb(1))))
    rho_ag = pyro.sample('rho_ag'.format(''), dist.Uniform(arg_15,arg_16))
    arg_17 = pyro.param('arg_17', torch.ones((amb(2))))
    arg_18 = pyro.param('arg_18', torch.ones((amb(2))), constraint=constraints.positive)
    with pyro.iarange('mu_ag_prange'):
        mu_ag = pyro.sample('mu_ag'.format(''), dist.Normal(arg_17,arg_18))
    for j in range(1, J+1):
        arg_19 = pyro.param('arg_19', torch.ones((amb(2))))
        arg_20 = pyro.param('arg_20', torch.eye((1)))
        with pyro.iarange('ag_prange'):
            ag[j-1] = pyro.sample('ag{0}'.format(j-1), dist.MultivariateNormal(loc=arg_19,covariance_matrix=arg_20))
        pass
    for j in range(1, J+1):
        pass
    
    pass
    return { "sigma_y": sigma_y,"b": b,"d": d,"ag": ag,"mu_ag": mu_ag,"rho_ag": rho_ag,"sigma_t": sigma_t,"rho_yt": rho_yt,"sigma_a": sigma_a,"sigma_g": sigma_g, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(J,siteset,yt,z,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('sigma_y_mean', np.array2string(dist.Uniform(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('b_mean', np.array2string(dist.LogNormal(pyro.param('arg_9'), pyro.param('arg_10')).mean.detach().numpy(), separator=','))
print('d_mean', np.array2string(dist.Beta(pyro.param('arg_7'), pyro.param('arg_8')).mean.detach().numpy(), separator=','))
print('ag_mean', np.array2string(dist.MultivariateNormal(pyro.param('arg_19'), pyro.param('arg_20')).mean.detach().numpy(), separator=','))
print('mu_ag_mean', np.array2string(dist.Normal(pyro.param('arg_17'), pyro.param('arg_18')).mean.detach().numpy(), separator=','))
print('rho_ag_mean', np.array2string(dist.Uniform(pyro.param('arg_15'), pyro.param('arg_16')).mean.detach().numpy(), separator=','))
print('sigma_t_mean', np.array2string(dist.Uniform(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('rho_yt_mean', np.array2string(dist.Uniform(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('sigma_a_mean', np.array2string(dist.Uniform(pyro.param('arg_11'), pyro.param('arg_12')).mean.detach().numpy(), separator=','))
print('sigma_g_mean', np.array2string(dist.Uniform(pyro.param('arg_13'), pyro.param('arg_14')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('sigma_y:')
    samplefile.write(np.array2string(np.array([guide(J,siteset,yt,z,N)['sigma_y'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('b:')
    samplefile.write(np.array2string(np.array([guide(J,siteset,yt,z,N)['b'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('d:')
    samplefile.write(np.array2string(np.array([guide(J,siteset,yt,z,N)['d'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('ag:')
    samplefile.write(np.array2string(np.array([guide(J,siteset,yt,z,N)['ag'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('mu_ag:')
    samplefile.write(np.array2string(np.array([guide(J,siteset,yt,z,N)['mu_ag'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('rho_ag:')
    samplefile.write(np.array2string(np.array([guide(J,siteset,yt,z,N)['rho_ag'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_t:')
    samplefile.write(np.array2string(np.array([guide(J,siteset,yt,z,N)['sigma_t'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('rho_yt:')
    samplefile.write(np.array2string(np.array([guide(J,siteset,yt,z,N)['rho_yt'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_a:')
    samplefile.write(np.array2string(np.array([guide(J,siteset,yt,z,N)['sigma_a'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_g:')
    samplefile.write(np.array2string(np.array([guide(J,siteset,yt,z,N)['sigma_g'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
