import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
rt= np.array([1, 9, 2, 1, 10, 1, 1, 90], dtype=np.float32).reshape(8,1)
rt=torch.tensor(rt)
nc= np.array([36, 135, 200, 46, 148, 56, 23, 1157], dtype=np.float32).reshape(8,1)
nc=torch.tensor(nc)
nt= np.array([40, 135, 200, 48, 150, 59, 25, 1159], dtype=np.float32).reshape(8,1)
nt=torch.tensor(nt)
N_studies=8
N_studies=torch.tensor(N_studies)
rc= np.array([2, 23, 7, 1, 8, 9, 3, 118], dtype=np.float32).reshape(8,1)
rc=torch.tensor(rc)
def model(rt,nc,nt,N_studies,rc):
    pc = torch.zeros([amb(N_priors),amb(N_studies)])
    theta = torch.zeros([amb(N_priors),amb(N_studies)])
    N_priors = torch.zeros([amb(1)])
    s0_sq = torch.zeros([amb(1)])
    p0_sigma = torch.zeros([amb(1)])
    N_priors=6
    s0_sq=0.1272041
    p0_sigma=1/torch.sqrt(dist.Normal(0,1).cdf(0.75)/s0_sq)
    inv_tau_sq_1 = pyro.sample('inv_tau_sq_1'.format(''), dist.Gamma(torch.tensor(0.001)*torch.ones([amb(1)]),torch.tensor(0.001)*torch.ones([amb(1)])))
    tau_sq_2 = pyro.sample('tau_sq_2'.format(''), dist.Uniform(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(50.0)*torch.ones([amb(1)])))
    tau_3 = pyro.sample('tau_3'.format(''), dist.Uniform(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(50.0)*torch.ones([amb(1)])))
    B0 = pyro.sample('B0'.format(''), dist.Uniform(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    D0 = pyro.sample('D0'.format(''), dist.Uniform(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    tau_sq_6 = pyro.sample('tau_sq_6'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),p0_sigma*torch.ones([amb(1)])))
    tau = torch.zeros([amb(N_priors)])
    tau[1-1]=1/torch.sqrt(inv_tau_sq_1)
    tau[2-1]=torch.sqrt(tau_sq_2)
    tau[3-1]=tau_3
    tau[4-1]=torch.sqrt(s0_sq*(1-B0)/B0)
    tau[5-1]=torch.sqrt(s0_sq)*(1-D0)/D0
    tau[6-1]=torch.sqrt(tau_sq_6)
    with pyro.iarange('mu_range_'.format(''), N_priors):
        mu = pyro.sample('mu'.format(''), dist.Uniform(-10*torch.ones([amb(N_priors)]),torch.tensor(10.0)*torch.ones([amb(N_priors)])))
    for prior in range(1, N_priors+1):
        with pyro.iarange('pc_range_{0}'.format(prior), N_priors,N_studies):
            pc[prior-1] = pyro.sample('pc{0}'.format(prior-1), dist.Uniform(torch.tensor(0.0)*torch.ones([amb(N_studies)]),torch.tensor(1.0)*torch.ones([amb(N_studies)])))
        with pyro.iarange('theta_range_{0}'.format(prior), N_priors,N_studies):
            theta[prior-1] = pyro.sample('theta{0}'.format(prior-1), dist.Normal(mu[prior-1]*torch.ones([amb(N_studies)]),tau[prior-1]*torch.ones([amb(N_studies)])))
    for prior in range(1, N_priors+1):
        tmpm = torch.zeros([amb(N_studies)])
        for i in range(1, N_studies+1):
            tmpm[i-1]=theta[prior-1,i-1]+torch.log(pc[prior-1,i-1]/(1+pc[prior-1,i-1]))
        pyro.sample('obs_{0}_100'.format(prior), dist.Binomial(nc,pc[prior-1]), obs=rc)
        pyro.sample('obs_{0}_101'.format(prior), dist.Binomial(total_count=nt,logits=tmpm), obs=rt)
    tmpm = torch.zeros([amb(N_studies)])
    pyro.sample('obs__102'.format(), dist.Binomial(total_count=nt,logits=tmpm), obs=rt)
    OR = torch.zeros([amb(N_priors)])
    for prior in range(1, N_priors+1):
        OR[prior-1]=torch.exp(mu[prior-1])
    
def guide(rt,nc,nt,N_studies,rc):
    pc = torch.zeros([amb(N_priors),amb(N_studies)])
    theta = torch.zeros([amb(N_priors),amb(N_studies)])
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    inv_tau_sq_1 = pyro.sample('inv_tau_sq_1'.format(''), dist.LogNormal(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))))
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))))
    tau_sq_2 = pyro.sample('tau_sq_2'.format(''), dist.Uniform(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))))
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))))
    tau_3 = pyro.sample('tau_3'.format(''), dist.Uniform(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))))
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))))
    B0 = pyro.sample('B0'.format(''), dist.Uniform(arg_7,arg_8))
    arg_9 = pyro.param('arg_9', torch.ones((amb(1))))
    arg_10 = pyro.param('arg_10', torch.ones((amb(1))))
    D0 = pyro.sample('D0'.format(''), dist.Uniform(arg_9,arg_10))
    arg_11 = pyro.param('arg_11', torch.ones((amb(1))))
    arg_12 = pyro.param('arg_12', torch.ones((amb(1))), constraint=constraints.positive)
    tau_sq_6 = pyro.sample('tau_sq_6'.format(''), dist.Cauchy(arg_11,arg_12))
    arg_13 = pyro.param('arg_13', torch.ones((amb(N_priors))))
    arg_14 = pyro.param('arg_14', torch.ones((amb(N_priors))))
    with pyro.iarange('mu_prange'):
        mu = pyro.sample('mu'.format(''), dist.Uniform(arg_13,arg_14))
    for prior in range(1, N_priors+1):
        arg_15 = pyro.param('arg_15', torch.ones((amb(N_studies))))
        arg_16 = pyro.param('arg_16', torch.ones((amb(N_studies))))
        with pyro.iarange('pc_prange'):
            pc[prior-1] = pyro.sample('pc{0}'.format(prior-1), dist.Uniform(arg_15,arg_16))
        arg_17 = pyro.param('arg_17', torch.ones((amb(N_studies))))
        arg_18 = pyro.param('arg_18', torch.ones((amb(N_studies))), constraint=constraints.positive)
        with pyro.iarange('theta_prange'):
            theta[prior-1] = pyro.sample('theta{0}'.format(prior-1), dist.LogNormal(arg_17,arg_18))
        pass
    for prior in range(1, N_priors+1):
        for i in range(1, N_studies+1):
            pass
        pass
    for prior in range(1, N_priors+1):
        pass
    
    pass
    return { "tau_sq_6": tau_sq_6,"tau_sq_2": tau_sq_2,"tau_3": tau_3,"mu": mu,"pc": pc,"B0": B0,"theta": theta,"inv_tau_sq_1": inv_tau_sq_1,"D0": D0, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(rt,nc,nt,N_studies,rc)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('tau_sq_6_mean', np.array2string(dist.Cauchy(pyro.param('arg_11'), pyro.param('arg_12')).mean.detach().numpy(), separator=','))
print('tau_sq_2_mean', np.array2string(dist.Uniform(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('tau_3_mean', np.array2string(dist.Uniform(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('mu_mean', np.array2string(dist.Uniform(pyro.param('arg_13'), pyro.param('arg_14')).mean.detach().numpy(), separator=','))
print('pc_mean', np.array2string(dist.Uniform(pyro.param('arg_15'), pyro.param('arg_16')).mean.detach().numpy(), separator=','))
print('B0_mean', np.array2string(dist.Uniform(pyro.param('arg_7'), pyro.param('arg_8')).mean.detach().numpy(), separator=','))
print('theta_mean', np.array2string(dist.LogNormal(pyro.param('arg_17'), pyro.param('arg_18')).mean.detach().numpy(), separator=','))
print('inv_tau_sq_1_mean', np.array2string(dist.LogNormal(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('D0_mean', np.array2string(dist.Uniform(pyro.param('arg_9'), pyro.param('arg_10')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('tau_sq_6:')
    samplefile.write(np.array2string(np.array([guide(rt,nc,nt,N_studies,rc)['tau_sq_6'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('tau_sq_2:')
    samplefile.write(np.array2string(np.array([guide(rt,nc,nt,N_studies,rc)['tau_sq_2'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('tau_3:')
    samplefile.write(np.array2string(np.array([guide(rt,nc,nt,N_studies,rc)['tau_3'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('mu:')
    samplefile.write(np.array2string(np.array([guide(rt,nc,nt,N_studies,rc)['mu'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('pc:')
    samplefile.write(np.array2string(np.array([guide(rt,nc,nt,N_studies,rc)['pc'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('B0:')
    samplefile.write(np.array2string(np.array([guide(rt,nc,nt,N_studies,rc)['B0'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('theta:')
    samplefile.write(np.array2string(np.array([guide(rt,nc,nt,N_studies,rc)['theta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('inv_tau_sq_1:')
    samplefile.write(np.array2string(np.array([guide(rt,nc,nt,N_studies,rc)['inv_tau_sq_1'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('D0:')
    samplefile.write(np.array2string(np.array([guide(rt,nc,nt,N_studies,rc)['D0'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
