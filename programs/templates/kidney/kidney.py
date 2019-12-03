import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
N_rc=18
N_rc=torch.tensor(N_rc)
sex_uc= np.array([0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32).reshape(58,1)
sex_uc=torch.tensor(sex_uc)
disease_uc= np.array([1, 2, 1, 1, 1, 1, 2, 2, 3, 2, 3, 1, 3, 1, 3, 1, 1, 1, 4, 1, 3, 3, 3, 2, 3, 2, 2, 3, 4, 2, 1, 4, 1, 1, 1, 1, 1, 2, 2, 3, 2, 3, 3, 1, 1, 4, 3, 3, 3, 2, 3, 2, 2, 3, 3, 4, 1, 4], dtype=np.int64).reshape(58,1)
disease_uc=torch.tensor(disease_uc)
sex_rc= np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0], dtype=np.float32).reshape(18,1)
sex_rc=torch.tensor(sex_rc)
age_uc= np.array([28, 48, 32, 31, 10, 16, 51, 55, 69, 51, 44, 34, 35, 17, 60, 60, 43, 44, 46, 30, 62, 42, 43, 10, 52, 53, 54, 56, 57, 44, 22, 60, 28, 32, 32, 10, 17, 51, 56, 69, 52, 44, 35, 60, 44, 47, 63, 43, 58, 10, 52, 53, 54, 56, 51, 57, 22, 52], dtype=np.float32).reshape(58,1)
age_uc=torch.tensor(age_uc)
t_uc= np.array([8.0, 23.0, 22.0, 447.0, 30.0, 24.0, 7.0, 511.0, 53.0, 15.0, 7.0, 141.0, 96.0, 536.0, 17.0, 185.0, 292.0, 15.0, 152.0, 402.0, 13.0, 39.0, 12.0, 132.0, 34.0, 2.0, 130.0, 27.0, 152.0, 190.0, 119.0, 63.0, 16.0, 28.0, 318.0, 12.0, 245.0, 9.0, 30.0, 196.0, 154.0, 333.0, 38.0, 177.0, 114.0, 562.0, 66.0, 40.0, 201.0, 156.0, 30.0, 25.0, 26.0, 58.0, 43.0, 30.0, 8.0, 78.0], dtype=np.float32).reshape(58,1)
t_uc=torch.tensor(t_uc)
t_rc= np.array([149.0, 22.0, 113.0, 5.0, 54.0, 6.0, 13.0, 8.0, 70.0, 25.0, 4.0, 159.0, 108.0, 24.0, 46.0, 5.0, 16.0, 8.0], dtype=np.float32).reshape(18,1)
t_rc=torch.tensor(t_rc)
patient_rc= np.array([14, 19, 26, 32, 36, 37, 2, 12, 14, 15, 16, 19, 20, 22, 24, 34, 36, 38], dtype=np.int64).reshape(18,1)
patient_rc=torch.tensor(patient_rc)
NP=38
NP=torch.tensor(NP)
N_uc=58
N_uc=torch.tensor(N_uc)
disease_rc= np.array([3, 2, 3, 3, 1, 4, 2, 1, 3, 1, 3, 2, 1, 1, 3, 2, 1, 4], dtype=np.int64).reshape(18,1)
disease_rc=torch.tensor(disease_rc)
patient_uc= np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 33, 34, 35, 38, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 17, 18, 21, 23, 25, 26, 27, 28, 29, 30, 31, 32, 33, 35, 37], dtype=np.int64).reshape(58,1)
patient_uc=torch.tensor(patient_uc)
age_rc= np.array([42, 53, 57, 50, 42, 52, 48, 34, 42, 17, 60, 53, 44, 30, 43, 45, 42, 60], dtype=np.float32).reshape(18,1)
age_rc=torch.tensor(age_rc)
def model(N_rc,sex_uc,disease_uc,sex_rc,age_uc,t_uc,t_rc,patient_rc,NP,N_uc,disease_rc,patient_uc,age_rc):
    b = torch.zeros([amb(NP)])
    alpha = pyro.sample('alpha'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    beta_age = pyro.sample('beta_age'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    beta_sex = pyro.sample('beta_sex'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    beta_disease2 = pyro.sample('beta_disease2'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    beta_disease3 = pyro.sample('beta_disease3'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    beta_disease4 = pyro.sample('beta_disease4'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(100.0)*torch.ones([amb(1)])))
    tau = pyro.sample('tau'.format(''), dist.Gamma(torch.tensor(0.001)*torch.ones([amb(1)]),torch.tensor(0.001)*torch.ones([amb(1)])))
    sigma = torch.zeros([amb(1)])
    yabeta_disease = torch.zeros([amb(4)])
    yabeta_disease[1-1]=0
    yabeta_disease[2-1]=beta_disease2
    yabeta_disease[3-1]=beta_disease3
    yabeta_disease[4-1]=beta_disease4
    sigma=torch.sqrt(1/tau)
    r = pyro.sample('r'.format(''), dist.Gamma(torch.tensor(1.0)*torch.ones([amb(1)]),torch.tensor(0.001)*torch.ones([amb(1)])))
    for i in range(1, NP+1):
        with pyro.iarange('b_range_{0}'.format(i), NP):
            b[i-1] = pyro.sample('b{0}'.format(i-1), dist.Normal(torch.tensor(0.0)*torch.ones([]),sigma*torch.ones([])))
    for i in range(1, N_uc+1):
        pyro.sample('obs_{0}_100'.format(i), dist.Weibull(r,torch.exp(-(alpha+beta_age*age_uc[i-1]+beta_sex*sex_uc[i-1]+yabeta_disease[disease_uc[i-1]-1]+b[patient_uc[i-1]-1])/r)), obs=t_uc[i-1])
    for i in range(1, N_rc+1):
        pyro.sample('obs_{0}_101'.format(i), dist.Bernoulli(torch.exp(-torch.pow(t_rc[i-1]/torch.exp(-(alpha+beta_age*age_rc[i-1]+beta_sex*sex_rc[i-1]+yabeta_disease[disease_rc[i-1]-1]+b[patient_rc[i-1]-1])/r), r))), obs=1)
    
def guide(N_rc,sex_uc,disease_uc,sex_rc,age_uc,t_uc,t_rc,patient_rc,NP,N_uc,disease_rc,patient_uc,age_rc):
    b = torch.zeros([amb(NP)])
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    alpha = pyro.sample('alpha'.format(''), dist.Weibull(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    beta_age = pyro.sample('beta_age'.format(''), dist.Pareto(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))))
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))), constraint=constraints.positive)
    beta_sex = pyro.sample('beta_sex'.format(''), dist.Cauchy(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))), constraint=constraints.positive)
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))))
    arg_9 = pyro.param('arg_9', torch.ones((amb(1))), constraint=constraints.positive)
    beta_disease2 = pyro.sample('beta_disease2'.format(''), dist.StudentT(df=arg_7,loc=arg_8,scale=arg_9))
    arg_10 = pyro.param('arg_10', torch.ones((amb(1))), constraint=constraints.positive)
    arg_11 = pyro.param('arg_11', torch.ones((amb(1))), constraint=constraints.positive)
    beta_disease3 = pyro.sample('beta_disease3'.format(''), dist.Weibull(arg_10,arg_11))
    arg_12 = pyro.param('arg_12', torch.ones((amb(1))), constraint=constraints.positive)
    arg_13 = pyro.param('arg_13', torch.ones((amb(1))), constraint=constraints.positive)
    beta_disease4 = pyro.sample('beta_disease4'.format(''), dist.Gamma(arg_12,arg_13))
    arg_14 = pyro.param('arg_14', torch.ones((amb(1))), constraint=constraints.positive)
    arg_15 = pyro.param('arg_15', torch.ones((amb(1))), constraint=constraints.positive)
    tau = pyro.sample('tau'.format(''), dist.Gamma(arg_14,arg_15))
    arg_16 = pyro.param('arg_16', torch.ones((amb(1))), constraint=constraints.positive)
    arg_17 = pyro.param('arg_17', torch.ones((amb(1))), constraint=constraints.positive)
    r = pyro.sample('r'.format(''), dist.Pareto(arg_16,arg_17))
    for i in range(1, NP+1):
        arg_18 = pyro.param('arg_18', torch.ones(()))
        arg_19 = pyro.param('arg_19', torch.ones(()), constraint=constraints.positive)
        with pyro.iarange('b_prange'):
            b[i-1] = pyro.sample('b{0}'.format(i-1), dist.Normal(arg_18,arg_19))
        pass
    for i in range(1, N_uc+1):
        pass
    for i in range(1, N_rc+1):
        pass
    
    pass
    return { "tau": tau,"beta_age": beta_age,"b": b,"beta_disease4": beta_disease4,"beta_sex": beta_sex,"beta_disease2": beta_disease2,"beta_disease3": beta_disease3,"r": r,"alpha": alpha, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(N_rc,sex_uc,disease_uc,sex_rc,age_uc,t_uc,t_rc,patient_rc,NP,N_uc,disease_rc,patient_uc,age_rc)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('tau_mean', np.array2string(dist.Gamma(pyro.param('arg_14'), pyro.param('arg_15')).mean.detach().numpy(), separator=','))
print('beta_age_mean', np.array2string(dist.Pareto(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('b_mean', np.array2string(dist.Normal(pyro.param('arg_18'), pyro.param('arg_19')).mean.detach().numpy(), separator=','))
print('beta_disease4_mean', np.array2string(dist.Gamma(pyro.param('arg_12'), pyro.param('arg_13')).mean.detach().numpy(), separator=','))
print('beta_sex_mean', np.array2string(dist.Cauchy(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('beta_disease2_mean', np.array2string(dist.StudentT(pyro.param('arg_7')).mean.detach().numpy(), separator=','))
print('beta_disease3_mean', np.array2string(dist.Weibull(pyro.param('arg_10'), pyro.param('arg_11')).mean.detach().numpy(), separator=','))
print('r_mean', np.array2string(dist.Pareto(pyro.param('arg_16'), pyro.param('arg_17')).mean.detach().numpy(), separator=','))
print('alpha_mean', np.array2string(dist.Weibull(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('tau:')
    samplefile.write(np.array2string(np.array([guide(N_rc,sex_uc,disease_uc,sex_rc,age_uc,t_uc,t_rc,patient_rc,NP,N_uc,disease_rc,patient_uc,age_rc)['tau'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta_age:')
    samplefile.write(np.array2string(np.array([guide(N_rc,sex_uc,disease_uc,sex_rc,age_uc,t_uc,t_rc,patient_rc,NP,N_uc,disease_rc,patient_uc,age_rc)['beta_age'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('b:')
    samplefile.write(np.array2string(np.array([guide(N_rc,sex_uc,disease_uc,sex_rc,age_uc,t_uc,t_rc,patient_rc,NP,N_uc,disease_rc,patient_uc,age_rc)['b'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta_disease4:')
    samplefile.write(np.array2string(np.array([guide(N_rc,sex_uc,disease_uc,sex_rc,age_uc,t_uc,t_rc,patient_rc,NP,N_uc,disease_rc,patient_uc,age_rc)['beta_disease4'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta_sex:')
    samplefile.write(np.array2string(np.array([guide(N_rc,sex_uc,disease_uc,sex_rc,age_uc,t_uc,t_rc,patient_rc,NP,N_uc,disease_rc,patient_uc,age_rc)['beta_sex'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta_disease2:')
    samplefile.write(np.array2string(np.array([guide(N_rc,sex_uc,disease_uc,sex_rc,age_uc,t_uc,t_rc,patient_rc,NP,N_uc,disease_rc,patient_uc,age_rc)['beta_disease2'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta_disease3:')
    samplefile.write(np.array2string(np.array([guide(N_rc,sex_uc,disease_uc,sex_rc,age_uc,t_uc,t_rc,patient_rc,NP,N_uc,disease_rc,patient_uc,age_rc)['beta_disease3'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('r:')
    samplefile.write(np.array2string(np.array([guide(N_rc,sex_uc,disease_uc,sex_rc,age_uc,t_uc,t_rc,patient_rc,NP,N_uc,disease_rc,patient_uc,age_rc)['r'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('alpha:')
    samplefile.write(np.array2string(np.array([guide(N_rc,sex_uc,disease_uc,sex_rc,age_uc,t_uc,t_rc,patient_rc,NP,N_uc,disease_rc,patient_uc,age_rc)['alpha'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
