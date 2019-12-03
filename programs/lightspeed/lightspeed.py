import pyro, numpy as np, torch, pyro.distributions as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
y= np.array([28.0, 26.0, 33.0, 24.0, 34.0, -44.0, 27.0, 16.0, 40.0, -2.0, 29.0, 22.0, 24.0, 21.0, 25.0, 30.0, 23.0, 29.0, 31.0, 19.0, 24.0, 20.0, 36.0, 32.0, 36.0, 28.0, 25.0, 21.0, 28.0, 29.0, 37.0, 25.0, 28.0, 26.0, 30.0, 32.0, 36.0, 26.0, 30.0, 22.0, 36.0, 23.0, 27.0, 27.0, 28.0, 27.0, 31.0, 27.0, 26.0, 33.0, 26.0, 32.0, 32.0, 24.0, 39.0, 28.0, 24.0, 25.0, 32.0, 25.0, 29.0, 27.0, 28.0, 29.0, 16.0, 23.0], dtype=np.float32).reshape(66,1)
y=torch.Tensor(y)
def model(y):
    with pyro.iarange('beta1_range', 1):
    	beta1 = pyro.sample('beta1', dist.Normal(torch.tensor(0.0),torch.tensor(1.0)).expand([1]))
    sigma = pyro.sample('sigma', dist.Normal(torch.tensor(0.0),torch.tensor(1.0)))
    pyro.sample('obs_'.format(), dist.Normal(beta1[0],sigma), obs=y)
    
def guide(y):
    arg_1 = pyro.param('arg_1', torch.ones((1)), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((1)), constraint=constraints.positive)
    with pyro.iarange('beta1'):
    	beta1 = pyro.sample('beta1', dist.Normal(arg_1,arg_2).expand([1]))
    arg_3 = pyro.param('arg_3', torch.ones((1)), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((1)), constraint=constraints.positive)
    sigma = pyro.sample('sigma', dist.Normal(arg_3,arg_4))
    
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO())
for i in range(4000):
	loss = svi.step(y)
	if ((i % 1000) == 0):
		print(loss)
for name in pyro.get_*param_store().get_all_param_names():
	print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('beta1_mean', np.array2string(dist.Normal(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('sigma_mean', np.array2string(dist.Normal(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
with open('pyro_out', 'w') as outputfile:
    outputfile.write('beta1, normal, {0}, {1}\n'.format( pyro.param('arg_1').detach().numpy(), pyro.param('arg_2').detach().numpy()))
    outputfile.write('sigma, normal, {0}, {1}\n'.format(pyro.param('arg_3').detach().numpy(), pyro.param('arg_4').detach().numpy()))

