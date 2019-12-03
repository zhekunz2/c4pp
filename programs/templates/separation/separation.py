import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
y= np.array([0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32).reshape(60,1)
y=torch.tensor(y)
x= np.array([-0.33924922708, 2.16776116177, 0.583946420212, -1.25010955939, -2.51903611689, 1.81544864388, -0.679428295956, -0.332381711251, 3.85474147774, 1.75146143243, 3.90671349584, 3.96823748614, 3.39536157511, 2.46322159108, 2.78883671383, 2.69848902326, 0.707973116235, 0.726405148003, 1.0444149724, 0.342009378857, -0.790683750144, 3.32954116489, 1.38079159128, 3.98181472599, 2.24877214672, 3.64920054529, 2.10202334115, 2.62895859353, -1.58423510539, 2.04590428929, 1.90034431094, -3.39115009146, 2.38640446127, -1.43181905943, 3.15308468479, -1.67968373162, -0.317596527305, -0.961773428635, -0.0056978758987, 2.24223051881, 4.00897366227, -0.247837875244, -2.01586022574, 2.3712751154, 0.804244899377, 0.204942438629, 4.71913059489, -3.18550290262, 0.0371760285379, 1.63050107483, 1.58176660124, 0.308978975631, 1.14906114364, -2.19539484324, 1.38753269146, 0.447612648141, 1.24317280127, 2.55202204617, -1.73857703162, -1.85137706115], dtype=np.float32).reshape(60,1)
x=torch.tensor(x)
N=60
N=torch.tensor(N)
def model(y,x,N):
    with pyro.iarange('beta_range_'.format('')):
        beta = pyro.sample('beta'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(2)]),torch.tensor(1234.0)*torch.ones([amb(2)])))
    pyro.sample('obs__100'.format(), dist.Bernoulli(logits=beta[0-1]+beta[2-1]*x), obs=y)
    
def guide(y,x,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(2))))
    arg_2 = pyro.param('arg_2', torch.ones((amb(2))), constraint=constraints.positive)
    with pyro.iarange('beta_prange'):
        beta = pyro.sample('beta'.format(''), dist.Cauchy(arg_1,arg_2))
    
    pass
    return { "beta": beta, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(y,x,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('beta_mean', np.array2string(dist.Cauchy(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(y,x,N)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
