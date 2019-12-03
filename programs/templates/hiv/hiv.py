import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
y= np.array([4.2426407, 6.0827625, 3.6055513, 3.6055513, 3.4641016, 5.4772256, 5.2915026, 5.4772256, 5.5677644, 5.0990195, 5.3851648, 5.9160798, 6.0, 6.4031242, 5.9160798, 5.4772256, 5.1961524, 4.7958315, 5.4772256, 5.4772256, 4.5825757, 4.6904158, 5.0990195, 5.0, 4.8989795, 3.8729833, 1.7320508, 1.7320508, 1.4142136, 6.4807407, 5.4772256, 5.3851648, 5.6568542, 5.7445626, 5.9160798, 4.472136, 4.6904158, 4.2426407, 4.5825757, 6.78233, 1.7320508, 5.4772256, 5.6568542, 2.6457513, 6.0, 4.7853944, 3.7416574, 4.6904158, 4.472136, 3.7416574, 5.0990195, 4.3588989, 4.472136, 4.8989795, 4.5825757, 4.8989795, 5.0, 6.4031242, 6.4807407, 6.0, 5.9160798, 6.0827625, 4.1593269, 4.7644517, 4.2895221, 5.00999, 4.7853944, 4.6797436, 4.8682646, 2.5495098, 2.2803509, 2.3664319, 4.4158804, 3.6878178, 3.5213634, 3.4928498, 2.3664319, 1.7029386, 5.5045436, 4.8270074, 5.4863467, 4.7222876, 4.9598387, 5.0892043, 5.2440442, 4.2071368, 4.4045431, 3.8470768, 3.4785054, 4.0620192, 4.2071368, 4.1713307, 4.1713307, 3.4785054, 4.0373258, 3.5071356, 3.591657, 1.9748418, 2.32379, 2.9325757, 3.2249031, 2.1213203, 1.2649111, 0.9486833, 2.3874673, 2.32379, 2.236068, 1.4832397, 1.6431677, 1.3784049, 2.6267851, 2.4698178, 1.6733201, 1.2247449, 1.6733201, 0.8944272, 1.0, 5.0695167, 5.2820451, 5.0398413, 5.3103672, 5.186521, 4.7958315, 5.3851648, 1.2247449, 0.7745967, 0.3162278, 3.4351128, 0.4472136, 3.3316662, 3.8078866, 2.6645825, 2.2135944, 3.2557641, 3.0, 1.4491377, 1.8708287, 1.5165751, 2.1908902, 6.3796552, 6.7527772, 6.4187226, 6.2369865, 5.3665631, 4.7328638, 6.6407831, 4.7116876, 5.9160798, 3.6055513, 7.5630682, 7.6157731, 4.8579831, 5.2345009, 5.5045436, 4.6904158, 3.8600518, 1.6124515, 1.3416408, 1.2247449, 1.3784049, 3.3166248, 6.0, 4.5825757, 2.6457513, 2.8284271, 4.8989795, 6.1806149, 5.138093, 5.6035703, 5.4680892, 4.2778499, 3.7148351, 6.1562976, 5.4313902, 6.7453688, 5.5587768, 3.8209946, 4.3588989, 3.1937439, 4.7328638, 4.6797436, 4.8887626, 4.8476799, 4.2308392, 4.5607017, 4.6797436, 4.5825757, 4.7539457, 4.4045431, 3.1622777, 3.4496377, 1.5165751, 2.2135944, 6.6858059, 6.7601775, 6.6407831, 6.5421709, 6.2128898, 6.6483081, 0.8944272, 0.9486833, 0.9486833, 0.6324555, 0.4472136, 3.5777088, 4.5934736, 3.5637059, 4.4271887, 5.3103672, 3.7682887, 4.2778499, 4.5166359, 5.4772256, 3.8729833, 5.8309519, 7.2801099, 4.7958315, 4.7958315, 5.2345009, 5.8309519, 7.3484692, 6.5574385, 7.4833148, 5.6568542, 5.5677644, 6.164414, 7.4833148, 4.7958315, 6.0, 6.78233, 6.9282032, 4.1231056, 3.7682887, 3.8987177, 5.2915026, 5.2915026, 2.8284271, 1.4142136, 1.0, 1.5491933, 0.83666, 7.2801099, 5.0990195, 5.0, 6.0166436, 5.8566202, 5.2820451, 5.8991525, 6.3245553, 5.1961524, 5.1961524, 4.3588989, 5.2249402, 5.5767374, 5.5677644, 5.2153619, 2.9664794, 4.669047, 7.2732386, 5.7008771, 5.0, 5.1088159, 5.6568542, 5.2249402, 4.7539457, 5.5946403, 5.1961524, 6.6558245, 4.9799598, 6.5878676, 6.97137, 6.041523, 6.5574385, 6.164414, 6.7082039, 5.9160798, 5.0990195, 4.8989795, 7.2111026, 7.8102497, 6.8556546, 7.2111026, 5.6568542, 4.8989795, 3.6742346, 1.7320508, 6.7082039, 3.3316662, 3.9115214, 3.4641016, 3.1622777, 3.8729833, 5.9160798, 5.2915026, 5.8309519, 6.164414, 5.0990195, 4.472136, 4.2426407, 4.1231056, 4.8989795, 5.2630789, 5.3572381, 5.4037024, 4.6797436, 4.9193496, 5.3009433, 5.1575188, 4.7644517, 5.3103672, 3.0, 3.9370039, 1.5491933, 0.7745967, 0.6324555, 3.1937439, 3.3316662, 3.5213634, 2.9154759, 3.8600518, 7.7459667, 7.4833148, 7.6157731, 2.8284271, 6.3245553, 5.9833101, 5.8906706, 5.8309519, 3.4641016, 3.2863353, 3.1304952, 2.0493902, 6.6332496, 6.7082039, 6.340347, 6.0580525, 5.9916609, 6.3718129, 5.3478968, 5.4772256, 4.472136, 5.00999, 5.4037024, 4.9396356, 4.3703547, 4.3817805, 4.0124805, 4.1593269, 4.6368092, 6.0083276, 5.6124861, 5.0695167, 5.2820451, 4.7958315, 7.0710678, 5.0990195, 4.9699095, 7.2111026, 4.5825757, 4.7958315, 4.2426407, 6.164414, 3.9370039, 5.2820451, 4.7328638, 5.5946403, 5.7445626, 5.8309519, 3.7416574], dtype=np.float32).reshape(369,1)
y=torch.tensor(y)
person= np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 9, 9, 10, 10, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 13, 13, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 16, 16, 16, 17, 17, 17, 17, 17, 17, 18, 18, 18, 18, 18, 18, 18, 19, 19, 19, 19, 19, 19, 19, 20, 20, 20, 20, 20, 21, 21, 21, 21, 21, 21, 21, 22, 22, 22, 22, 22, 22, 23, 23, 23, 23, 23, 23, 23, 24, 24, 24, 24, 24, 24, 24, 25, 25, 25, 25, 25, 27, 27, 27, 27, 27, 28, 28, 28, 28, 28, 29, 29, 29, 29, 29, 30, 30, 30, 30, 30, 30, 30, 31, 31, 31, 31, 31, 32, 32, 32, 32, 33, 33, 33, 34, 34, 35, 36, 36, 36, 36, 37, 37, 38, 38, 38, 38, 38, 38, 38, 39, 39, 39, 39, 39, 39, 39, 40, 40, 40, 40, 40, 41, 41, 42, 42, 42, 42, 42, 42, 43, 43, 43, 43, 43, 44, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46, 46, 46, 46, 46, 47, 47, 47, 47, 47, 47, 47, 48, 49, 49, 49, 49, 49, 49, 49, 50, 50, 50, 50, 50, 50, 50, 51, 51, 51, 51, 51, 51, 51, 52, 52, 52, 52, 52, 52, 53, 53, 53, 53, 53, 53, 54, 54, 54, 55, 55, 55, 55, 55, 56, 56, 56, 57, 57, 58, 58, 58, 59, 59, 60, 60, 60, 60, 61, 61, 62, 62, 63, 64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 66, 66, 66, 66, 67, 67, 67, 68, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69, 70, 70, 70, 70, 70, 71, 71, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72, 73, 73, 73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 75, 75, 75, 75, 76, 76, 76, 77, 77, 77, 78, 78, 78, 79, 79, 80, 80, 80, 80, 81, 81, 82, 83, 83, 84], dtype=np.int64).reshape(369,1)
person=torch.tensor(person)
J=84
J=torch.tensor(J)
time= np.array([0.0, 0.5583333, 0.7883333, 1.4208333, 1.9383333, 0.0, 0.23, 0.4791667, 0.7283333, 0.9583333, 1.1883333, 0.23, 0.46, 0.7091667, 0.9583333, 1.1883333, 0.0, 0.5941667, 0.8241667, 0.0, 0.2491667, 0.69, 0.92, 1.15, 1.3716667, 0.0, 0.4983333, 0.7283333, 0.9608333, 0.0, 0.4983333, 0.7616667, 1.0108333, 1.265, 1.495, 0.0, 0.2133333, 0.4291667, 0.6733333, 0.9033333, 1.1358333, 0.0, 0.4791667, 0.0, 0.7008333, 0.0, 0.6516667, 1.0925, 1.35, 1.5908333, 0.0, 0.2408333, 0.46, 0.7008333, 0.9333333, 1.2016667, 1.47, 0.0, 0.2575, 0.0, 0.515, 0.7258333, 0.0, 0.2575, 0.5041667, 0.7258333, 0.9641667, 1.1966667, 1.4975, 0.0, 0.2433333, 0.4733333, 0.0, 0.4791667, 0.7116667, 0.9416667, 1.1716667, 1.4016667, 0.0, 0.2491667, 0.5008333, 0.725, 0.955, 1.2041667, 1.4258333, 0.0, 0.2458333, 0.4758333, 0.7058333, 0.9358333, 1.1775, 1.415, 0.0, 0.2541667, 0.4816667, 0.725, 0.9766667, 0.0, 0.2275, 0.4575, 0.685, 0.9175, 1.1475, 1.3583333, 0.0, 0.2358333, 0.4658333, 0.715, 0.9391667, 1.3908333, 0.0, 0.2491667, 0.4875, 0.7091667, 0.9391667, 1.1691667, 1.3991667, 0.0, 0.2383333, 0.465, 0.6925, 0.9416667, 1.1575, 1.3875, 0.0, 0.2491667, 0.4708333, 0.7008333, 0.9308333, 0.0, 0.23, 0.5175, 0.805, 1.0925, 0.0, 0.46, 0.69, 0.9391667, 1.1883333, 0.0, 0.23, 0.46, 0.69, 1.3416667, 0.0, 0.23, 0.46, 0.69, 0.92, 1.1883333, 1.4183333, 0.0, 0.2491667, 0.4791667, 0.7475, 0.9966667, 0.0, 0.2491667, 0.46, 0.69, 0.0, 0.9391667, 1.47, 0.0, 0.23, 0.0, 0.0, 0.2566667, 0.5141667, 1.2258333, 0.0, 0.2491667, 0.0, 0.2166667, 0.4816667, 0.7475, 0.9966667, 1.2316667, 1.4925, 0.0, 0.23, 0.4983333, 0.7633333, 1.0125, 1.2616667, 1.5, 0.0, 0.3583333, 0.7583333, 1.3991667, 1.9083333, 0.0, 0.3316667, 0.0, 0.2458333, 0.495, 0.7441667, 0.9825, 1.4725, 0.0, 0.2491667, 0.4791667, 0.7283333, 1.1883333, 0.0, 0.3066667, 0.5366667, 1.3991667, 0.0, 0.4983333, 1.035, 1.265, 0.0, 0.2358333, 0.4766667, 0.7066667, 0.9775, 1.2325, 1.4625, 0.0, 0.2266667, 0.4733333, 0.7091667, 0.9525, 1.2016667, 1.4566667, 0.0, 0.0, 0.2516667, 0.485, 0.7366667, 1.0025, 1.2483333, 1.4816667, 0.0, 0.2358333, 0.4625, 0.7341667, 0.9775, 1.2075, 1.4458333, 0.0, 0.2241667, 0.4758333, 0.7058333, 0.9358333, 1.1766667, 1.4125, 0.0, 0.2441667, 0.5008333, 0.7475, 0.9775, 1.2291667, 0.0, 0.2516667, 0.4841667, 0.7175, 0.9475, 1.1883333, 0.0, 0.23, 0.46, 0.0, 0.2458333, 0.4816667, 0.7116667, 0.955, 0.0, 0.23, 0.4766667, 0.0, 0.2325, 0.0, 0.2875, 0.8625, 0.0, 0.2491667, 0.0, 0.2491667, 0.5175, 0.7666667, 0.78, 1.0866667, 0.0, 0.6708333, 0.0, 0.0, 0.46, 0.7283333, 0.9775, 1.1883333, 0.0, 0.2491667, 0.4791667, 0.7091667, 1.2458333, 0.0, 0.2491667, 0.9008333, 1.1308333, 0.0, 0.2216667, 0.4566667, 0.0, 0.3641667, 0.7091667, 0.9583333, 1.1883333, 1.4241667, 0.0, 0.2266667, 0.9358333, 1.1658333, 1.3958333, 0.0, 0.4983333, 0.7666667, 1.0541667, 1.5333333, 0.0, 0.2516667, 0.4791667, 0.72, 0.9391667, 1.1716667, 1.41, 0.0, 0.23, 0.46, 0.6983333, 1.1583333, 0.0, 0.23, 0.46, 0.69, 0.92, 1.1525, 1.38, 0.0, 0.23, 0.46, 0.7033333, 1.1608333, 0.0, 0.23, 0.4741667, 0.6708333, 0.0, 0.2566667, 0.6841667, 0.0, 0.2358333, 0.6958333, 0.0, 0.23, 1.2041667, 0.0, 0.2325, 0.0, 0.23, 0.4875, 0.7308333, 0.0, 0.2491667, 0.0, 0.0, 0.2458333, 0.0], dtype=np.float32).reshape(369,1)
time=torch.tensor(time)
N=369
N=torch.tensor(N)
def model(y,person,J,time,N):
    sigma_y = pyro.sample('sigma_y'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    sigma_a2 = pyro.sample('sigma_a2'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    sigma_a1 = pyro.sample('sigma_a1'.format(''), dist.Normal(torch.tensor(1234.0)*torch.ones([amb(1)]),torch.tensor(1234.0)*torch.ones([amb(1)])))
    mu_a1 = pyro.sample('mu_a1'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    mu_a2 = pyro.sample('mu_a2'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1.0)*torch.ones([amb(1)])))
    with pyro.iarange('a1_range_'.format('')):
        a1 = pyro.sample('a1'.format(''), dist.Normal(mu_a1*torch.ones([amb(J)]),sigma_a1*torch.ones([amb(J)])))
    with pyro.iarange('a2_range_'.format('')):
        a2 = pyro.sample('a2'.format(''), dist.Normal(0.1*mu_a2*torch.ones([amb(J)]),sigma_a2*torch.ones([amb(J)])))
    y_hat = torch.zeros([amb(N)])
    for i in range(1, N+1):
        y_hat[i-1]=a1[person[i-1]-1]+a2[person[i-1]-1]*time[i-1]
    pyro.sample('obs__100'.format(), dist.Normal(y_hat,sigma_y), obs=y)
    
def guide(y,person,J,time,N):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    sigma_y = pyro.sample('sigma_y'.format(''), dist.Pareto(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))))
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    sigma_a2 = pyro.sample('sigma_a2'.format(''), dist.Cauchy(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))), constraint=constraints.positive)
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))))
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))), constraint=constraints.positive)
    sigma_a1 = pyro.sample('sigma_a1'.format(''), dist.StudentT(df=arg_5,loc=arg_6,scale=arg_7))
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))), constraint=constraints.positive)
    arg_9 = pyro.param('arg_9', torch.ones((amb(1))), constraint=constraints.positive)
    mu_a1 = pyro.sample('mu_a1'.format(''), dist.Beta(arg_8,arg_9))
    arg_10 = pyro.param('arg_10', torch.ones((amb(1))))
    arg_11 = pyro.param('arg_11', torch.ones((amb(1))), constraint=constraints.positive)
    mu_a2 = pyro.sample('mu_a2'.format(''), dist.LogNormal(arg_10,arg_11))
    arg_12 = pyro.param('arg_12', torch.ones((amb(J))), constraint=constraints.positive)
    arg_13 = pyro.param('arg_13', torch.ones((amb(J))))
    arg_14 = pyro.param('arg_14', torch.ones((amb(J))), constraint=constraints.positive)
    with pyro.iarange('a1_prange'):
        a1 = pyro.sample('a1'.format(''), dist.StudentT(df=arg_12,loc=arg_13,scale=arg_14))
    arg_15 = pyro.param('arg_15', torch.ones((amb(J))), constraint=constraints.positive)
    arg_16 = pyro.param('arg_16', torch.ones((amb(J))), constraint=constraints.positive)
    with pyro.iarange('a2_prange'):
        a2 = pyro.sample('a2'.format(''), dist.Gamma(arg_15,arg_16))
    for i in range(1, N+1):
        pass
    
    pass
    return { "sigma_y": sigma_y,"sigma_a2": sigma_a2,"sigma_a1": sigma_a1,"a1": a1,"a2": a2,"mu_a2": mu_a2,"mu_a1": mu_a1, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(y,person,J,time,N)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('sigma_y_mean', np.array2string(dist.Pareto(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('sigma_a2_mean', np.array2string(dist.Cauchy(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('sigma_a1_mean', np.array2string(dist.StudentT(pyro.param('arg_5')).mean.detach().numpy(), separator=','))
print('a1_mean', np.array2string(dist.StudentT(pyro.param('arg_12')).mean.detach().numpy(), separator=','))
print('a2_mean', np.array2string(dist.Gamma(pyro.param('arg_15'), pyro.param('arg_16')).mean.detach().numpy(), separator=','))
print('mu_a2_mean', np.array2string(dist.LogNormal(pyro.param('arg_10'), pyro.param('arg_11')).mean.detach().numpy(), separator=','))
print('mu_a1_mean', np.array2string(dist.Beta(pyro.param('arg_8'), pyro.param('arg_9')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('sigma_y:')
    samplefile.write(np.array2string(np.array([guide(y,person,J,time,N)['sigma_y'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_a2:')
    samplefile.write(np.array2string(np.array([guide(y,person,J,time,N)['sigma_a2'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_a1:')
    samplefile.write(np.array2string(np.array([guide(y,person,J,time,N)['sigma_a1'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('a1:')
    samplefile.write(np.array2string(np.array([guide(y,person,J,time,N)['a1'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('a2:')
    samplefile.write(np.array2string(np.array([guide(y,person,J,time,N)['a2'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('mu_a2:')
    samplefile.write(np.array2string(np.array([guide(y,person,J,time,N)['mu_a2'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('mu_a1:')
    samplefile.write(np.array2string(np.array([guide(y,person,J,time,N)['mu_a1'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
