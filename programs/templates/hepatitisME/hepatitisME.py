import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
tvec1= np.array([6.541, 5.84064165737, 6.60123011873, 5.8636311756, 5.91350300564, 5.91889385427, 5.87493073085, 5.82008293035, 5.91350300564, 5.98896141689, 5.95064255259, 5.75574221359, 5.92157841964, 5.93489419562, 5.91350300564, 5.97126183979, 5.97126183979, 6.59987049921, 5.92157841964, 6.71780469502, 5.92958914339, 5.92157841964, 5.92157841964, 5.92157841964, 5.92157841964, 5.84064165737, 5.89440283426, 5.93489419562, 5.93224518745, 5.96870755999, 5.8805329864, 6.62671774925, 5.87773578178, 6.5903010482, 5.92958914339, 5.92958914339, 5.86078622347, 5.89440283426, 5.95064255259, 5.93224518745, 5.97126183979, 6.57088296234, 5.93753620508, 5.87493073085, 5.89164421183, 6.74170069465, 5.89440283426, 5.94803498918, 5.77455154554, 5.97126183979, 5.96870755999, 5.8805329864, 5.89440283426, 6.62936325344, 5.93224518745, 5.99146454711, 5.88887795833, 5.93489419562, 5.94017125272, 5.95583736946, 5.93489419562, 5.92692602597, 6.64639051485, 5.93224518745, 5.96870755999, 5.93224518745, 5.93224518745, 5.93224518745, 5.94542060861, 5.86929691313, 6.65801104587, 5.89715386764, 6.00881318544, 5.97126183979, 5.97126183979, 5.97126183979, 6.00881318544, 6.64639051485, 5.88610403145, 5.75257263883, 5.68017260902, 5.77144112313, 5.96100533962, 5.75257263883, 5.81413053183, 5.63478960317, 5.77455154554, 5.77765232322, 5.63121178182, 5.70711026475, 5.89164421183, 5.77455154554, 6.05912319558, 5.81711115996, 5.65248918027, 5.60947179518, 5.83773044717, 5.91350300564, 5.77455154554, 5.83773044717, 5.98645200528, 5.81711115996, 6.49526555594, 5.68697535634, 5.81711115996, 5.85793315448, 6.963, 6.52941883826, 7.02553831464, 6.51767127291, 6.56948142041, 6.56948142041, 6.97447891103, 6.56244409369, 6.55108033504, 7.03350648429, 6.59987049921, 6.65544035037, 6.7068623366, 6.63987583383, 6.63987583383, 6.55961523749, 6.60123011873, 7.0021559544, 6.68959926918, 6.98286275147, 6.56807791141, 6.71780469502, 6.54821910276, 6.54821910276, 6.54821910276, 6.54821910276, 6.4907235345, 6.7068623366, 6.5610306659, 6.52795791762, 6.56244409369, 6.94889722231, 6.54965074223, 7.00669522684, 6.56807791141, 6.56807791141, 6.55819780281, 6.5903010482, 6.51767127291, 6.701960366, 6.60934924317, 7.01571242049, 6.52062112756, 6.5610306659, 6.5903010482, 6.98933526597, 6.54102999919, 6.55108033504, 6.49828214948, 6.54965074223, 6.58892647753, 6.5903010482, 6.54102999919, 6.98471632012, 6.62007320653, 6.63856778917, 6.59441345975, 6.61873898352, 6.62140565176, 6.62936325344, 6.61873898352, 6.61204103483, 7.04403289727, 6.62007320653, 6.64639051485, 6.64768837356, 6.62007320653, 6.62671774925, 6.62140565176, 6.58755001482, 7.05875815252, 6.59714570189, 6.62936325344, 6.66568371778, 6.66440902035, 6.66440902035, 7.05185562296, 7.04403289727, 6.59167373201, 6.96129604591, 6.50876913697, 6.56244409369, 6.66695679243, 6.49828214948, 7.02108396429, 6.4676987261, 6.48768401848, 6.52209279817, 6.55535689181, 6.45833828334, 6.48920493133, 6.48920493133, 6.62671774925, 6.49223983502, 6.42162226781, 6.4441312567, 6.48616078894, 6.52649485957, 6.45047042214, 6.48463523564, 6.51025834052, 6.48463523564, 6.9421567057, 6.47850964221, 6.51174532964, 6.92264389148, 6.98193467716, 6.9679092018, 7.00850518208, 7.00760061395, 7.00306545879, 7.00940893271, 6.9555926084, 6.98286275147, 6.97821374263, 6.97821374263, 7.00124562207, 7.00033446028, 6.96224346427, 6.98286275147, 6.98193467716, 6.98193467716, 6.98100574072, 6.98100574072, 6.95654544315, 7.00306545879, 7.03350648429, 6.9421567057, 6.97728134163, 6.988413182, 6.98193467716, 6.98193467716, 6.97447891103, 6.98378996526, 6.97447891103, 6.97447891103, 6.97073007814, 6.97447891103, 6.98378996526, 6.98286275147, 6.99668148818, 6.9584483933, 7.01481435128, 6.98378996526, 6.98286275147, 7.03174125876, 7.04053639022, 7.02908756415, 7.04577657688, 7.05272104923, 7.05272104923, 7.04490511713, 7.04141166379, 7.03174125876, 7.02820143206, 7.03174125876, 7.02731451404, 7.05272104923, 7.03174125876, 7.03966034986, 7.03878354139, 7.04403289727, 7.04403289727, 7.02820143206, 6.97541392746, 6.99759598298, 6.99025650049, 6.95081476844, 6.92067150425, 6.91473089272, 6.92165818415, 6.92165818415, 7.01391547481, 6.94119005507, 6.86380339145, 6.92165818415, 6.97447891103, 6.92755790628, 6.91869521902, 6.96413561242, 6.94793706861, 6.93634273583], dtype=np.float32).reshape(288,1)
tvec1=torch.tensor(tvec1)
Yvec1= np.array([4.997, 6.83, 3.95124371858, 6.79794041297, 4.7184988713, 5.7899601709, 6.13122648948, 6.76272950693, 6.14846829592, 10.7579879835, 5.63121178182, 3.4657359028, 7.92479591396, 9.16115012779, 8.74257423767, 5.91079664404, 6.54678541076, 2.94443897917, 7.05272104923, 0.0, 9.13905916997, 8.48260174665, 7.11963563802, 7.64300363556, 9.42294862138, 8.25530881179, 4.82028156561, 6.54678541076, 8.67470962929, 7.86326672401, 4.67282883446, 7.41878088275, 6.84587987526, 5.46383180503, 3.43398720449, 7.75533881285, 9.04239498113, 8.45212119467, 7.91425227874, 7.72885582385, 9.31433996199, 6.35957386867, 2.30258509299, 8.29354951506, 8.90082160492, 8.87905466204, 11.5129154649, 8.79618763547, 8.40155784782, 8.89370997757, 7.1800698743, 4.82028156561, 5.77765232322, 6.47079950378, 4.40671924726, 8.17413934343, 2.30258509299, 9.77104128524, 6.95081476844, 5.9763509093, 5.8805329864, 5.73979291218, 5.75574221359, 5.1532915945, 8.11761074647, 8.54636356872, 7.58171964013, 6.90374725758, 4.66343909411, 7.47986413117, 3.61091791264, 4.64439089914, 4.39444915467, 8.23509549726, 6.12905021006, 8.44246964522, 6.88755257166, 7.29233717617, 6.38856140555, 8.75998249498, 6.08677472691, 4.02535169074, 4.38202663467, 6.85540879861, 8.17919979842, 4.24849524205, 8.98343977178, 8.47219582549, 9.48021482578, 6.14632925767, 9.93542217107, 4.27666611902, 5.36597601502, 4.60517018599, 3.21887582487, 6.81783057145, 6.58479139239, 5.91889385427, 8.67641669696, 2.48490664979, 6.73933662736, 3.91202300543, 4.56434819147, 9.15925758175, 7.07326971746, 9.08636319216, 8.028, 4.905, 4.35670882669, 5.27299955856, 3.49650756147, 3.68887945411, 3.85014760171, 3.295836866, 5.25749537203, 10.4925789215, 4.20469261939, 1.79175946923, 7.54750168281, 8.34236350038, 7.07580886398, 5.11198778836, 4.93447393313, 3.17805383035, 6.25382881158, 7.19067603433, 7.82724090175, 7.5169772246, 5.83481073706, 6.3561076607, 8.57073395834, 7.16394668434, 3.40119738166, 5.95583736946, 7.3238305662, 7.63385355968, 3.36729582999, 6.7615727688, 8.38845031552, 9.921916688, 1.60943791243, 7.44366368312, 6.64768837356, 6.85118492749, 6.16961073249, 9.75214127004, 8.26359043262, 5.84643877506, 0.0, 7.60589000105, 6.25958146406, 8.88391747121, 8.54286093816, 7.86095636488, 6.91075078796, 7.59588991772, 7.03174125876, 2.99573227355, 3.91202300543, 5.45103845357, 5.12989871492, 7.88419993368, 2.30258509299, 9.3289230878, 10.5472081164, 5.71042701737, 6.50128967054, 5.53338948873, 4.99043258678, 3.61091791264, 7.34729970074, 8.03138533063, 6.92264389148, 6.72022015514, 3.43398720449, 6.19236248947, 3.78418963392, 4.51085950652, 3.55534806149, 7.81116338503, 6.19847871649, 3.55534806149, 5.07517381523, 7.25417784646, 6.21460809842, 7.19593722648, 4.66343909411, 3.78418963392, 2.48490664979, 6.08221891038, 6.04500531404, 3.52636052462, 9.69990150044, 7.76004068088, 7.78113850985, 6.28971557091, 9.39806397806, 5.34710753072, 5.88887795833, 5.29831736655, 4.27666611902, 3.4657359028, 7.21523997873, 5.68357976734, 7.72179177682, 1.38629436112, 6.73221070647, 5.49716822529, 2.83321334406, 8.58073121222, 7.1260872733, 7.93594510335, 6.295, 4.31748811354, 4.29045944115, 3.43398720449, 5.37527840768, 3.49650756147, 2.77258872224, 6.40687998607, 7.94838528511, 6.39191711339, 5.37989735354, 5.86929691313, 5.34233425196, 4.45434729625, 6.54247196051, 6.8936563546, 8.79026911148, 7.46278915741, 3.66356164613, 4.99721227376, 8.03040956213, 7.1600692076, 3.17805383035, 6.31535800152, 2.77258872224, 7.82803803213, 7.1115121165, 7.01481435128, 6.00881318544, 9.59703032476, 2.30258509299, 10.1017644761, 7.0656133636, 8.97575663052, 7.08757370556, 7.22256601882, 2.07944154168, 2.7080502011, 3.61091791264, 3.85014760171, 7.22766249873, 0.0, 9.61767040669, 8.66905554073, 5.24174701506, 5.48479693349, 4.04305126783, 6.55250788703, 7.35115822643, 6.52502965784, 5.74939298591, 3.09104245336, 2.99573227355, 3.295836866, 7.54855597917, 4.81218435537, 2.94443897917, 5.03043792139, 2.3978952728, 2.94443897917, 5.45958551414, 3.58351893846, 8.74097653802, 5.87493073085, 9.17294998276, 3.4657359028, 3.36729582999, 3.58351893846, 7.63867982388, 5.90808293817, 5.81711115996, 6.19031540585, 2.56494935746, 4.51085950652, 4.39444915467, 5.0106352941], dtype=np.float32).reshape(288,1)
Yvec1=torch.tensor(Yvec1)
N1=288
N1=torch.tensor(N1)
N=106
N=torch.tensor(N)
idxn1= np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 2, 4, 5, 6, 9, 11, 12, 13, 14, 15, 16, 17, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35, 36, 37, 38, 39, 40, 43, 44, 45, 47, 48, 49, 51, 52, 53, 55, 56, 57, 58, 59, 60, 61, 62, 65, 66, 67, 68, 69, 72, 73, 74, 75, 76, 79, 81, 82, 84, 86, 87, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 101, 102, 105], dtype=np.float32).reshape(288,1)
idxn1=torch.tensor(idxn1)
y0= np.array([8.613, 7.105, 6.896, 5.63835466933, 6.35088571671, 5.03043792139, 5.86646805693, 5.73979291218, 7.07834157956, 8.56655462095, 6.86171134048, 6.97354301952, 7.34213173058, 8.82629423124, 6.48768401848, 6.66440902035, 7.03438792992, 4.49980967033, 6.85118492749, 3.04452243772, 8.63905677917, 1.79175946923, 7.01660968389, 7.09090982208, 7.42356844426, 8.09620827165, 5.70711026475, 7.01031186731, 7.18311170174, 8.15219801586, 5.99645208862, 7.08254856936, 5.06890420222, 7.04228617194, 4.09434456222, 7.32052696227, 7.1268908089, 7.24136628332, 6.62007320653, 7.36897040219, 8.4772041832, 7.20340552108, 2.8903717579, 7.39141523468, 6.25190388317, 8.38160253711, 11.5129154649, 6.60258789219, 7.61283103041, 7.42476176182, 6.71174039506, 5.56068163102, 6.02827852023, 7.10085190894, 7.52671756135, 7.92948652331, 1.60943791243, 6.85856503479, 8.05006542292, 5.83188247728, 7.15851399733, 7.01481435128, 7.70120018086, 7.16239749736, 5.99893656195, 5.4161004022, 8.40782465436, 7.18462915272, 3.25809653802, 8.14380797677, 5.24702407216, 5.73979291218, 4.5538768916, 7.72444664563, 7.91425227874, 7.11476944837, 8.24064886337, 8.56044423341, 6.78558764501, 10.8861838116, 2.07944154168, 5.82894561761, 4.09434456222, 7.53208814354, 5.66988092298, 3.4657359028, 10.722914899, 8.02910705462, 9.23092700558, 6.42971947804, 11.5129154649, 7.25700270709, 6.62007320653, 9.41995278901, 5.28826703069, 8.04654935728, 7.58984151218, 6.39024066707, 6.04025471128, 2.77258872224, 7.62948991639, 7.8547691835, 7.09007683578, 5.50125821054, 6.98100574072, 5.50533153593], dtype=np.float32).reshape(106,1)
y0=torch.tensor(y0)
def model(tvec1,Yvec1,N1,N,idxn1,y0):
    y0_mean = torch.zeros([amb(1)])
    y0_mean=torch.mean(y0)
    oldn = torch.zeros([amb(1)])
    m = torch.zeros([amb(N1)])
    sigmasq_y = pyro.sample('sigmasq_y'.format(''), dist.Gamma(torch.tensor(0.001)*torch.ones([amb(1)]),torch.tensor(0.001)*torch.ones([amb(1)])))
    sigmasq_alpha = pyro.sample('sigmasq_alpha'.format(''), dist.Gamma(torch.tensor(0.001)*torch.ones([amb(1)]),torch.tensor(0.001)*torch.ones([amb(1)])))
    sigmasq_beta = pyro.sample('sigmasq_beta'.format(''), dist.Gamma(torch.tensor(0.001)*torch.ones([amb(1)]),torch.tensor(0.001)*torch.ones([amb(1)])))
    sigma_y = torch.zeros([amb(1)])
    sigma_alpha = torch.zeros([amb(1)])
    sigma_beta = torch.zeros([amb(1)])
    sigma_y=torch.sqrt(sigmasq_y)
    sigma_alpha=torch.sqrt(sigmasq_alpha)
    sigma_beta=torch.sqrt(sigmasq_beta)
    sigma_mu0 = pyro.sample('sigma_mu0'.format(''), dist.Gamma(torch.tensor(0.001)*torch.ones([amb(1)]),torch.tensor(0.001)*torch.ones([amb(1)])))
    alpha0 = pyro.sample('alpha0'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1000.0)*torch.ones([amb(1)])))
    with pyro.iarange('alpha_range_'.format(''), N):
        alpha = pyro.sample('alpha'.format(''), dist.Normal(alpha0*torch.ones([amb(N)]),sigma_alpha*torch.ones([amb(N)])))
    beta0 = pyro.sample('beta0'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1000.0)*torch.ones([amb(1)])))
    with pyro.iarange('beta_range_'.format(''), N):
        beta = pyro.sample('beta'.format(''), dist.Normal(beta0*torch.ones([amb(N)]),sigma_beta*torch.ones([amb(N)])))
    gamma = pyro.sample('gamma'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1000.0)*torch.ones([amb(1)])))
    theta = pyro.sample('theta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(1000.0)*torch.ones([amb(1)])))
    with pyro.iarange('mu0_range_'.format(''), N):
        mu0 = pyro.sample('mu0'.format(''), dist.Normal(theta*torch.ones([amb(N)]),sigma_mu0*torch.ones([amb(N)])))
    for n in range(1, N1+1):
        oldn=idxn1[n-1]
        m[n-1]=alpha[oldn-1]+beta[oldn-1]*(tvec1[n-1]-6.5)+gamma*(mu0[oldn-1]-y0_mean)
    for n in range(1, N+1):
        pyro.sample('obs_{0}_100'.format(n), dist.Normal(mu0[n-1],sigma_y), obs=y0[n-1])
    
def guide(tvec1,Yvec1,N1,N,idxn1,y0):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    sigmasq_y = pyro.sample('sigmasq_y'.format(''), dist.Gamma(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))))
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))), constraint=constraints.positive)
    sigmasq_alpha = pyro.sample('sigmasq_alpha'.format(''), dist.LogNormal(arg_3,arg_4))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))), constraint=constraints.positive)
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))), constraint=constraints.positive)
    sigmasq_beta = pyro.sample('sigmasq_beta'.format(''), dist.Gamma(arg_5,arg_6))
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))), constraint=constraints.positive)
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))), constraint=constraints.positive)
    sigma_mu0 = pyro.sample('sigma_mu0'.format(''), dist.Gamma(arg_7,arg_8))
    arg_9 = pyro.param('arg_9', torch.ones((amb(1))), constraint=constraints.positive)
    arg_10 = pyro.param('arg_10', torch.ones((amb(1))), constraint=constraints.positive)
    alpha0 = pyro.sample('alpha0'.format(''), dist.Gamma(arg_9,arg_10))
    arg_11 = pyro.param('arg_11', torch.ones((amb(N))), constraint=constraints.positive)
    arg_12 = pyro.param('arg_12', torch.ones((amb(N))), constraint=constraints.positive)
    with pyro.iarange('alpha_prange'):
        alpha = pyro.sample('alpha'.format(''), dist.Gamma(arg_11,arg_12))
    arg_13 = pyro.param('arg_13', torch.ones((amb(1))), constraint=constraints.positive)
    arg_14 = pyro.param('arg_14', torch.ones((amb(1))), constraint=constraints.positive)
    beta0 = pyro.sample('beta0'.format(''), dist.Pareto(arg_13,arg_14))
    arg_15 = pyro.param('arg_15', torch.ones((amb(N))), constraint=constraints.positive)
    arg_16 = pyro.param('arg_16', torch.ones((amb(N))), constraint=constraints.positive)
    with pyro.iarange('beta_prange'):
        beta = pyro.sample('beta'.format(''), dist.Pareto(arg_15,arg_16))
    arg_17 = pyro.param('arg_17', torch.ones((amb(1))))
    arg_18 = pyro.param('arg_18', torch.ones((amb(1))), constraint=constraints.positive)
    gamma = pyro.sample('gamma'.format(''), dist.Normal(arg_17,arg_18))
    arg_19 = pyro.param('arg_19', torch.ones((amb(1))), constraint=constraints.positive)
    arg_20 = pyro.param('arg_20', torch.ones((amb(1))), constraint=constraints.positive)
    theta = pyro.sample('theta'.format(''), dist.Gamma(arg_19,arg_20))
    arg_21 = pyro.param('arg_21', torch.ones((amb(N))), constraint=constraints.positive)
    arg_22 = pyro.param('arg_22', torch.ones((amb(N))), constraint=constraints.positive)
    with pyro.iarange('mu0_prange'):
        mu0 = pyro.sample('mu0'.format(''), dist.Beta(arg_21,arg_22))
    for n in range(1, N1+1):
        pass
    for n in range(1, N+1):
        pass
    
    pass
    return { "alpha0": alpha0,"mu0": mu0,"sigma_mu0": sigma_mu0,"sigmasq_y": sigmasq_y,"sigmasq_alpha": sigmasq_alpha,"beta": beta,"sigmasq_beta": sigmasq_beta,"beta0": beta0,"alpha": alpha,"theta": theta,"gamma": gamma, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(tvec1,Yvec1,N1,N,idxn1,y0)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('alpha0_mean', np.array2string(dist.Gamma(pyro.param('arg_9'), pyro.param('arg_10')).mean.detach().numpy(), separator=','))
print('mu0_mean', np.array2string(dist.Beta(pyro.param('arg_21'), pyro.param('arg_22')).mean.detach().numpy(), separator=','))
print('sigma_mu0_mean', np.array2string(dist.Gamma(pyro.param('arg_7'), pyro.param('arg_8')).mean.detach().numpy(), separator=','))
print('sigmasq_y_mean', np.array2string(dist.Gamma(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('sigmasq_alpha_mean', np.array2string(dist.LogNormal(pyro.param('arg_3'), pyro.param('arg_4')).mean.detach().numpy(), separator=','))
print('beta_mean', np.array2string(dist.Pareto(pyro.param('arg_15'), pyro.param('arg_16')).mean.detach().numpy(), separator=','))
print('sigmasq_beta_mean', np.array2string(dist.Gamma(pyro.param('arg_5'), pyro.param('arg_6')).mean.detach().numpy(), separator=','))
print('beta0_mean', np.array2string(dist.Pareto(pyro.param('arg_13'), pyro.param('arg_14')).mean.detach().numpy(), separator=','))
print('alpha_mean', np.array2string(dist.Gamma(pyro.param('arg_11'), pyro.param('arg_12')).mean.detach().numpy(), separator=','))
print('theta_mean', np.array2string(dist.Gamma(pyro.param('arg_19'), pyro.param('arg_20')).mean.detach().numpy(), separator=','))
print('gamma_mean', np.array2string(dist.Normal(pyro.param('arg_17'), pyro.param('arg_18')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('alpha0:')
    samplefile.write(np.array2string(np.array([guide(tvec1,Yvec1,N1,N,idxn1,y0)['alpha0'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('mu0:')
    samplefile.write(np.array2string(np.array([guide(tvec1,Yvec1,N1,N,idxn1,y0)['mu0'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma_mu0:')
    samplefile.write(np.array2string(np.array([guide(tvec1,Yvec1,N1,N,idxn1,y0)['sigma_mu0'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigmasq_y:')
    samplefile.write(np.array2string(np.array([guide(tvec1,Yvec1,N1,N,idxn1,y0)['sigmasq_y'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigmasq_alpha:')
    samplefile.write(np.array2string(np.array([guide(tvec1,Yvec1,N1,N,idxn1,y0)['sigmasq_alpha'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta:')
    samplefile.write(np.array2string(np.array([guide(tvec1,Yvec1,N1,N,idxn1,y0)['beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigmasq_beta:')
    samplefile.write(np.array2string(np.array([guide(tvec1,Yvec1,N1,N,idxn1,y0)['sigmasq_beta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('beta0:')
    samplefile.write(np.array2string(np.array([guide(tvec1,Yvec1,N1,N,idxn1,y0)['beta0'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('alpha:')
    samplefile.write(np.array2string(np.array([guide(tvec1,Yvec1,N1,N,idxn1,y0)['alpha'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('theta:')
    samplefile.write(np.array2string(np.array([guide(tvec1,Yvec1,N1,N,idxn1,y0)['theta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('gamma:')
    samplefile.write(np.array2string(np.array([guide(tvec1,Yvec1,N1,N,idxn1,y0)['gamma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
