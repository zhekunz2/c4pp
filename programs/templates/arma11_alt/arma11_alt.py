import pyro, numpy as np, torch, pyro.distributions   as dist, torch.nn as nn
from pyro.optim import Adam
import torch.distributions.constraints as constraints
from pyro.infer import SVI
if pyro.__version__ > '0.1.2': from pyro.infer import Trace_ELBO
from pyro.contrib.autoguide import *
import math
def amb(x):
    return x.data.numpy().tolist() if isinstance(x, torch.Tensor) else x
y= np.array([-1.03314101424, -1.88369773346, -1.55658680489, -0.0729770262202, -1.09399326027, -3.09597060186, -2.79707064565, -2.28881681937, -2.32311278743, -1.40842260144, -1.47543413464, -2.71783430562, -2.649754238, -1.03590411042, -0.0118149743562, -0.0100582653536, -1.76094703603, -2.72551340857, -1.53712472007, -2.01664014748, -2.4383129143, -1.98660992843, -1.73852983897, 0.197443433567, -0.655814528081, -1.49114378801, -2.74345131607, -1.7677293988, -1.11936956941, -1.45557239928, -1.77638204365, -0.835970296835, -1.85032706786, -3.5055118983, -2.31436872491, -2.68857122474, -2.93525774831, -1.86500375209, -1.83820258609, -2.42726530402, -2.39644189346, -2.47058297591, -2.03223114059, -2.34174597382, -2.57160270093, -1.4175632237, -1.20167721979, -1.42669979261, -0.995137707141, -0.86605619942, -1.14966967804, -3.39556882547, -2.55377406591, -0.731434990223, -1.287077482, -1.45711085252, -1.54881534328, -2.21500015884, -3.02798313558, -2.55329656552, -1.67234676324, -1.3570944139, -2.06096482761, -2.32845034361, -2.38542817595, -0.0530587873162, -0.266087873391, -0.137793528161, 0.171314749356, -0.102407122613, -0.822335096804, -0.499044901111, -0.709733198402, -1.94135882931, -1.41741218615, -1.91142434189, -1.12461376927, 0.976478136153, 0.781545518055, 0.0286223180534, -0.847592613129, 0.0572722807039, -0.0588499390157, -0.834197392482, -1.96439115611, -2.40959784579, -2.44896749385, -1.66906259167, -0.642694626262, 0.0917768497504, -1.27853801753, -2.7549101755, -2.69616094596, -2.14052953151, -2.48488416352, -1.00613580932, 0.410714290164, -0.247355644931, -1.76115230291, -2.56490228186, -1.13554555218, -0.76264276656, -1.91714356561, -3.41936121113, -2.79713911489, -1.85533663573, -2.72731793044, -2.77080556747, -2.50762743059, -1.77188055817, -1.6757212572, -0.909563787454, -1.29904879156, -1.90642183417, -2.3155657545, -2.11726267261, -0.147767243449, 0.123314907108, -1.04853437633, -1.97516636598, -1.41898854104, -1.32094480258, -1.06152658827, -0.512009382346, -1.14128653768, -1.42383835534, -0.873110669928, -1.59248993918, -2.38752204085, -1.96195840501, -2.97836694727, -1.37847634888, -1.10026914474, -2.26188809827, -2.12344081297, -2.21115598068, -1.33145277514, -2.08091663759, -0.684350541327, -0.294805017655, -0.964119935215, -1.49700401717, -1.60332599728, -2.41735684156, -3.63294401496, -2.17610472765, 0.00758917820987, -0.852278408703, -1.14099690317, -2.23112649343, -2.68865075457, -2.89085576148, -2.33880052655, -0.375035650536, -1.73571721893, -1.04975340017, -0.6752550113, -0.468426201082, -1.55112814768, -3.10495052569, -4.00821106598, -2.8496809526, -2.3758020911, -1.63995614387, -0.607101771863, -1.48457649469, -1.663235537, -2.04704488417, -1.09329573841, 0.245599946322, -0.549786873167, -1.00513887938, -0.924357268251, -1.41261234695, -1.86310963338, -2.25716852405, -2.13836518267, -2.21602276479, -1.28736827069, -0.602611302369, -0.405219098314, -0.285409502031, -0.481230788935, -1.53235580512, -1.21834743436, -1.86393288929, -1.95054058777, -1.21452622296, -2.31093479226, -1.31300829716, -1.33279358049, -0.883120033519, -1.25579059579, -0.993738145485, -1.42986773188, -2.48427429385, -2.7960845136, -1.39173744277, -0.426850467564, -1.15810252228, -2.43582574286, -1.55670684605, -1.20390031583, -0.395444339763, -0.127414745062, -1.17201623377, -2.32172964563, -2.38678790775, -2.33507262236, -2.27068981216, -2.64807711529, -2.37557897721, -0.927690283347, -1.53602403191, -1.93814466716, -1.76573697414, -1.67677928606, -3.19909305232, -2.50205006287, -0.513307968878, -1.22031764545, -1.88852191157, -1.39732947549, -0.30693128479, -0.227907638748, -0.903890955658, -1.33573615336, -2.28190211287, -2.77653959319, -1.72041547887, -1.23843186054, -1.89338584867, -2.10623785029, -2.38767900654, -1.14957521037, -0.0320220884, -0.266550322799, -1.90245197533, -2.49798198709, -1.04764480843, -0.867095929574, -0.532874921361, -1.42920436247, -1.28878439144, -2.26499901051, -1.00178471749, -1.92868305503, -2.83728102557, -2.20471821157, -0.947621465841, -0.601445099336, -1.89336815597, -1.03017971235, -0.405764308612, -0.1965061091, 0.158467278134, -0.358964677037, -1.97067404207, -1.41341103011, -0.726454300684, -0.486159908139, -0.562857770287, -1.80121186276, -2.20227726892, -1.15728594642, -0.237542050368, -0.757154393654, -0.747704842649, -0.298785082625, -1.74149236746, -1.71161062388, -2.16974950695, -3.30107210554, -3.3448366704, -2.76423316307, -2.8631897237, -0.664549084465, -0.948082466535, -0.752484432114, -0.108917061278, -0.950587758572, -2.0491903863, -2.53034729206, -1.82494037978, -0.131204052789, -0.342518283195, -1.31897968266, -2.50349993205, -2.91501636299, -3.13571130766, -3.71061392569, -2.48381072647, -2.145867928, -2.42391558936, -2.02651692878, -1.671615022, -0.367424525415, -2.827528074, -2.47189122636, -1.57881227237, -1.32991953296, -1.34998517265, -1.38667197569, -1.2664184696, -0.872917320601, -1.42020696662, -1.72643979547, -0.871130203285, 0.269263795434, -0.98833575737, -0.734810459023, -1.12160731896, -2.19796690591, -3.21464158946, -3.53156938539, -2.281094101, -2.97169102041, -1.59081272313, -0.408448214372, 0.328874511456, -0.590706105655, -0.637473583958, -1.80486581478, -2.19642794654, -1.16540392297, -2.05259560654, -2.52236585477, -2.59430555994, -0.120209877058, -1.94892620817, -1.67059375728, -1.07617723511, -0.834948844697, -1.87229473813, -3.00095809706, -2.38696203673, -2.46584062697, -1.73750064726, -0.782644767307, -2.07676796768, -2.44667711605, -2.46661250524, -2.8816216367, -3.11854099346, -3.31076088598, -2.17659205303, -0.31643876631, 0.155440789194, -0.0268573613341, -0.722044866276, -1.37876744404, 0.17719699255, -0.590151887074, -1.7436379704, -2.64183847248, -2.01293732044, -2.8916879617, -3.30881469719, -1.84674991912, -0.985815137067, -1.92843210662, -1.71012464331, -1.45344439471, -2.09549962318, -2.14631986349, -2.4458192035, -1.23404578626, -0.683669939743, -1.92014334151, -1.95238479403, -2.92991745247, -3.88763553471, -2.36146748302, -1.40499145075, -0.149822584892, -1.02826706802, -1.77669272799, -1.8315955452, -1.35389515141, -1.59629687998, -0.873018879041, -1.51890721925, -1.48967385698, -0.786928754401, 0.333065913304, 1.14243802797, 0.612379532096, -3.47318741653, -3.01732284342, -3.39319550819, -3.13690179646, -1.96979475155, -1.15041843105, -1.20668804005, -1.67686664532, -2.30462645554, -1.32667195298, -3.13320663106, -3.21874322738, -1.43649934044, -1.36776814886, -1.65592053346, -1.63899740282, -1.63642541722, -1.9397552127, -3.17124156503, -1.82761070042, -0.383372522235, -0.22125665053, 0.106114136751, -0.355859717591, -1.59466040831, -2.47149745385, -2.92650878871, -1.87357403731, -1.10850579708, -0.973651793803, -1.76513784171, -1.76427796412, -2.48411311731, -2.98669214717, -2.52762322549, -2.09228603214, -1.93734040759, -1.54560527415, -0.713872756879, -0.87168355845, -1.60073597123, -3.17086206497, -1.04373084693, -1.27327591504, -0.581013840537, -1.60779347784, -1.65869713432, -2.51681819664, -1.90618183378, -2.15190221159, -2.60201293457, -1.48729046444, -0.211322571946, -1.01075912706, -0.991040915201, -1.30747830181, -1.48999179294, -1.40969651612, -1.26750867135, -0.662411897695, -1.02023808276, -0.906542475711, -1.4202175567, -0.740833756008, -1.98210347378, -2.27239272692, 0.319579628776, 0.143227773416, -0.147794608252, -1.91767976367, -2.48569676688, -2.59295067118, -1.6529454322, -1.29821257353, -0.879506971182, -2.2569965538, -2.04264667243, -0.224853457822, -1.0797657348, -0.50505381006, 0.0479342747953, -0.302528988151, -0.995513671686, -1.76606954065, -2.03784293729, -1.1807625982, -1.1905261238, -2.32008870362, -1.91320208241, -2.66543404033, -2.69637719069, -2.55721860622, -1.87009911486, -1.77437095961, -1.49623827216, -1.61987897627, -0.0866342060747, -0.209764631542, -1.38443606675, -0.876775296851, -1.65264009399, -1.90267371263, -3.15826686003, -1.63846658761, -1.03227264979, -0.242061230457, -0.514915952868, -1.84802340096, -2.89108634962, -1.19543580308, -0.107389009151, -0.0505540534673, -1.24005686, -1.78031750562, -1.72372827246, -1.65286960035, -0.835499963602, -3.11423243573, -3.613776695, -1.66788008173, -2.41819414869, -0.928319711026, -0.796300320013, -1.37469580984, -1.55509034681, -1.25125661247, -1.81816814663, -1.21144188161, -1.71551918459, -1.42986110465, -1.01441910847, -2.19913088642, -1.67452834208, -2.22999443927, -3.01410231967, -1.67648474018, -1.38353024811, -2.3655781129, -2.15095545256, -0.754647605877, -2.10116775703, -3.77074571722, -2.79635604311, -1.60136622343, -1.31348220717, -2.36674702947, -2.36770713222, -1.95527583089, -2.72733837026, -1.72555717263, -1.5782589778, -1.94571549665, -2.19676324441, -1.63902544748, -2.06309217441, -1.91357789386, -1.80760389068, -2.95692533961, -2.54182646759, -1.49243559159, -2.43098817179, -1.57901274905, 0.424802578137, -0.453675650788, -1.76599250449, -2.15020342747, -2.21501772492, -2.18358547792, -1.52041710199, -1.15385286069, -1.40300073902, -1.2167283399, -0.760326582271, -1.80933253702, -2.48045081039, -3.09717843311, -2.37076903474, -1.27242965849, -0.951773354423, -1.49056480952, -0.134604897498, -1.27916953221, -3.14771094274, -1.98461339932, -1.60688666225, -0.762632303733, -1.42361043468, -2.4629950346, -1.96468916184, -1.58904240691, -1.81399093653, -1.74753762085, -1.77938406906, -1.80718654911, -1.83860577201, -2.13149505638, -1.58515283274, -0.716088622535, -1.37283824006, -1.79486819392, -0.790774647923, -1.095395631, -0.722287544675, -1.16734157886, -0.897379884245, -0.711069015246, -0.752946716327, -0.815670000452, -0.36872650316, -0.254655318126, -0.688054426833, -0.944281012676, -0.931980785829, 0.310885028571, -0.518662952649, -0.752243772096, -2.13049299031, -1.81586648478, -0.600733077678, -2.04215133405, -2.39160364032, -1.33843558531, -0.418412279249, -1.34429561315, -0.111020370321, 1.78228868655, 0.315315273953, -1.48227267526, -3.0166534984, -3.6194994391, -1.47159969807, 0.278021976725, 0.183186900885, -1.08582568252, -0.542974570235, -1.14996316717, -2.83127384167, -2.45848281714, -2.84918439155, -2.09100052669, -1.09421186357, -1.64422696505, -0.592504077375, -0.495325441865, -2.16691540025, -0.673824167325, -1.62131370456, -2.02136564575, -1.5456373723, -1.44308713877, -2.4154176381, -3.99824974816, -3.03563602597, -2.78267044523, -2.44781146682, -2.02722580155, -2.64366183581, -2.64244922114, -1.91246277273, -2.19197847526, -2.08125885025, -2.41807525892, -2.72243827767, -2.77233536749, -4.45700728291, -3.26972150042, -2.09349029651, -2.02971078112, -2.19926212488, -1.65406191491, -2.47696738371, -2.41435227944, -1.47658291858, -0.585436700077, -0.947456627621, -3.41319317018, -2.09131559827, -2.61055218597, -1.81199218137, -0.722433575672, -1.24701018627, -0.309253376024, -0.616063838346, -1.25911972905, -1.46809170975, -1.06119763535, -0.889488259985, -0.669380568907, -1.20292362554, -1.71327739464, -0.715762359185, -0.279695241742, -1.36285325325, -1.06698393928, -1.14745144749, -1.28757379389, -2.27284169839, -2.12569366451, -1.20658206541, -1.63961493926, -1.18223735556, -1.49834167689, -1.89435166168, -0.0776556735198, -0.015807718531, -1.56176700818, -2.54561183082, -2.3602734854, -2.03011292077, -2.86638486921, -3.83787522554, -2.39104932428, -1.13696190685, -1.27700545445, -2.53483902567, -1.74307624713, -1.64601933668, -1.0947639967, -2.10842631706, -1.86086574563, -0.710038217822, -0.897496635089, -1.89769719592, -1.4686651918, -2.01025083659, -1.54459777721, -2.12808315935, -3.82563211009, -3.09803004186, -1.59010950407, -1.49946825099, -1.19550725778, -2.56400021312, -1.69737371536, -0.774650450005, -2.01222008755, -1.82091948018, -2.19504893944, -2.80091337224, -2.33263870538, -3.20106404456, -3.3620687919, -2.70889285132, -1.47050977948, -2.44782884672, -2.91447777878, -3.195865417, -2.83569985341, -1.95470711653, -2.21194582525, -2.38689392492, -1.04540553588, -1.2604269548, -1.27095490047, -0.886113408375, 0.217719638024, -0.776999349191, -2.1185254573, -2.00443107268, -1.68762229061, -2.41292705527, -4.23544284947, -3.0677054215, -0.861038990643, -1.40376039978, -1.31450514504, -1.72939271975, -1.83007489227, -1.35696886604, -1.68826398216, -1.42181901862, -2.08569832343, -1.99554663551, -1.3958749656, -1.542165292, -1.01006470407, 0.128511592426, -0.521048615518, -1.50226114735, -1.636455329, -2.62628986174, -1.75792692074, -0.883057750394, -2.29147613025, -1.99843557559, -1.60910379635, -2.09476631633, -2.04356866372, -1.55818854075, -2.78568591371, -2.59179855309, -2.01479791939, -2.89720353512, -2.27173210325, -0.838780849951, -1.2248368111, -1.30410071731, -2.00116260868, -1.42573055958, -0.882940613036, -1.71391916735, -1.79596472891, -1.71841857768, -0.405213957983, -0.338910804721, -0.913539331488, -1.69999173306, -1.60209513308, -1.46350987287, -1.59772062737, -1.43939915265, -2.13303359976, -1.48395504467, -2.02766446413, -1.7240366849, -0.784574430646, -0.628261225503, -1.52561834916, -1.27414807603, -0.860035257165, -0.95778866614, -1.18486301092, -1.45398237114, -0.6199547722, -0.0759904545961, -1.23960395453, -1.93599596144, -0.82093783744, -0.909596359481, -2.27334848113, -2.81936937006, -2.96937785636, -3.04065598939, -1.5827976676, 0.0204071644515, -0.250108848411, -1.28708897012, -2.81769894423, -1.41299050572, -1.5202266522, -2.14368676886, -0.673267161426, -1.00897980088, -1.33825611783, -0.384102048769, 0.0569719002433, -0.656873002615, -0.450592048445, -2.1010173675, -0.687843363652, 0.105292152684, -1.78056365361, -1.73053537222, -1.0973935181, -0.992498447904, -0.79929178439, -1.50753248178, -3.01124276071, -3.53750879649, -3.03059596955, -2.31220780377, -1.15091471955, -0.583617770875, -2.55544990287, -2.33431578303, -2.73345162877, -2.26065180415, -1.30768191553, -1.66679474431, -1.2157375286, -0.912588729825, -0.352925323543, -0.531043526492, -1.07719230777, -2.92546865018, -3.41901013744, -2.52582056467, -1.34061949846, -1.63924289538, -0.954509285204, -0.796741892423, -0.498627584818, -0.926339639891, -0.522710185323, -0.647544083188, -0.958474613914, -1.21858910799, -1.44344524958, -1.56366538391, -1.56276857571, -0.875121547015, -0.933737097967, -2.93713813015, -2.85418790315, -2.56471960047, -3.00748237921, -1.40368037781, -0.64381721929, -0.35945203643, -0.21488102106, -0.00542678482878, -1.86159541408, -1.93629747435, -1.76607767683, -2.0778404602, -1.64624964257, -0.675553590268, 0.443158993948, -0.753429576784, -1.29830676226, -1.15142706383, -0.256372652838, -0.941825998748, -2.03139424201, -1.87097751539, -2.00804774053, -0.579917279486, -0.346047268463, -1.04857106256, -1.36112879542, -1.38723088032, -2.22029934714, -2.12366568722, -2.50444426849, -1.6392333044, -2.03817806035, -3.00945684112, -1.16410284447, 0.755276171121, -0.0218441152117, -1.47203282411, -1.28863601857, -1.0595266933, -0.096225865753, -0.815505861208, -0.580152037957, -1.47203712566, -2.25497089404, -1.55890066227, -2.50836258058, -1.17073469436, -1.22296592974, -1.9204857543, -2.37484291654, -2.90436203525, -3.07547660754, -1.36652041454, -1.53068257439, -0.637594749621, -1.33759768862, -2.07063798492, -1.5475952233, -1.13698289392, -2.48278384904, -3.47521734666, -3.59242931633, -3.47864881094, -2.38116632898, -1.85044770878, -1.72369606821, -0.147684845595, -1.33718399068, -1.36413313275, -0.84686636448, -2.99934224409, -2.86701072047, -1.80371903849, -0.507680895244, 0.354057882458, 0.278599810341, -1.49381865155, -2.66824001259, -0.192197937955, 0.529705761, -0.821260130403, -1.07096876032, -1.13987969959, -3.03073072545, -2.87914056854, -2.50821569445, -1.55496187423, -0.559155515081, -1.037572687, -1.71696205753, -1.70684019624, -3.27017021721, -3.76531652031, -3.08011124931, -2.2793934994, -1.89561190945, -0.379446598169, -1.91960286717, -2.61296415324, -1.7849881465, -1.56379558203, -1.4051683845, -0.841035267259, -0.756535199485, -1.18399511789, -1.18939049633, -0.808664734571, 0.774366124896, -1.33472826259, -2.24106879726, -1.18611996711, -2.1037530083, -1.43612996604, -2.82923576154, -3.00673638037, -1.75761038657, -0.915829686864, -1.17744837159, -1.52790276038], dtype=np.float32).reshape(1000,1)
y=torch.tensor(y)
T=1000
T=torch.tensor(T)
def model(y,T):
    err = torch.zeros([amb(1)])
    mu = pyro.sample('mu'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(10.0)*torch.ones([amb(1)])))
    phi = pyro.sample('phi'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(2.0)*torch.ones([amb(1)])))
    theta = pyro.sample('theta'.format(''), dist.Normal(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(2.0)*torch.ones([amb(1)])))
    sigma = pyro.sample('sigma'.format(''), dist.Cauchy(torch.tensor(0.0)*torch.ones([amb(1)]),torch.tensor(5.0)*torch.ones([amb(1)])))
    err=y[1-1]-mu+phi*mu
    pyro.sample('obs__100'.format(), dist.Normal(0.0,sigma), obs=err)
    for t in range(2, T+1):
        err=y[t-1]-(mu+phi*y[t-1-1]+theta*err)
        pyro.sample('obs_{0}_101'.format(t), dist.Normal(0.0,sigma), obs=err)
    pyro.sample('obs__102'.format(), dist.Normal(0.0,sigma), obs=err)
    
def guide(y,T):
    arg_1 = pyro.param('arg_1', torch.ones((amb(1))), constraint=constraints.positive)
    arg_2 = pyro.param('arg_2', torch.ones((amb(1))), constraint=constraints.positive)
    mu = pyro.sample('mu'.format(''), dist.Beta(arg_1,arg_2))
    arg_3 = pyro.param('arg_3', torch.ones((amb(1))), constraint=constraints.positive)
    arg_4 = pyro.param('arg_4', torch.ones((amb(1))))
    arg_5 = pyro.param('arg_5', torch.ones((amb(1))), constraint=constraints.positive)
    phi = pyro.sample('phi'.format(''), dist.StudentT(df=arg_3,loc=arg_4,scale=arg_5))
    arg_6 = pyro.param('arg_6', torch.ones((amb(1))), constraint=constraints.positive)
    arg_7 = pyro.param('arg_7', torch.ones((amb(1))), constraint=constraints.positive)
    theta = pyro.sample('theta'.format(''), dist.Pareto(arg_6,arg_7))
    arg_8 = pyro.param('arg_8', torch.ones((amb(1))), constraint=constraints.positive)
    arg_9 = pyro.param('arg_9', torch.ones((amb(1))), constraint=constraints.positive)
    sigma = pyro.sample('sigma'.format(''), dist.Weibull(arg_8,arg_9))
    for t in range(2, T+1):
        pass
    
    pass
    return { "mu": mu,"theta": theta,"phi": phi,"sigma": sigma, }
optim = Adam({'lr': 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO() if pyro.__version__ > '0.1.2' else 'ELBO')
for i in range(4000):
    loss = svi.step(y,T)
    if ((i % 1000) == 0):
        print(loss)
for name in pyro.get_param_store().get_all_param_names():
    print(('{0} : {1}'.format(name, pyro.param(name).data.numpy())))
print('mu_mean', np.array2string(dist.Beta(pyro.param('arg_1'), pyro.param('arg_2')).mean.detach().numpy(), separator=','))
print('theta_mean', np.array2string(dist.Pareto(pyro.param('arg_6'), pyro.param('arg_7')).mean.detach().numpy(), separator=','))
print('phi_mean', np.array2string(dist.StudentT(pyro.param('arg_3')).mean.detach().numpy(), separator=','))
print('sigma_mean', np.array2string(dist.Weibull(pyro.param('arg_8'), pyro.param('arg_9')).mean.detach().numpy(), separator=','))
np.set_printoptions(threshold=np.inf)
with open('samples','w') as samplefile:
    samplefile.write('mu:')
    samplefile.write(np.array2string(np.array([guide(y,T)['mu'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('theta:')
    samplefile.write(np.array2string(np.array([guide(y,T)['theta'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('phi:')
    samplefile.write(np.array2string(np.array([guide(y,T)['phi'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
    samplefile.write('sigma:')
    samplefile.write(np.array2string(np.array([guide(y,T)['sigma'].data.numpy() for _ in range(1000)]), separator=',').replace('\n',''))
    samplefile.write('\n')
