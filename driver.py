import pystan
import pickle
from hashlib import md5
import json
import subprocess as sp
from pystan.chains import *
from pystan.misc import *

def StanModel_cache(model_code, rebuild=False):
    code_hash = md5(model_code.encode('ascii')).hexdigest()

    cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    try:
        if not rebuild:
            sm = pickle.load(open(cache_fn, 'rb'))
        else:
            raise FileNotFoundError
    except:
        sm = pystan.StanModel(file=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm

def callstansummary(filename):
    import subprocess
    process = subprocess.Popen('stansummary {0}'.format(filename), stdout=subprocess.PIPE)
    output,err = process.communicate()
    return output,err

import sys

if len(sys.argv) > 1:
    inf_type = sys.argv[1]
else:
    print("Err: Missing inference name")
    exit(-1)
params = False
rebuild = False
if len(sys.argv) > 2:
    if sys.argv[2] == '-r':
        rebuild = True

    if sys.argv[2] == '-p':
        params = True
    elif len(sys.argv) > 3 and sys.argv[3] == '-p':
        params = True
    elif len(sys.argv) > 3 and sys.argv[3] == '-r':
        rebuild = True

sm = StanModel_cache('model.stan', rebuild)
with open('data.json') as dataFile:
    data = json.load(dataFile)

import numpy as np

if inf_type == 'sampling':
    fit = sm.sampling(data=data, iter=4000, chains=4)
    print(fit)
    chains = fit.sim['samples'][0].chains
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    with open('params', 'w') as outputfile:
        for i in range(0, len(fit.sim['fnames_oi'])):
            param = fit.sim['fnames_oi'][i]
            if param == 'lp__':
                continue
            if "[" in param:
                paramtext = param.replace("[", "_").replace("]", "")
                outputfile.write(paramtext + ",")
            else:
                outputfile.write(param + ",")
            outputfile.write(np.array2string(chains[param][2001:], separator=' '))
            outputfile.write(',' + str(ess_and_splitrhat(fit.sim, i)[1]) + "\n")


elif inf_type == 'hmc':
    fit = sm.sampling(data=data, iter=4000, chains=4, algorithm='HMC')
    print(fit)
    chains = fit.sim['samples'][0].chains
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    with open('params', 'w') as outputfile:
        for i in range(0, len(fit.sim['fnames_oi'])):
            param = fit.sim['fnames_oi'][i]
            if param == 'lp__':
                continue
            if "[" in param:
                paramtext = param.replace("[", "_").replace("]", "")
                outputfile.write(paramtext + ",")
            else:
                outputfile.write(param + ",")
            outputfile.write(np.array2string(chains[param][2000:], separator=' '))
            outputfile.write(',' + str(ess_and_splitrhat(fit.sim, i)[1]) + "\n")

elif inf_type == 'vb':
    fit = sm.vb(data=data)
    print(fit['args']['sample_file'])
    result=callstansummary(fit['args']['sample_file'])
    print(result[0])
    np.set_printoptions(threshold=np.inf, linewidth=np.inf)
    with open('params', 'w') as outputfile:
        for i in range(0, len(fit['sampler_param_names'])):
            param_name = fit['sampler_param_names'][i]
            samples = fit['sampler_params'][i]
            if "." in param_name:
                val = int(param_name.split(".")[1])

                paramtext = param_name.replace(".", "_").replace(str(val), str(val - 1))

                outputfile.write(paramtext + ",")
            else:
                outputfile.write(param_name + ",")
            outputfile.write(str(samples).replace(",", " "))
            # outputfile.write(np.array2string(samples, separator=' '))
            outputfile.write("\n")
            # outputfile.write(',' + str(ess_and_splitrhat(fit.sim, i)[1]) + "\n")

