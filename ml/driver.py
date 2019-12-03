import pystan
import pystan
import pickle
from hashlib import md5
import json
import subprocess as sp
from pystan.chains import *
from pystan.misc import *
import numpy as np

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


def predict(Y_test,  Y_test_predicted):
    print("True: " + str(sum([x == 1 for x in Y_test])))
    print("False: " + str(sum([x == 0 for x in Y_test])))

    predictions=[1 if Y_test_predicted[i] >= 0.5  else 0
            for i in range(len(Y_test_predicted))]
    print(Y_test[:10])
    print(Y_test_predicted[:10])
    print(predictions[:10])

    error = sum([pp[0] != pp[1] for pp in zip(predictions, Y_test)])
    correct = sum([pp[0] == pp[1] for pp in zip(predictions, Y_test)])
    # print("Accuracy score: {}".format(accuracy_score(y_test, predictions)))
    print("Error rate: {}/{} = {}".format(error, len(Y_test), error / float(len(Y_test))))
    print("Accuracy: {}/{} = {}".format(correct, len(Y_test), correct / float(len(Y_test))))
    TP = sum([pp[0] == pp[1] and pp[1] for pp in zip(predictions, Y_test)])
    TN = sum([pp[0] == pp[1] and not pp[1] for pp in zip(predictions, Y_test)])
    FP = sum([pp[0] != pp[1] and pp[0] for pp in zip(predictions, Y_test)])
    FN = sum([pp[0] != pp[1] and not pp[0] for pp in zip(predictions, Y_test)])
    print("Precision: {}/{} = {}".format(TP, TP + FP, TP / float(TP + FP)))
    print("Recall   : {}/{} = {}".format(TP, TP + FN, TP / float(TP + FN)))
    print("F1 score :       = {}".format(2 / (1 / (TP / float(TP + FP)) + 1 / (TP / float(TP + FN)))))


def compare(Y_predicted):
    import json
    with open('mydata.json') as f:
        w=json.load(f)
    Y_test=[int(x) for x in w['Y_test']]
    predict(Y_test, Y_predicted)


def sample(data_name, data_size, inf_type, iter=1000, chains=4, printfit=True):
    sm = StanModel_cache('predictor.stan', False)
    with open('mydata.json') as dataFile:
        data = json.load(dataFile)

    if inf_type == 'sampling':
        fit = sm.sampling(data=data, iter=iter, chains=chains)
        if printfit:
            with open('fit', 'w') as f:
                f.write(str(fit))

        predictions = []
        for i in range(data_size):
            vals = [fit.sim['samples'][c].chains['{1}[{0}]'.format(i, data_name)][iter/2:] for c in range(fit.sim['chains'])]
            mean = np.mean(vals)
            predictions.append(mean)
        compare(predictions)

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
                outputfile.write(np.array2string(chains[param][iter/2:], separator=' '))
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
        result = callstansummary(fit['args']['sample_file'])
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

if __name__ == '__main__':
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

    sample('y_test', 100, inf_type)




