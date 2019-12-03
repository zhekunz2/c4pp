import json

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree.export import export_graphviz
import ast
import json
import os
import random
import jsonpickle
import glob

# constants
reverse_metrics=['ks', 't']

# grid search results:
# ./train.py -f csvs/mixture_mutants_1_features.csv csvs/mixture_mutants_2_features.csv -l csvs/mixture_mutants_1_metrics_new.csv csvs/mixture_mutants_2_metrics.csv -a rf -m rhat_min -suf avg -bw -th 1.1 -st -grid -cv_temp
# {'bootstrap': False, 'min_samples_leaf': 4, 'n_estimators': 20, 'min_samples_split': 10, 'criterion': 'entropy', 'max_features': 'sqrt', 'max_depth': 40}

def default(o):
    if isinstance(o, np.int64):
        return int(o)
    if isinstance(o, np.float64):
        return float(o)
    raise TypeError

# functions
def write_csv(results, thresholds, metric_name, args):
    split_results = np.split(np.array(results), len(thresholds))
    keys = results[0].keys()
    if not os.path.exists('results/results.csv'):
        s = 'dataset,algorithm,metric,threshold' + ',' + ','.join(keys) + '\n'
    else:
        s = ''
    n=0
    for res in split_results:
        cols = [args.feature_file[0].split("/")[-1], args.algorithm, metric_name, thresholds[n]]
        for k in keys:
            cols.append(np.mean([x[k] for x in res]))
        s += ','.join([str(x) for x in cols]) + '\n'
        n += 1

    with open('results/results.csv', 'a+') as results_file:
        results_file.write(s)
    with open('results/'+args.saveas.split('/')[1]+'.txt', 'w') as res:
        d = {'results' : results, 'thresholds': thresholds, 'metric_name' : metric_name}
        s=json.dumps(d, default=default)
        #s=jsonpickle.encode(d)
        res.write(s)


def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()


def print_stats(data, metric):
    data = data.replace('inf', np.inf)
    data = data.replace('nan', np.nan)
    data = data[pd.isnull(data[metric]) == False]
    print(data[metric][0])
    infs=data[np.isfinite(data[metric]) == False]
    nans=data[np.isnan(data[metric])]
    print("Infinite: {0}".format(len(infs.index)))
    print("Nans: {0}".format(len(nans.index)))


def show_coefficients(classifier, clf, feature_names, top_features=20):
    if classifier == "svml":
        coef = clf.coef_.ravel()
    elif classifier == "rf":
        if isinstance(clf, Pipeline):
            clf = clf.named_steps['clf']
        coef = clf.feature_importances_
    elif classifier == "dt":
        export_graphviz(clf, out_file='tree.dot', feature_names=feature_names)
        coef = clf.feature_importances_
    elif classifier == 'xgb':
        coef = clf.feature_importances_
    else:
        return
    top_positive_coefficients = np.argsort(coef)[-top_features:][::-1]

    # top_negative_coefficients = np.argsort(coef)[:top_features]
    # top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])

    feature_names = np.array(feature_names)
    print(list(zip(feature_names[top_positive_coefficients], map(lambda x: x, sorted(coef, reverse=True)))))


def filter_by_metrics(features, metric_label, metric_value, labels, threshold):
    # filter by label and drop na

    labels = labels[labels[metric_value].isnull() == False].fillna(0).replace(np.inf, 99999999).replace(-np.inf, 99999999)

    if threshold is None:
        # use default label
        labels[metric_label] = labels[metric_label].apply(lambda x: 1 if str(x).strip() in [True, 'True'] else 0)
    else:
        if metric_label in reverse_metrics:
            labels[metric_label] = labels[metric_value].apply(
                lambda x: 1 if float(str(x).strip()) > float(threshold) else 0)
        else:
            labels[metric_label] = labels[metric_value].apply(
                lambda x: 1 if float(str(x).strip()) < float(threshold) else 0)

    feature_indices = list(features.index.tolist())
    labels_indices = list(labels.index.tolist())
    common_indices = [x for x in feature_indices if x in labels_indices]
    print("common " + str(len(common_indices)))
    # merge
    table = features.join(labels, sort=False)

    table = table.ix[common_indices]
    table = table.fillna(0).replace(np.inf, 99999999).replace(-np.inf, 99999999)
    return table


def plot_results(metric_label, exp_results, metric_thresholds, xlabels, saveas, args):
    font = {'family': 'normal',
            'size': 20}
    #measures = {"Recall": "Rec", "Precision": "Prec", "F1": "F1", "AUC": "AUC", "Diff": "Diff"}
    measures = {"F1": "F1", "AUC": "AUC"}
    matplotlib.rc('font', **font)
    ax = plt.gca()

    from matplotlib.ticker import FormatStrFormatter
    if metric_label in xlabels and xlabels[metric_label] == 'MCMC Iterations':
        pass
    elif metric_label == 't' or metric_label == 'ks':
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    else:
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    min_value = 0.45
    for measure in measures.keys():
        mean=aggregate(np.mean, exp_results, len(metric_thresholds), measure)
        std=aggregate(np.std, exp_results, len(metric_thresholds), measure)
        plt.errorbar(metric_thresholds, mean, label=measures[measure],
                     marker='s', linewidth=3.0, yerr=std)
        min_value=min(min_value, np.min(mean))


    plt.xticks(metric_thresholds)
    plt.grid(True)
    plt.ylim((min_value, 1.05))
    plt.yticks(np.arange(np.round(min_value+0.1, 1), 1.01, 0.1))
    if metric_label in xlabels:
        plt.xlabel(xlabels[metric_label])
    plt.ylabel("Scores")
    # plt.legend()
    plt.legend(loc='center', bbox_to_anchor=(0.5, 1.11), ncol=4, prop={'size': 16})
    plt.tight_layout()
    # annotate(threshold, [x["Precision"] for x in results])
    if args is not None:
        write_csv(exp_results, metric_thresholds, metric_label, args)

    if saveas is not None:
        plt.savefig(saveas)
    else:
        plt.show()


def aggregate(f, arr, groups, field):
    if f == np.std:
        return [[np.mean([x[field] for x in part]) - np.min([x[field] for x in part])
                for part in np.split(np.array(arr), groups)],
                [np.max([x[field] for x in part]) - np.mean([x[field] for x in part])
                 for part in np.split(np.array(arr), groups)]]

    else:
        return [f([x[field] for x in part]) for part in np.split(np.array(arr), groups)]


def my_read_csv(file, index):
    f = open(file).readlines()
    df = pd.DataFrame([x.strip().split(',') for x in f[1:]], columns=f[0].strip().split(','))
    df = df.set_index(index)
    # df=df.astype(np.float32)
    return df


def getRunConfig(model, metric):
    f=json.load(open("rf_params.json"))
    if type(model) == list:
        model=model[0]

    model=model.split('/')[-1].split('.')[0].replace("_features", "")
    return f[model][metric]


def read_all_csvs(csv_files, prog_map, index):
    """
    :type csv_files: list
    :type prog_map: dict
    :type index: str
    """
    if csv_files is None:
        print("No files")
        exit(-1)
    full_data = None
    if not type(csv_files) == list:
        csv_files = [csv_files]
    for file in csv_files:
        if full_data is None:
            full_data = pd.read_csv(file, index_col=index).astype(np.float32)
        else:
            df = pd.read_csv(file, index_col=index).astype(np.float32)
            full_data=full_data.append(df)
    for i in full_data.index.tolist():
        try:
            id=i.split('/')[-2]
            if id not in prog_map:
                prog_map[id]= i.split('/')[-1].replace(".stan", "")
        except:
            pass
    return full_data


def update_features(original_features, files, ignored_indices, prog_map):
    # update table with additional features
    newdata = read_all_csvs(files, prog_map, index='program')
    # checkna(newdata)
    # remove some columns
    for ig in ignored_indices:
        newdata = newdata.filter(set(newdata).difference(set(newdata.filter(regex=(ig)))))
    for i in list(newdata):
        if newdata[i].dtype == np.float64:
            newdata[i] = newdata[i].astype(np.float32)
    original_features = pd.merge(left=original_features, right=newdata, how='left', left_index=True, right_index=True)
    for i in list(original_features):
        if original_features[i].dtype == np.float64:
            original_features[i] = original_features[i].astype(np.float32)
    # original_features = original_features.join(newdata)
    original_features = original_features.replace('inf', np.inf)
    original_features = original_features.fillna(0).replace(np.inf, 99999999).replace(-np.inf, -99999999)
    return original_features


def checkna(table, col=None):
    cols=list(table)
    #print(cols)
    for i in table.index.tolist():
        #print(i)
        if col is None:
            for c in cols:
                try:
                    # float(str(table.ix[i][c]).strip())
                    if np.isreal(table.ix[i][c]) and not np.isfinite(table.ix[i][c]):
                        print('Col {0}'.format(c))
                        print(table.ix[i])
                        exit(1)
                except:
                    print(i)
                    print(table.ix[i])
                    exit(1)
                # if np.isreal(table.ix[i][c]) and not np.isfinite(table.ix[i][c]):
                #     print(table.ix[i])
                #     break
        else:
            try:
                np.float32(str(table.ix[i][col]).strip())
            except:
                #print(i)
                print(table.ix[i])
                exit(1)
            # if not np.isreal(table.ix[i][col]):# and not np.isfinite(table.ix[i][col]):
            # #if np.isreal(table.ix[i][col]) and not np.isfinite(table.ix[i][col]):
            #     print(table.ix[i][col])
            #     break


def annotate(X, Y):
    ax=plt.gca()
    for i,j in zip(X,Y):
        ax.text(i,j,  "{:.2f}".format(j))


def balance(full_table, metric_label):
    # under sampling
    pos = full_table[full_table[metric_label] == 1].index
    neg = full_table[full_table[metric_label] == 0].index
    pos_samples = len(list(pos))
    neg_samples = len(list(neg))
    # print("Positive samples: " + str(pos_samples))
    # print("Negative samples : " + str(neg_samples))
    if pos_samples == 0 or neg_samples == 0:
        indices = list(full_table.index.tolist())
        print("Skipping balance, skewed data...")
    elif pos_samples < neg_samples:
        new_neg_samples = full_table[full_table[metric_label] == 0].sample(pos_samples).index
        indices = list(pos) + list(new_neg_samples)
    else:
        new_pos_samples = full_table[full_table[metric_label] == 1].sample(neg_samples).index
        indices = list(neg) + list(new_pos_samples)

    print("Balanced indices size: {0}".format(len(indices)))
    return indices


def test_train_split_class(full_table, indices, features, metric_label, prog_map, classname, classfile):
    # select programs and its mutants
    print(classname)
    progs=classfile[classname]
    train_ind = filter(lambda x: prog_map[x] not in progs, indices)
    test_ind = filter(lambda x: prog_map[x] in progs, indices)
    X_train = [list(features.ix[i]) for i in train_ind]
    Y_train = [full_table.ix[i][metric_label] for i in train_ind]
    X_test = [list(features.ix[i]) for i in test_ind]
    Y_test = [full_table.ix[i][metric_label] for i in test_ind]

    return X_train, X_test, Y_train, Y_test, train_ind, test_ind


def test_train_validation_splits_by_template(indices, folds, prog_map, feature_file):
    assert folds > 2
    templates = list(set([x.split('_')[0] for x in indices]))
    print(feature_file)
    if 'lrm' in feature_file[0]:
        classes_file='subcategories/lrm2.json'
    elif 'timeseries' in feature_file[0]:
        classes_file = 'subcategories/ts.json'
    else:
        classes_file = 'subcategories/mix.json'
    all_classes = []
    all_classes = json.load(open(classes_file))[3]["multi-level-linear"]

    # for x in json.load(open(classes_file)):
    #     all_classes = all_classes + x[list(x.keys())[0]]
    classes=random.sample(all_classes, 10)
    print(classes)
    #move=filter(lambda x:prog_map[x] in classes, indices)
    #print(move)
    #exit(0)
    print("Templates : {0}".format(len(templates)))
    split_points = np.linspace(0, len(templates), folds+1)
    cv_indices = []
    trains=[]
    tests=[]
    for f in classes:
        #cur_templates = templates[int(split_points[f]):int(split_points[f+1])]
        train_ind = []
        test_ind = []
        # train_ind = filter(lambda x: prog_map[x] not in f, indices)
        # test_ind = filter(lambda x: prog_map[x] in f, indices)
        for i, ind in enumerate(indices):
            if prog_map[ind] not in f:
                train_ind.append(i)
            # if ind.split('_')[0] not in f:
            #     train_ind.append(i)

        for i, ind in enumerate(indices):
            # if ind.split('_')[0] in f:
            #     test_ind.append(i)
            if prog_map[ind] in f:
                test_ind.append(i)
        print(len(train_ind))
        print(len(test_ind))
        if len(test_ind) == 0:
            continue
        trains.append(pd.Index(train_ind))
        tests.append(pd.Index(test_ind))
        cv_indices.append((train_ind, test_ind))

    return zip(trains, tests)


def test_train_split_template(full_table, indices, features, metric_label, ratio, shuffle_templates, prog_map):
    templates = list(set([x.split('_')[0] for x in indices]))
    if shuffle_templates:
        np.random.shuffle(templates)
    print("Templates : {0}".format(len(templates)))
    if type(ratio) == str:
        # filter out only given template name
        print(ratio)
        template_id = list(filter(lambda x: prog_map[x] == ratio, indices))[0].split('_')[0]
        print('id',template_id)
        train_set_templates = list(filter(lambda x:  x != template_id, templates))
        test_set_templates = list(filter(lambda x: x == template_id, templates))
        
    else:
        if type(ratio) == float:
            split_point = int(ratio*len(templates))
        else:
            # leave x out
            split_point = len(templates) - ratio

        train_set_templates = templates[:split_point]
        test_set_templates = templates[split_point:]
        
    # lrm
    # train_set_templates.remove('progs20190331-173424715678') if 'progs20190331-173424715678' in train_set_templates else 0
    # test_set_templates.append('progs20190331-173424715678') if 'progs20190331-173424715678' not in test_set_templates else 0

    # mixture
    #train_set_templates.remove('progs20190404-003947473630') if 'progs20190404-003947473630' in train_set_templates else 0
    #test_set_templates.append('progs20190404-003947473630') if 'progs20190404-003947473630' not in test_set_templates else 0

    train_ind = list(filter(lambda x: x.split('_')[0] in train_set_templates,  indices))
    test_ind = list(filter(lambda x: x.split('_')[0] in test_set_templates,  indices))
    ################ Add some test to train ###############################

    # print("WARNING!! Adding test sample to train")
    # move_samples=200
    # train_ind = train_ind + test_ind[0:move_samples]
    # test_ind=test_ind[move_samples:]
    #######################################################################
    if prog_map is not None:
        print("Templates removed : " ,set([prog_map[x] for x in test_ind]))
    print("Train size: {0}, Test size: {1}".format(len(train_ind), len(test_ind)))
    X_train = features.ix[train_ind]
    Y_train = [full_table.ix[i][metric_label] for i in train_ind]
    X_test = features.ix[test_ind]
    Y_test = [full_table.ix[i][metric_label] for i in test_ind]

    return X_train, X_test, Y_train, Y_test, train_ind, test_ind
