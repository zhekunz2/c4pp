#!/usr/bin/env python
# e.g.: ./train.py -f feature.csv -l pyrotable2 -m hell -ignore_vi -split 0.7 -th 0.5
# e.g.: python train.py -f feature.csv -l all_metrics_on_galeb2.csv -m rhat_min -t 1 -ignore_vi -a rf -st -plt -b
import time
import csv
import xgboost as xgb
from sklearn import metrics
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV, RFE
from utils import *
import driver
import matplotlib
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import validation_curve, learning_curve, GridSearchCV, RandomizedSearchCV
from plot_learning_curve1 import plot_learning_curve
from treeinterpreter import treeinterpreter as ti

start_time = time.time()

pd.set_option('expand_frame_repr', False)


def read_args():
    parser = argparse.ArgumentParser(description='ProbFuzz')
    parser.add_argument('-f', nargs='+', dest='feature_file')
    parser.add_argument('-fo', dest='feature_other', nargs='+')
    parser.add_argument('-l', nargs='+', dest='labels_file')
    parser.add_argument('-m', dest='metric')
    parser.add_argument('-th', dest='threshold', default=None, type=float)
    parser.add_argument('-ignore_vi', dest='ignore_vi', action='store_true')
    parser.add_argument('-split', dest='split_ratio', default=0.8, type=float)
    parser.add_argument('-tname', dest='split_template_name')
    parser.add_argument('-st', dest='split_by_template', action='store_true')
    parser.add_argument('-b', dest='balance', action='store_true', help='balance by subsampling')
    parser.add_argument('-bw', dest='balance_by_weight', action='store_true', help='balance by weight')
    parser.add_argument('-a', dest='algorithm', default='rf')
    parser.add_argument('-cv', dest='cv', action='store_true')
    parser.add_argument('-cv_temp', dest='cv_template', action='store_true', help='cross validation by template split')
    parser.add_argument('-plt', dest='plot', action='store_true')
    parser.add_argument('-validation', dest='validation', action='store_true')
    parser.add_argument('-learning', dest='learning', action='store_true')
    parser.add_argument('-grid', dest='grid', action='store_true')
    parser.add_argument('-suf', dest='metrics_suffix', default=None)
    parser.add_argument('-runtime', dest='runtime', action='store_true')
    parser.add_argument('-predict', nargs='+', dest='predict')
    parser.add_argument('--tree', dest='tree', action='store_true')
    parser.add_argument('--train_by_size', dest='train_by_size', action='store_true')
    parser.add_argument('-class', dest='split_class')
    parser.add_argument('-shuffle', dest='shuffle', action='store_true')
    parser.add_argument('-saveas', dest='saveas', default=None)
    parser.add_argument('-feature_select', dest='feature_select', action='store_true')
    parser.add_argument('-plt_temp', dest='plt_template', action='store_true')
    parser.add_argument('-warmup', dest='warmup', action='store_true')
    parser.add_argument('-stratify', dest='stratify_data', action='store_true')
    parser.add_argument('-special', dest='special_index')
    parser.add_argument('-ignore', dest='ignore', nargs='+')
    parser.add_argument('-keep', dest='keep', nargs='+')
    parser.add_argument('-selected', dest='selected')
    parser.add_argument('-with_noise', dest='with_noise', action='store_true')
    parser.add_argument('-tfpn', dest='tfpn', action='store_true')
    parser.add_argument('-testf', dest='test_features', nargs='+', help='external test features')
    parser.add_argument('-testl', dest='test_labels', help='external test labels')

    args = parser.parse_args()
    print(args)
    return args


def split_dataset(X, Y, metric, ratio, shuffle, stratify_data, threshold):
    if stratify_data:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=ratio,
                                                            shuffle=shuffle)
        print('Train (before) : ', sum([x<threshold for x in Y_train]), sum([x>threshold for x in Y_train]))
        print('Test (before) : ', sum([x<threshold for x in Y_test]), sum([x>threshold for x in Y_test]))

        X_train, Y_train = stratify(X_train, Y_train, metric)
        X_test, Y_test = stratify(X_test, Y_test, metric)

        print('Train (after) : ', sum([x < threshold for x in Y_train]), sum([x >= threshold for x in Y_train]))
        print('Test (after) : ', sum([x < threshold for x in Y_test]), sum([x >= threshold for x in Y_test]))
        if threshold is not None:
            Y_train = [float(y < threshold) for y in Y_train]
            Y_test = [float(y < threshold) for y in Y_test]
        return X_train, X_test, Y_train, Y_test
    else:
        return train_test_split(X, Y, train_size=ratio, shuffle=shuffle)


def stratify(X, Y, metric):
    bins = np.histogram(Y, range=(0.0, 2.0) if metric == 'wass' else (1.0, 2.0), bins=5)

    #nonzerobins=filter(lambda x: bins[0][x-1] != 0, range(1, len(bins[1])))

    digs = np.digitize(Y, bins[1])
    bin_choices = [np.random.choice(list(set(digs))) for _ in range(len(Y))]
    indices = [np.random.choice(list(filter(lambda x: digs[x] == bin, range(len(Y))))) for bin in bin_choices]
    other_indices = list(set(range(len(Y))).difference(set(indices)))
    if isinstance(X, pd.DataFrame):
        Xnew = X.ix[indices+other_indices]
    else:
        Xnew = [X[i] for i in indices + other_indices]
    if isinstance(Y, pd.DataFrame):
        Ynew = Y.ix[indices + other_indices]
    else:
        Ynew = [Y[i] for i in indices + other_indices]

    return Xnew, Ynew



def predict(clf, X_test, Y_test, weighted=False):
    stats = dict()
    stats["True"] = sum([x == 1 for x in Y_test])
    stats["False"] = sum([x == 0 for x in Y_test])
    print("True:" +str(stats["True"]))
    print("False:" + str(stats["False"]))
    stats["Diff"] = (stats["True"]+0.0)/len(Y_test)
    start=time.time()
    predictions = clf.predict(X_test)
    end=time.time()
    print('Total prediction time: ', (end-start))
    print('Predicting for', len(predictions))
    print('Prediction time per instance', ((end-start+0.0)/len(predictions)))
    if isinstance(clf, xgb.XGBClassifier):
        predictions = [round(v) for v in predictions]

    prediction_probs = clf.predict_proba(X_test)

    # print(Y_test[:10])
    # print(predictions[:10])
    # print(prediction_probs[:10])


    incorrect_classes = dict()
    # for i in range(0, len(Y_test)):
    #     original=Y_test[i]
    #     pred = predictions[i]
    #     c=prog_map[Y_test.index[i]]
    #     if original != pred:
    #         if c in incorrect_classes:
    #             incorrect_classes[c][0] += 1
    #         else:
    #             incorrect_classes[c] = [1, 0]
    #     else:
    #         if c in incorrect_classes:
    #             incorrect_classes[c][1] += 1
    #         else:
    #             incorrect_classes[c] = [0, 1]
    # print(incorrect_classes)
    error = sum([pp[0] != pp[1] for pp in zip(predictions, Y_test)])
    correct = sum([pp[0] == pp[1] for pp in zip(predictions, Y_test)])
    stats["Error"] = error / float(len(Y_test))
    stats["Accuracy"] = correct / float(len(Y_test))
    # print("Accuracy score: {}".format(accuracy_score(y_test, predictions)))
    print("Error rate: {}/{} = {}".format(error, len(Y_test),stats["Error"]))
    print("Accuracy: {}/{} = {}".format(correct, len(Y_test), stats["Accuracy"]))
    TP = sum([pp[0] == pp[1] and pp[1] for pp in zip(predictions, Y_test)])
    TN = sum([pp[0] == pp[1] and not pp[1] for pp in zip(predictions, Y_test)])
    FP = sum([pp[0] != pp[1] and pp[0] for pp in zip(predictions, Y_test)])
    FN = sum([pp[0] != pp[1] and not pp[0] for pp in zip(predictions, Y_test)])
    stats["TP"] = TP
    stats["TN"] = TN
    stats["FP"] = FP
    stats["FN"] = FN
    #stats["Precision"] = TP / float(TP + FP)
    #stats["Recall"] = TP / float(TP + FN)
    stats["Precision"] = metrics.precision_score(Y_test, predictions, average='weighted' if weighted else 'binary')
    stats["Recall"] = metrics.recall_score(Y_test, predictions, average='weighted' if weighted else 'binary')
    #stats["F1"] = 2 / (1 / (TP / float(TP + FP)) + 1 / (TP / float(TP + FN)))
    stats["F1"] = metrics.f1_score(Y_test, predictions, average='weighted' if weighted else 'binary')
    try:
        stats["AUC"] = roc_auc_score(Y_test, prediction_probs[:, 1], average="weighted" if weighted else 'macro')
    except:
        stats["AUC"] = 0.0

    # fpr, tpr, thresholds = roc_curve([ int(x) for x in Y_test], prediction_probs[:,1], pos_label=1)
    # print(fpr)
    # print(tpr)
    # print(thresholds)

    #print(metrics.auc(fpr, tpr))
    #plot_roc_curve(fpr, tpr)
    # for i in range(0, len(Y_test)):
    #     print(Y_test.index[i], Y_test[i], predictions[i], Y_test[i] == predictions[i])
    fields.append(stats["Diff"])
    fields.append(stats["Precision"])
    fields.append(stats["Recall"])
    fields.append(stats["F1"])
    fields.append(stats["AUC"])
    # print("TP : ", TP)
    # print("TN : ", TN)
    # print("FP : ", FP)
    # print("FN : ", FN)
    # print("Diff :", stats["Diff"])
    # print("Precision: {}/{} = {}".format(TP, TP + FP, stats["Precision"]))
    # print("Recall   : {}/{} = {}".format(TP, TP + FN, stats["Recall"]))
    # print("F1 score :       = {}".format(stats["F1"]))
    # print("AUC: = {}".format(stats["AUC"]))
    if args.tfpn:
        tfpn = pd.DataFrame(zip(predictions, Y_test, X_test.index), columns = ['prediction', 'Y', 'program'])
        tfpn.set_index("program", inplace=True)
        # tfpn["TFPN"] = ""
        tfpn.loc[(tfpn["prediction"] == tfpn["Y"]) & tfpn["prediction"], "TFPN"] = "TP"
        tfpn.loc[(tfpn["prediction"] == tfpn["Y"]) & (tfpn["prediction"] == 0), "TFPN"] = "TN"
        tfpn.loc[(tfpn["prediction"] != tfpn["Y"]) & tfpn["prediction"], "TFPN"] = "FP"
        tfpn.loc[(tfpn["prediction"] != tfpn["Y"]) & (tfpn["prediction"] == 0), "TFPN"] = "FN"
        tfpn.to_csv("/home/zixin/Documents/are/c4pp/c4pp/ml/tfpn.csv", columns=["TFPN"])
        # print(tfpn)

    return stats


def run_stan(X_train, Y_train, X_test, Y_test):
    data = {}
    data['N'] = len(X_train)
    data['V'] = len(X_train[0])
    data['X_train'] = X_train
    data['Y_train'] = Y_train
    data['N_test'] = len(X_test)
    data['X_test'] = X_test
    data['Y_test'] = Y_test
    with open('mydata.json', 'w') as jsonfile:
        jsonfile.write(json.dumps(data))
    driver.sample('y_test', data['N_test'], 'sampling', 2000, printfit=True)


def run_majority(X_train, Y_train, X_test, Y_test):
    class pred_majority:
        def __init__(self, X_train, Y_train):
            self.true_per = (sum([x == 1 for x in Y_train]) + 0.0)/ len(Y_train)
            print('true per:' , self.true_per)

        def predict(self, test_set):
            return [np.random.choice([0, 1], p=[1 -self.true_per, self.true_per]) for _ in test_set]

        def predict_proba(self, test_set):
            return np.array([ [1 - self.true_per, self.true_per] for _ in test_set])

    clf=DummyClassifier(strategy='stratified')
    clf.fit(X_train, Y_train)
    #return predict(pred_majority(X_train, Y_train), X_test, Y_test)
    return predict(clf, X_test, Y_test)


def run_xgboost(X_train, Y_train, X_test, Y_test, features, cv, grid):
    if grid:
        param_dict = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5],
            'n_estimators' : [20, 50, 100, 200, 500],
            'learning_rate' : [0.01, 0.05, 0.1, 0.5, 1]
        }

        model = xgb.XGBClassifier(nthread=1, objective='binary:logistic')
        grid_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dict,
            n_iter=500,
            cv=5,
            verbose=2,
            n_jobs=-1,
            scoring=metrics.scorer.f1_scorer
            )

        grid_search.fit(X_train, Y_train)
        # print(grid_search.best_params_)
        best_grid = grid_search.best_estimator_
        predict(best_grid, X_test, Y_test)
        exit(1)
    else:
        param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic', 'n_jobs': -1}
        model = xgb.XGBClassifier(**param)
        model.fit(np.array(X_train), np.array(Y_train))
        show_coefficients("xgb", model, features)
        return predict(model, X_test, Y_test)


def run_rf_regression(X_train, Y_train, X_test, Y_test):
    #clf = Ridge(alpha=1.0,  max_iter=1000)

    clf = RandomForestRegressor(**{'bootstrap': False, 'min_samples_leaf': 4, 'n_estimators': 100, 'min_samples_split': 10,  })
    # transform
    print(max(Y_train))
    Y_train = [np.log10(x) if not np.isnan(x) and x > 0.0 else 0 for x in Y_train]
    clf.fit(X_train, Y_train)

    predictions = clf.predict(X_test)
    Y_test = [np.log10(x) if not np.isnan(x) and x > 0.0 else 0 for x in Y_test]

    # normalize:
    # indices = range(0, len(Y_test))
    # indices = filter( lambda x: Y_test[x] < 10, indices)
    # Y_test = [Y_test[i] for i in indices]
    # predictions = [predictions[i] for i in indices]
    print('Max ', np.max(Y_train))
    print('Max ', np.max(Y_test))
    print('MAE: ', metrics.mean_absolute_error(Y_test, predictions))
    print('MSE: ', metrics.mean_squared_error(Y_test, predictions, ))

    print('RMSE: ', np.sqrt(metrics.mean_squared_error(Y_test, predictions)))
    #print('NRMSE:', np.sqrt(metrics.mean_squared_error(Y_test, predictions))/(np.mean(Y_test)-np.min(Y_test)))
    print(Y_test[:10])
    print(predictions[:10])
    plt.plot(range(-10, 30), range(-10, 30), color='black')
    colors = np.random.rand(len(Y_test))
    plt.scatter(Y_test, predictions, c=colors, alpha=0.7, cmap='viridis',s=100*np.random.rand(len(Y_test)))

    plt.xlabel('Log Wass (Actual)')
    plt.ylabel('Log Wass (Predicted)')
    plt.show()
    exit(1)


def run_rf(X_train, Y_train, X_test, Y_test, feature_names, cv=False,  trees=20, validation=False, learning=False,
           gridsearch=False, tree=False, cv_template=False, train_test_indices=None, balance_by_weight=False,
           feature_selection=False, filename=None, test_X=None, test_Y=None):
    # best features for LR: {'bootstrap': True, 'n_estimators': 50, 'min_samples_split': 6, 'criterion': 'entropy',
    # 'max_features': 10, 'max_depth': None}
    # TImeseries{'bootstrap': True, 'n_estimators': 50, 'min_samples_split': 6, 'criterion': 'gini', 'max_features': 10,
    # 'max_depth': None} 91 mins

    # ./train.py -f lrm_mutants_7_features.csv -l lrm_mutants_7_pyro_metrics.csv -m klfix -suf avg -a rf -bw -cv_temp  -grid -th 1.5
    # LR pyro: {'bootstrap': True, 'n_estimators': 20, 'min_samples_split': 10, 'criterion': 'entropy', 'max_features': 10, 'max_depth': None}

    # assert not cv or not cv_template
    if feature_selection:
        results = []
        for f in range(5, 100, 100):
            print('Features : ', f)
            clf = RandomForestClassifier(bootstrap=True, n_estimators=50, min_samples_split=6,
                                         criterion='entropy', max_depth=None, n_jobs=-1)
            selector = RFECV(clf, step=1, cv=5, verbose=True, n_jobs=-1, scoring=metrics.make_scorer(metrics.f1_score))
            selector = selector.fit(X_train, Y_train)
            for i in range(len(feature_names)):
                print(feature_names[i],':', selector.support_[i], ', ', selector.ranking_[i])

            X_train_new = selector.transform(X_train)
            X_test_new = selector.transform(X_test)
            print(np.shape(X_train_new))
            print(np.shape(X_test_new))
            #clf.fit(X_train_new, Y_train)
            results.append(predict(selector, X_test, Y_test))
        plot_results(label, results, range(5, 100, 100), xlabels, args.saveas, args)

    elif cv and len(Y_train) > 2:
        best_k, best_score = -1, -1
        clfs = {}
        rfconfig = getRunConfig(args.feature_file, args.metric)
        if cv_template and train_test_indices is not None:
            # do cross validation based on templates
            cv_indices = test_train_validation_splits_by_template(train_test_indices[0], 5, prog_map, args.feature_file)
            print('cv_indices', np.shape(cv_indices))
        else:
            cv_indices = int(min(5, sum(Y_train), len(Y_train) - sum(Y_train)))

        for k in [10, 40,  80, 100, 250]:
            #rfconfig = getRunConfig(args.feature_file, args.metric)
            rfconfig = dict()
            rfconfig['n_estimators'] = k
            rfconfig['n_jobs'] = -1
            rfconfig['class_weight'] = 'balanced' if balance_by_weight else None
            from sklearn import preprocessing
            pipe = Pipeline([['clf', RandomForestClassifier(**rfconfig)]])
            pipe.fit(X_train, Y_train)
            scores = cross_val_score(pipe, X_train, Y_train, cv=cv_indices, scoring=metrics.make_scorer(metrics.f1_score), n_jobs=-1)
            print('rf-n_est={}\nValidation accuracy: {}'.format(k, scores.mean()))
            print(scores)
            if scores.mean() > best_score:
                best_k, best_score = k, scores.mean()
            clfs[k] = pipe
        clf = clfs[best_k]
    elif tree and len(Y_train) > 2:
        from cleanlab.classification import LearningWithNoisyLabels
        from cleanlab.noise_generation import generate_noisy_labels, generate_noise_matrix_from_trace
        from cleanlab.util import print_noise_matrix


        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced' if balance_by_weight else None,
                                     n_jobs=-1)
        #clf = LearningWithNoisyLabels(clf=clf)
        clf.fit(np.array(X_train), np.array(Y_train))
        #clf.fit(np.array(X_train), np.array([0 if yy < 0.5 else 1 for yy in Y_train]))

        #arr = filter(lambda x: x.name == 'progs20190331-205646326686_prob_rand_83', X_test)
        #arr = filter(lambda x: x.name == 'progs20190331-173424715678_prob_rand_87', X_test)
        #arr = filter(lambda x: x.name == 'progs20190621-205408227197_prob_rand_196', X_test)
        arr = X_test.ix[args.special_index]
        prediction, bias, contributions = ti.predict(clf, np.array([arr]))
        print(clf)
        # print(prediction)
        # print(bias)
        # print(contributions)
        for i in range(len(prediction)):
            print("Instance", i)
            print("Bias (trainset mean)", bias[i])
            print("Prediction :", prediction[i])

            #print(Y_test.ix[args.special_index])
            print("Feature contributions:")
            for c, feature in sorted(zip(contributions[i],
                                         feature_names),
                                     key=lambda x: -abs(x[0][1]))[:20]:

                print("{0} :: {1}".format(feature, c))
            print("-" * 20)
        predict(clf, X_test, Y_test)
        exit(-1)
    elif learning and len(Y_train) > 2:
        clf = RandomForestClassifier(n_estimators=trees, class_weight='balanced' if balance_by_weight else None, n_jobs=-1)
        plot_learning_curve(clf, 'abc', X_train, Y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5, n_jobs=9)
        plt.show()
        exit(1)

        clf = RandomForestClassifier(n_estimators=trees, class_weight='balanced' if balance_by_weight else None, n_jobs=-1)
        print(len(X_train))
        print(len(Y_train))
        train_sizes, train_scores, valid_scores= learning_curve(clf, X_train, Y_train, train_sizes=np.linspace(0.1, 1.0, 10), cv=5)
        print(train_sizes)
        print(train_scores)
        print(valid_scores)
        plt.plot(train_sizes, [np.mean(x) for x in train_scores], label='training')
        plt.plot(train_sizes, [np.mean(x) for x in valid_scores], label='validation')
        plt.legend()
        plt.show()
        exit(1)

    elif validation and len(Y_train) > 2:
        train_sizes = np.linspace(80, 250, 10, dtype=np.int)
        train_scores, valid_scores = validation_curve(RandomForestClassifier(), X_train, Y_train, "n_estimators", train_sizes, cv=10)
        plt.plot(train_sizes, [np.mean(x) for x in train_scores], label='training')
        plt.plot(train_sizes, [np.mean(x) for x in valid_scores], label='validation')
        plt.legend()
        plt.show()
        exit(1)
    elif gridsearch and len(Y_train) > 2:
        param_dist = {"max_depth": list(range(10, 100, 10)) + [None],
                      "max_features": ['auto', 10, 20, 30, 50, 80, 100],
                      "min_samples_split": np.linspace(2, 20, 5, dtype=np.int),
                      "bootstrap": [True, False],
                      "min_samples_leaf": [ 1, 2, 4, 7],
                      "criterion": ["gini", "entropy"],
                      "n_estimators": [20, 50, 100, 250, 500, 1000]

                      }
        if cv_template and train_test_indices is not None:
            # do cross validation based on templates
            cv_indices = test_train_validation_splits_by_template(train_test_indices[0], 5, prog_map)
            print('cv_indices', np.shape(cv_indices))
        else:
            cv_indices = int(min(4, sum(Y_train), len(Y_train) - sum(Y_train)))

        grid_search = RandomizedSearchCV(estimator=RandomForestClassifier(class_weight='balanced' if balance_by_weight else None),
                                         param_distributions=param_dist,
                                         n_iter=500,
                                         cv=cv_indices,
                                         verbose=2,
                                         n_jobs=-1,
                                         scoring=metrics.scorer.f1_scorer
                                         )
        # grid_search = GridSearchCV(estimator=RandomForestClassifier(class_weight='balanced' if balance_by_weight else None),
        #                            param_grid=param_dist,
        #                            cv=cv_indices,
        #                            n_jobs=-1,
        #                            verbose=2,
        #                            scoring=metrics.scorer.f1_scorer)
        grid_search.fit(X_train, Y_train)
        print(grid_search.best_params_)
        best_grid = grid_search.best_estimator_
        predict(best_grid, X_test, Y_test)
        exit(1)
    else:
        #rfconfig=getRunConfig(args.feature_file, args.metric)
        rfconfig=None

        #clf = RandomForestClassifier(n_estimators=200, bootstrap=True, n_jobs=-1)
        # clf = RandomForestClassifier(n_estimators=50, min_samples_split=6, criterion='entropy', max_features=10, max_depth=None)
        #clf = RandomForestClassifier(n_estimators=100, max_features=100, bootstrap=True, min_samples_split=10, criterion='entropy')
        #clf = RandomForestClassifier(n_estimators=20, min_samples_split=10, criterion='entropy', max_features = 10 if len(list(X_train)) > 10 else len(list(X_train)),                                     max_depth=None, bootstrap=True, n_jobs=-1)
        if rfconfig is not None:
            print(rfconfig)
            clf = RandomForestClassifier(**rfconfig)
        else:
            clf = RandomForestClassifier(n_estimators=50, criterion='gini' if args.metric == 'rhat_min' else 'entropy', class_weight='balanced' if balance_by_weight else None)
            #clf = RandomForestClassifier(n_estimators=50, min_samples_split=6, criterion='gini', max_features=10, max_depth=None, class_weight='balanced' if balance_by_weight else None)
        if args.with_noise:
            ###################################Noisy
            from cleanlab.classification import LearningWithNoisyLabels
            from cleanlab.noise_generation import generate_noisy_labels, generate_noise_matrix_from_trace
            from cleanlab.util import print_noise_matrix

            lnl = LearningWithNoisyLabels(clf=clf)

            ##Generate Noisy Label
            for noise_level in [0.00,0.01,0.02,0.03,0.04,0.05]: #[0.1,0.2,0.3,0.4]: #
                print("Noise level: {}".format(noise_level))
                np.random.seed(seed=42)
                py = np.bincount([0 if yy < 0.5 else 1 for yy in Y_train]) / float(len(Y_train))
                noise_matrix = generate_noise_matrix_from_trace( K = 2, trace = 2 * (1-noise_level) , py = py, frac_zero_noise_rates = 0)
                print_noise_matrix(noise_matrix)
                Y_noisy_train = generate_noisy_labels(Y_train, noise_matrix)

                ##Fit with Noisy Label
                lnl.fit(np.array(X_train), np.array([0 if yy < 0.5 else 1 for yy in Y_noisy_train]))
                #predictions = lnl.predict(X_test)

                predict(lnl, X_test, Y_test)
                clf.fit(np.array(X_train), np.array(Y_noisy_train))
                predict(clf, X_test, Y_test)
            return
            ###################################Noisy
        else:
            clf.fit(np.array(X_train), np.array(Y_train))
    #clf.fit(np.array(X_train), np.array(Y_train))
    if not cv:
        show_coefficients("rf", clf, feature_names=feature_names, top_features=20)
    if test_X is not None:
        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        return predict(clf, test_X, test_Y)
        print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    if Y_test is not None:
        return predict(clf, X_test, Y_test)
    else:
        return clf


def run_dtree(X_train, Y_train, X_test, Y_test, features):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion="entropy")
    clf.fit(np.array(X_train), np.array(Y_train))
    show_coefficients("dt", clf, feature_names=features)
    return predict(clf, X_test, Y_test)


def clean(f):
    f.index = f.index.map(lambda x: x.split('/')[-2] if len(x.split('/')) > 1 else x)
    for pat in ignore_indices:
        f = f.filter(set(f).difference(set(f.filter(regex=(pat)))))
    f = f.replace('inf', np.inf)
    f = f.fillna(0).replace(np.inf, 99999999).replace(-np.inf, -99999999)
    return f

# parse arguments
args = read_args()
prog_map = {}
# read features and labels
features=read_all_csvs(args.feature_file, prog_map, 'program')
labels=read_all_csvs(args.labels_file, prog_map, 'program')

fields=[]
fields.append(args.feature_other)

# keep only the selected indices using lsh
###########################################
if args.selected is not None:
    selected_indices=open(args.selected).read().strip().splitlines()
    selected_indices=list(set(list(labels.index)).intersection(selected_indices))
    print('Before:: ', len(labels))
    labels = labels.ix[selected_indices]
    print('Keeping indices:: ', len(labels))
################################

label=args.metric
threshold=args.threshold
test_train_split = args.split_ratio
algorithm = args.algorithm
plot_table=args.plot
split_by_template=args.split_by_template
split_by_class=args.split_class
ignore_indices=['vi_.*', '17'] if args.ignore_vi else []
ignore_indices_other = ['iter']

shuffle=args.shuffle
thresholds={ 'kl': [0.1, 1, 2, 3, 4, 5],
             'ekl' : [0.1, 0.5, 0.75, 1, 2, 3, 4],
             'klfix' : [0.1, 1, 2, 3, 4, 5],
             'smkl': [0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
             'smklfix' : [0.01, 0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9],
             'ks': list(np.arange(0.01,0.1,0.01)),
             't' : list(np.arange(0.01,0.1,0.01)),
             'hell': list(np.arange(0.1,1, 0.1)),
             'rhat_min' : [1.05, 1.1, 1.2, 1.4, 1.6,  1.8,  2.0],
             'ess_min' : list(np.arange(0.01, 0.09, 0.01)),
             'wass' : [0.05] + list(np.arange(0.1, 1.0, 0.1)),
             'js' : list(np.arange(0.1,0.9, 0.1))
             }
xlabels = {'kl': 'Threshold of KL Divergence',
           'klfix' : 'Threshold of KL Divergence',
           'rhat_min': 'Threshold for Gelman-Rubin Diagnostic',
           'wass' : 'Threshold of Wasserstein Dist'}

runtimexlabels = {'kl': 'MCMC Iterations', 'rhat_min' : 'MCMC Iterations', 'klfix': 'MCMC Iterations', 'wass': 'MCMC Iterations'}
if args.plot:
    threshold = thresholds[label]

# cleaning
features.index = features.index.map(lambda x:  x.split('/')[-2] if len(x.split('/')) > 1 else x)

if args.keep is not None:
    f_keep = []
    for k in args.keep:
        f_keep += list(filter(lambda x: x.startswith(k), list(features)))
    # print(f_keep)

    features=features.filter(f_keep)

for pat in ignore_indices:
    features=features.filter(set(features).difference(set(features.filter(regex=(pat)))))


# join additional features
if args.feature_other is not None:
    # update table with additional features
    for fo in args.feature_other:
        feature_other=read_all_csvs(fo, prog_map, index='program')
        feature_other.index = feature_other.index.map(lambda x:  x.split('/')[-2] if len(x.split('/')) > 1 else x)
        for pat in ignore_indices:
            feature_other = feature_other.filter(set(feature_other).difference(set(feature_other.filter(regex=(pat)))))
        # remove some columns
        for ig in ignore_indices_other:
            feature_other=feature_other.filter(set(feature_other).difference(set(feature_other.filter(regex=(ig)))))
        # for i in list(feature_other):
        #     if feature_other[i].dtype == np.float64:
        #         feature_other[i]=feature_other[i].astype(np.float32)

        features=pd.merge(left=features, right=feature_other, how='outer', left_index=True, right_index=True)

#print(list(features))

features=features.replace('inf', np.inf)
features=features.fillna(0).replace(np.inf, 99999999).replace(-np.inf, -99999999)
#checkna(features)
#exit(1)
# # remove missing
# feature_indices=list(features.index.tolist())
# labels_indices=list(labels.index.tolist())
# common_indices=list(set(feature_indices).intersection(set(labels_indices)))
if args.metrics_suffix is None:
    metric_label = label + '_result'
    metric_value = label + '_value'
else:
    metric_label = label + '_result_' + args.metrics_suffix
    metric_value = label + '_value_' + args.metrics_suffix
#checkna(labels, metric_value)
print_stats(labels, metric_value)

# shuffling features and labels
features = features.reindex(np.random.permutation(features.index))
#labels = labels.reindex(np.random.permutation(labels.index))

for i in list(features.index):
    if 'progs20190621-213255446133' in i or 'progs20190621-205423791989' in i:
        #print(list(features))
        features=features.drop(i)
        # temp=features.ix[i]["dt_max_autocorr_predictor_1"]
        # temp2=features.ix[i]['dt_max_autocorr_predictor_2']
        # features.loc[i]['dt_max_autocorr_observe_1']=temp
        # features.loc[i]['dt_max_autocorr_observe_2'] = temp2
        # features.loc[i]['dt_max_autocorr_predictor_1']=0.0
        # features.loc[i]['dt_max_autocorr_predictor_2']=0.0
        #
        # features.loc[i]['dt_observe_skew'] = features.loc[i]['dt_predictor_skew']
        # features.loc[i]['dt_observe_kurtosis'] = features.loc[i]['dt_predictor_kurtosis']
        # features.loc[i]['dt_predictor_skew'] = 0.0
        # features.loc[i]['dt_predictor_kurtosis'] = 0.0
                                                                                            
cols=list(features)
np.random.shuffle(cols)
features = features[cols]
test_features=args.test_features
# to test on external set of programs
if test_features is not None and args.test_labels is not None:
    test_features=read_all_csvs(args.test_features, prog_map, 'program')
    test_labels=read_all_csvs(args.test_labels, prog_map, 'program')
    test_features.index=test_features.index.map(lambda x:  x.split('/')[-2] if len(x.split('/')) > 1 else x)
    for pat in ignore_indices:
        if test_features is not None:
            test_features = test_features.filter(set(test_features).difference(set(test_features.filter(regex=(pat)))))
    if f_keep is not None:
        import re
        ast_stuff=list(filter(lambda x: len(re.findall('[a-zA-Z]',x)) ==0, list(test_features)))
        test_features = test_features.filter(f_keep+ast_stuff)
    test_features = test_features.replace('inf', np.inf)
    test_features = test_features.fillna(0).replace(np.inf, 99999999).replace(-np.inf, -99999999)
    #test_full_table = filter_by_metrics(test_features, metric_label, metric_value, test_labels, t)

# transform features
#features['data_size'] = features['data_size'].apply(lambda x: np.log10(x+1))


def get_class_dist(X_train, X_test):
    assert len(set(X_train.index).intersection(set(X_test.index))) == 0
    train_class=dict()
    test_class=dict()
    for i in list(X_train.index):
        p = prog_map[i]
        if p in train_class:
            train_class[p] += 1
        else:
            train_class[p] = 1
    for i in list(X_test.index):
        p = prog_map[i]
        if p in test_class:
            test_class[p] += 1
        else:
            test_class[p] = 1
    # print("Train")
    # print(train_class)
    # print("Test")
    # print(test_class)

if type(threshold) == list:
    results = []
    repetitions = 3
    for t in threshold:
        print("Threshold : {0}============================================".format(t))
        full_table = filter_by_metrics(features, metric_label, metric_value, labels, t)
        if test_features is not None:
            test_features=test_features.reindex(columns=list(features))
            test_features = test_features.replace('inf', np.inf)
            test_features=test_features.fillna(0).replace(np.inf, 99999999).replace(-np.inf, -99999999)
            test_full_table = filter_by_metrics(test_features, metric_label, metric_value, test_labels, t)
        else:
            test_full_table = None
        #print(list(full_table))
        pos = full_table[full_table[metric_label] == 1].index
        neg = full_table[full_table[metric_label] == 0].index
        pos_samples = len(list(pos))
        neg_samples = len(list(neg))
        print("Total Positive samples: " + str(pos_samples))
        print("Total Negative samples : " + str(neg_samples))
        if args.balance:
            ind = balance(full_table, metric_label)
        else:
            ind = list(full_table.index.tolist())
        for _ in range(repetitions):
            if split_by_template:
                X_train, X_test, Y_train, Y_test, train_ind, test_ind = test_train_split_template(
                    full_table,
                    ind,
                    features,
                    metric_label,
                    test_train_split if args.split_template_name is None else args.split_template_name,
                    shuffle,
                    prog_map
                )
            else:
                X = [list(features.ix[i]) for i in ind]
                if algorithm == 'rf_reg' or args.stratify_data:
                    Y = [full_table.ix[i][metric_value] for i in ind]
                else:
                    Y = [full_table.ix[i][metric_label] for i in ind]

                X_train, X_test, Y_train, Y_test = split_dataset(X, Y,
                                                                 label,
                                                                 ratio=test_train_split,
                                                                 shuffle=shuffle,
                                                                 stratify_data=args.stratify_data,
                                                                 threshold=t if algorithm != 'rf_reg' else None)
                train_ind = test_ind = None
            if algorithm == 'stan':
                results.append(run_stan(X_train, Y_train, X_test, Y_test))
            elif algorithm == 'rf':
                if test_full_table is not None:
                    print(test_full_table.index.tolist())
                    results.append(run_rf(X_train, Y_train, X_test, Y_test, list(features), cv=args.cv,
                                          cv_template=args.cv_template, train_test_indices=(train_ind, test_ind),
                                          filename=args.feature_file,
                                          test_X=[list(test_features.ix[i]) for i in list(test_full_table.index.tolist())],
                                          test_Y=[test_full_table.ix[i][metric_label] for i in list(test_full_table.index.tolist())]))
                else:
                    results.append(run_rf(X_train, Y_train, X_test, Y_test, list(features), cv=args.cv,
                                      cv_template=args.cv_template, train_test_indices=(train_ind, test_ind),
                                      filename=args.feature_file))
            elif algorithm == 'dt':
                results.append(run_dtree(X_train, Y_train, X_test, Y_test,  list(features)))
            elif algorithm == 'maj':
                results.append(run_majority(X_train, Y_train, X_test, Y_test))
            elif algorithm == 'xgb':
                results.append(run_xgboost(X_train, Y_train, X_test, Y_test, list(features), args.cv, False))
            elif algorithm == 'rf_reg':
                results.append(run_rf_regression(X_train, Y_train, X_test, Y_test))

    plot_results(label, results, threshold, xlabels, args.saveas, args)
elif args.plt_template:
    # plt performance by templates
    results = []
    for templates in range(1,11):
        full_table = filter_by_metrics(features, metric_label, metric_value, labels, threshold)
        if args.balance:
            ind = balance(full_table, metric_label)
        else:
            ind = list(full_table.index.tolist())
        X_train, X_test, Y_train, Y_test, train_ind, test_ind = test_train_split_template(full_table,
                                                                                     ind,
                                                                                     features,
                                                                                     metric_label,
                                                                                     templates,
                                                                                     shuffle,
                                                                                      prog_map)

        results.append(run_rf(X_train, Y_train, X_test, Y_test, list(features), cv=args.cv,
                                      cv_template=args.cv_template, train_test_indices=(train_ind, test_ind)))
    plot_results(metric_label, results, range(1,11), xlabels, args.saveas, args)
    exit(1)

elif args.runtime:

    # for feature_file in args.feature_file:
    #     runtime_files += ['{0}_runtime_{1}.csv'.format(feature_file.split('_features')[0], i) for i in range(10,101,10)]
    # print(runtime_files)

    runtime_results=[]
    if args.warmup:
        runtimes= range(0,1001, 100)
    else:
        runtimes = range(0, 101, 20)

    for iteration in runtimes:
        print("Runtime iteration {0}=====================".format(iteration))
        repetitions = 3
        for _ in range(repetitions):
            features_new = features
            if iteration > 0:
                runtime_files=[]
                for file in args.feature_file:
                    print(file)
                    if args.warmup:
                        runtime_files += ['{0}_warmup_runtime_{1}.csv'.format(file.split('_features')[0], iteration)]
                    else:
                        runtime_files += ['{0}_runtime_{1}.csv'.format(file.split('_features')[0], iteration)]
                print(runtime_files)
                features_new = update_features(features_new, runtime_files, ignore_indices_other, prog_map)

            full_table = filter_by_metrics(features_new, metric_label, metric_value, labels, threshold)
            # if iteration > 0:
            #     checkna(full_table)
            pos = full_table[full_table[metric_label] == 1].index
            neg = full_table[full_table[metric_label] == 0].index
            pos_samples = len(list(pos))
            neg_samples = len(list(neg))
            print("Positive samples: " + str(pos_samples))
            print("Negative samples : " + str(neg_samples))
            if args.balance:
                ind = balance(full_table, metric_label)
            else:
                ind = list(full_table.index.tolist())

            if split_by_template:
                X_train, X_test, Y_train, Y_test, train_ind, test_ind = test_train_split_template(full_table,
                                                                                                  ind,
                                                                                                  features_new,
                                                                                                  metric_label,
                                                                                                  test_train_split if args.split_template_name is None else args.split_template_name,
                                                                                                  shuffle,
                                                                                                  prog_map)
            else:

                X = [list(features_new.ix[i]) for i in ind]
                if algorithm == 'rf_reg' or args.stratify_data:
                    Y = [full_table.ix[i][metric_value] for i in ind]
                else:
                    Y = [full_table.ix[i][metric_label] for i in ind]
                X_train, X_test, Y_train, Y_test = split_dataset(X, Y,
                                                                 label,
                                                                 ratio=test_train_split,
                                                                 shuffle=shuffle,
                                                                 stratify_data=args.stratify_data,
                                                                 threshold=threshold if algorithm != 'rf_reg' else None)

                train_ind = test_ind = None
            if algorithm == 'stan':
                runtime_results.append(run_stan(X_train, Y_train, X_test, Y_test))
            elif algorithm == 'rf':
                runtime_results.append(run_rf(X_train, Y_train, X_test, Y_test, list(features_new), cv=args.cv,
                                              cv_template=args.cv_template, train_test_indices=(train_ind, test_ind)))
            elif algorithm == 'dt':
                runtime_results.append(run_dtree(X_train, Y_train, X_test, Y_test, list(features_new)))
            elif algorithm == 'maj':
                runtime_results.append(run_majority(X_train, Y_train, X_test, Y_test))
            elif algorithm == 'xgb':
                runtime_results.append(run_xgboost(X_train, Y_train, X_test, Y_test, list(features), args.cv, False))

    plot_results(label, runtime_results, runtimes, runtimexlabels, args.saveas, args)
    exit(1)

elif args.train_by_size:
    results = []
    train_size=[0.5, 0.6, 0.7, 0.8, 0.9]
    full_table = filter_by_metrics(features, metric_label, metric_value, labels, threshold)
    pos = full_table[full_table[metric_label] == 1].index
    neg = full_table[full_table[metric_label] == 0].index
    pos_samples = len(list(pos))
    neg_samples = len(list(neg))
    print("Positive samples: " + str(pos_samples))
    print("Negative samples : " + str(neg_samples))
    # balance_by_template(full_table, metric_label)

    # under sampling
    if args.balance:
        common_indices = balance(full_table, metric_label)
    else:
        common_indices = list(full_table.index.tolist())
    import random
    if shuffle:
        random.shuffle(common_indices)
    for size in train_size:
        print("Training size : {0}=====================".format(size))
        X = [list(features.ix[i]) for i in common_indices]
        Y = [full_table.ix[i][metric_label] for i in common_indices]
        print(np.shape(X))
        print(np.shape(Y))
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=size,
                                                            test_size=1.0 - size, shuffle=shuffle)
        print("Train {0}".format(len(X_train)))
        print("Test {0}".format(len(X_test)))
        results.append(run_rf(X_train, Y_train, X_test, Y_test, list(features), cv=args.cv, cv_template=args.cv_template))

    plot_results(label, results, train_size, xlabels, args.saveas, args)
    # font = {'family': 'normal',
    #         'size': 20}
    #
    # matplotlib.rc('font', **font)
    # #plt.figure(figsize=(12, 8))
    # plt.plot(train_size, [x["Recall"] for x in results], label='Recall', marker='s', linewidth=3.0)
    # plt.plot(train_size, [x["Precision"] for x in results], label='Precision', marker='s', linewidth=3.0)
    # #plt.plot(train_size, [x["Diff"] for x in results], label='Diff', marker='s')
    # #plt.plot(train_size, [x["Accuracy"] for x in results], label='Accuracy', marker='s')
    # plt.plot(train_size, [x["F1"] for x in results], label='F1 Score', marker='s', linewidth=3.0)
    # plt.xticks(train_size)
    # plt.grid(True)
    # #plt.ylim((0.5, 1.0))
    # plt.xlabel('Training Size')
    # plt.ylabel("Scores")
    # plt.legend()
    #
    # # annotate(threshold, [x["Precision"] for x in results])
    # # plt.xticks(np.arange(min(threshold), max(threshold), 1))
    #
    # # plt.xticks(threshold)
    # plt.show()
elif args.predict is not None:
    topredict = read_all_csvs(args.predict,prog_map, 'program')
    topredict = clean(topredict)
    full_table = filter_by_metrics(features, metric_label, metric_value, labels, threshold)
    pos = full_table[full_table[metric_label] == 1].index
    neg = full_table[full_table[metric_label] == 0].index
    pos_samples = len(list(pos))
    neg_samples = len(list(neg))
    print("Positive samples: " + str(pos_samples))
    print("Negative samples : " + str(neg_samples))
    # balance_by_template(full_table, metric_label)

    # under sampling
    if args.balance:
        common_indices = balance(full_table, metric_label)
    else:
        common_indices = list(full_table.index.tolist())

    X = [features.ix[i] for i in common_indices]
    if algorithm == 'rf_reg':
        Y = [full_table.ix[i][metric_value] for i in common_indices]
    else:
        Y = [full_table.ix[i][metric_label] for i in common_indices]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=test_train_split,
                                                        test_size=1.0 - test_train_split)
    if algorithm == 'rf':
        clf = run_rf(X_train, Y_train, None, None, list(features), cv=args.cv, cv_template=args.cv_template,
               gridsearch=args.grid, learning=args.learning, validation=args.validation, tree=args.tree,
               train_test_indices=([x.name for x in X_train], [x.name for x in X_test]),
               feature_selection=args.feature_select)
        predictions = clf.predict(topredict)

        print(sum(predictions))
        print(len(predictions) - sum(predictions))

else:
    full_table = filter_by_metrics(features, metric_label, metric_value, labels, threshold)
    pos = full_table[full_table[metric_label] == 1].index
    neg = full_table[full_table[metric_label] == 0].index
    pos_samples = len(list(pos))
    neg_samples = len(list(neg))
    print("Positive samples: " + str(pos_samples))
    print("Negative samples : " + str(neg_samples))
    #balance_by_template(full_table, metric_label)

    # under sampling
    if args.balance:
        common_indices = balance(full_table, metric_label)
    else:
        common_indices = list(full_table.index.tolist())

    # split
    if split_by_class is not None:
        import json
        classfile=json.load(open(split_by_class))
        results=[]
        xlabels = []
        for c in classfile:
            X_train, X_test, Y_train, Y_test, train_ind, test_ind = test_train_split_class(full_table,
                                                                                           common_indices,
                                                                                           features,
                                                                                           metric_label,
                                                                                           prog_map,
                                                                                           c.keys()[0],
                                                                                           c)
            xlabels.append(str(c.keys()[0]))
            results.append(run_rf(X_train, Y_train, X_test, Y_test, list(features), cv=args.cv,
                                  cv_template=args.cv_template, train_test_indices=(train_ind, test_ind)))
        font = {'family': 'normal',
                'size': 20}

        matplotlib.rc('font', **font)
        #plt.figure(figsize=(12, 8))
        barwdith = 0.25
        rng = range(1, len(xlabels) + 1)
        plt.bar(rng, [x["Recall"] for x in results], label='Recall', width=barwdith)
        plt.bar([x + barwdith for x in rng], [x["Precision"] for x in results], label='Precision', width=barwdith)
        #plt.plot(threshold, [x["Diff"] for x in results], label='Diff', marker='s')
        #plt.plot(range(1, len(xlabels) + 1), [x["Accuracy"] for x in results], label='Accuracy', marker='s')
        plt.bar([x + 2*barwdith for x in rng], [x["F1"] for x in results], label='F1', width=barwdith)
        plt.xticks([x + barwdith for x in rng], xlabels, rotation=20)
        plt.grid(True)
        #plt.ylim((0.5, 1.0))
        #plt.xlabel("Mixture Model Subclasses")
        plt.ylabel("Scores")
        plt.legend(loc='center',  bbox_to_anchor=(0.5,1.11),  ncol=3, prop={'size': 16})
        #plt.imshow(aspect='auto')
        #plt.tight_layout()
        plt.show()
        exit(1)

    elif split_by_template:
        X_train, X_test, Y_train, Y_test, train_ind, test_ind = test_train_split_template(full_table,
                                                                                          common_indices,
                                                                                          features,
                                                                                          metric_value
                                                                                          if algorithm == 'rf_reg'
                                                                                          else metric_label,
                                                                                          test_train_split if args.split_template_name is None else args.split_template_name,
                                                                                          shuffle,
                                                                                          prog_map)
    else:
        # temp=['progs20190330-174717050219_prob_rand_188']
        # print(common_indices.index(temp[0]))
        # ind = filter(lambda x: sum([i in x for i in temp]) > 0, common_indices)
        # #print(features.ix[ind])
        #######################################
        #special_index='progs20190621-205408227197_prob_rand_196'
        #special_index='progs20190621-212844701711_prob_rand_554'
        if args.special_index is not None:
            common_indices.remove(args.special_index)

        ######################################33
        X = features.ix[common_indices]

        if algorithm == 'rf_reg' or args.stratify_data:
            #Y = [full_table.ix[i][metric_value] for i in common_indices]
            Y = full_table.ix[common_indices][metric_value]
        else:
            #Y = [full_table.ix[i][metric_label] for i in common_indices]
            Y = full_table.ix[common_indices][metric_label]

        print(np.shape(X))
        print(np.shape(Y))
        X_train, X_test, Y_train, Y_test = split_dataset(X, Y,
                                                         label,
                                                         ratio=test_train_split,
                                                         shuffle=shuffle,
                                                         stratify_data=args.stratify_data,
                                                         threshold=threshold if algorithm != 'rf_reg' else None)
        # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=test_train_split,
        #                                                     test_size=1.0 - test_train_split)
        if args.special_index is not None:
            X_test=X_test.append(features.ix[args.special_index])
            print('data',features.ix[args.special_index])
            Y_test=Y_test.append(pd.Series({args.special_index: full_table.ix[args.special_index][metric_label]}))

    get_class_dist(X_train, X_test)

    if algorithm == 'stan':
        run_stan(X_train, Y_train, X_test, Y_test)
    elif algorithm == 'rf':
        run_rf(X_train, Y_train, X_test, Y_test, list(features), cv=args.cv, cv_template=args.cv_template, gridsearch=args.grid, learning=args.learning, validation=args.validation, tree=args.tree, train_test_indices=(list(X_train.index), list(X_test.index)), feature_selection=args.feature_select)
    elif algorithm == 'dt':
        run_dtree(X_train, Y_train, X_test, Y_test, list(features))
    elif algorithm == 'maj':
        run_majority(X_train, Y_train, X_test, Y_test)
    elif algorithm == 'xgb':
        run_xgboost(X_train, Y_train, X_test, Y_test, list(features), args.cv, args.grid)
    elif algorithm == 'rf_reg':
        run_rf_regression(X_train, Y_train, X_test, Y_test)

# write to csv
time = time.time() - start_time
fields.append(time)
with open('precision_timeseries.csv', 'a') as f:
    writer = csv.writer(f)
    writer.writerow(fields)