import xgboost
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from utils import *
import sys
import argparse
import re
import matplotlib.pyplot as plt

ignore_indices = []

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', nargs='+', dest='feature_file')
    parser.add_argument('-fo', dest='features_original')
    parser.add_argument('-l', dest='labels')
    parser.add_argument('-pf', dest='predict_feature')
    parser.add_argument('-pl', dest='predict_label')
    args = parser.parse_args()
    print(args)
    return args


def clean(f):
    f.index = f.index.map(lambda x: x.split('/')[-2] if len(x.split('/')) > 1 else x)
    for pat in ignore_indices:
        f = f.filter(set(f).difference(set(f.filter(regex=(pat)))))
    f = f.replace('inf', np.inf)
    f = f.fillna(0).replace(np.inf, 99999999).replace(-np.inf, -99999999)

    return f


def predict_combine(clfs, X_test, Y_test, weighted=False):
    stats = dict()
    stats["True"] = sum([x == 1 for x in Y_test])
    stats["False"] = sum([x == 0 for x in Y_test])
    print("True:" + str(stats["True"]))
    print("False:" + str(stats["False"]))
    stats["Diff"] = (stats["True"] + 0.0) / len(Y_test)
    predictions = []
    prediction_probs = []
    for clf in clfs:
        predictions.append(clf.predict(X_test))
        prediction_probs.append(clf.predict(X_test))

    combined_predictions = [np.sum([p[pr] for p in predictions]) >= 1 for pr in range(len(predictions[0]))]

    error = sum([pp[0] != pp[1] for pp in zip(combined_predictions, Y_test)])
    correct = sum([pp[0] == pp[1] for pp in zip(combined_predictions, Y_test)])
    stats["Error"] = error / float(len(Y_test))
    stats["Accuracy"] = correct / float(len(Y_test))
    # print("Accuracy score: {}".format(accuracy_score(y_test, predictions)))
    print("Error rate: {}/{} = {}".format(error, len(Y_test), stats["Error"]))
    print("Accuracy: {}/{} = {}".format(correct, len(Y_test), stats["Accuracy"]))
    TP = sum([pp[0] == pp[1] and pp[1] for pp in zip(combined_predictions, Y_test)])
    TN = sum([pp[0] == pp[1] and not pp[1] for pp in zip(combined_predictions, Y_test)])
    FP = sum([pp[0] != pp[1] and pp[0] for pp in zip(combined_predictions, Y_test)])
    FN = sum([pp[0] != pp[1] and not pp[0] for pp in zip(combined_predictions, Y_test)])
    stats["TP"] = TP
    stats["TN"] = TN
    stats["FP"] = FP
    stats["FN"] = FN
    # stats["Precision"] = TP / float(TP + FP)
    # stats["Recall"] = TP / float(TP + FN)
    stats["Precision"] = metrics.precision_score(Y_test, combined_predictions, average='weighted' if weighted else 'binary')
    stats["Recall"] = metrics.recall_score(Y_test, combined_predictions, average='weighted' if weighted else 'binary')
    # stats["F1"] = 2 / (1 / (TP / float(TP + FP)) + 1 / (TP / float(TP + FN)))
    stats["F1"] = metrics.f1_score(Y_test, combined_predictions, average='weighted' if weighted else 'binary')
    #stats["AUC"] = roc_auc_score(Y_test, prediction_probs[:, 1], average="weighted" if weighted else 'macro')

    # fpr, tpr, thresholds = roc_curve([ int(x) for x in Y_test], prediction_probs[:,1], pos_label=1)
    # plot_roc_curve(fpr, tpr)

    print("TP : ", TP)
    print("TN : ", TN)
    print("FP : ", FP)
    print("FN : ", FN)
    print("Precision: {}/{} = {}".format(TP, TP + FP, stats["Precision"]))
    print("Recall   : {}/{} = {}".format(TP, TP + FN, stats["Recall"]))
    print("F1 score :       = {}".format(stats["F1"]))
    #print("AUC: = {}".format(stats["AUC"]))



def predict(clf, X_test, Y_test, weighted=False):
    stats = dict()
    stats["True"] = sum([x == 1 for x in Y_test])
    stats["False"] = sum([x == 0 for x in Y_test])
    print("True:" +str(stats["True"]))
    print("False:" + str(stats["False"]))
    stats["Diff"] = (stats["True"]+0.0)/len(Y_test)

    predictions = clf.predict(X_test)
    if isinstance(clf, xgboost.XGBClassifier):
        predictions = [round(v) for v in predictions]

    prediction_probs = clf.predict_proba(X_test)

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
    stats["AUC"] = roc_auc_score(Y_test, prediction_probs[:, 1], average="weighted" if weighted else 'macro')

    # fpr, tpr, thresholds = roc_curve([ int(x) for x in Y_test], prediction_probs[:,1], pos_label=1)
    # plot_roc_curve(fpr, tpr)

    print("TP : ", TP)
    print("TN : ", TN)
    print("FP : ", FP)
    print("FN : ", FN)
    print("Precision: {}/{} = {}".format(TP, TP + FP, stats["Precision"]))
    print("Recall   : {}/{} = {}".format(TP, TP + FN, stats["Recall"]))
    print("F1 score :       = {}".format(stats["F1"]))
    print("AUC: = {}".format(stats["AUC"]))

    return stats


args = read_args()
mutant_features = clean(read_all_csvs(args.feature_file, None, index='program'))
original_features = clean(read_all_csvs(args.features_original, None, index='program'))
labels = clean(read_all_csvs(args.labels, None, index='program'))

common_indices = set(mutant_features.index.tolist()).intersection(set(labels.index.tolist()))

original_features = original_features.ix[common_indices]
mutant_features = mutant_features.ix[common_indices]
labels = labels.ix[common_indices]


predict_feature = clean(read_all_csvs(args.predict_feature, None, index='program'))
predict_label = clean(read_all_csvs(args.predict_label, None, index='program'))

common_indices_predict = set(predict_feature.index.tolist()).intersection(set(predict_label.index.tolist()))

predict_feature = predict_feature.ix[common_indices_predict]
predict_label = predict_label.ix[common_indices_predict]
missing=set(mutant_features).difference(set(predict_feature))
for m in missing:
    predict_feature[m] = 0.0

feature_order = list(mutant_features)
predict_feature = predict_feature[feature_order]

mutation_feature_labels = []
for fx in list(original_features):
    if 'A' <= fx[0] <= 'Z':
        mutation_feature_labels.append(fx)
orig = []
muta = []
comb = []
hardness = []

trials = 5
metric = 'F1'
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 100.0]


for th in thresholds:
    print("Using original features...")
    X_train, X_test, Y_train, Y_test = train_test_split(original_features,
                                                        [1 if x < th else 0 for x in labels['wass_value_avg']],
                                                        train_size=0.8)

    # scores = []
    # for i in range(trials):
    #     clf1 = RandomForestClassifier(bootstrap=True, n_estimators=50)
    #     clf1.fit(X_train, Y_train)
    #     scores.append(predict(clf1, X_test, Y_test)[metric])
    # orig.append(np.mean(scores))

    print("Using mutant features...")
    X_train, X_test, Y_train, Y_test = train_test_split(mutant_features, [1 if x < th else 0 for x in labels['wass_value_avg']], train_size=0.8)

    scores = []
    scores_hard = []


    for i in range(trials):
        clf2 = RandomForestClassifier(**{'bootstrap': False, 'n_estimators': 50, 'min_samples_split': 6, 'criterion': 'gini', 'max_features': 10, 'max_depth': None, 'class_weight': 'balanced'})
        # param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic', 'n_jobs': -1}
        # clf2 = xgboost.XGBClassifier(**param)
        clf2.fit(X_train, Y_train)
        scores.append(predict(clf2, X_test, Y_test)[metric])
        print("Predicting hardness out-of-box set")
        scores_hard.append(
            predict(clf2, predict_feature, [1 if x < th else 0 for x in predict_label['wass_value_avg']], weighted=True)[metric])

    muta.append(np.mean(scores))
    hardness.append(np.mean(scores_hard))


    # print("Using combined features...")
    # #predict_combine([clf1, clf2], X_test, Y_test)
    #
    # #print("Merging mutation info")
    #
    # combined_features = pd.merge(mutant_features, original_features[mutation_feature_labels], how='left', left_index=True, right_index=True)
    # X_train, X_test, Y_train, Y_test = train_test_split(combined_features, [1 if x < th else 0 for x in labels['wass_value_avg']], train_size=0.8)
    # #print(combined_features.head(2))
    # scores = []
    # for i in range(trials):
    #     clf3 = RandomForestClassifier(bootstrap=True, n_estimators=50)
    #     clf3.fit(X_train, Y_train)
    #     scores.append(predict(clf3, X_test, Y_test)[metric])
    # comb.append(np.mean(scores))


#plt.plot(thresholds, orig, label='orig')
plt.plot(thresholds, muta, label='muta')
#plt.plot(thresholds, comb, label='comb')
plt.plot(thresholds, hardness, label='hardness')
plt.xlabel(thresholds)
plt.ylabel(metric)
plt.legend()
plt.show()