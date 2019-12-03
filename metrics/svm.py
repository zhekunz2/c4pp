#!/usr/bin/env python3.6

# ./svm.py rf feature_azure_batch_example-models_.csv [new_pred_file] iter_600_nuts_result cv
# ./svm.py rf feature_csv1115_2/reduce.csv hell_result 0.4 cv

from sklearn import svm
from sklearn.model_selection import train_test_split
#from sklearn import cross_validation
import csv
import pandas as pd
import numpy as np
import sys
#from confidentlearning.classification import RankPruning
from sklearn.tree.export import export_graphviz

class DevNull:
    def write(self, msg):
        pass
sys.stderr = DevNull()

#label = "rhat2_result"
if sys.argv[-1] != "cv":
    try:
        label_thres = float(sys.argv[-1])
        label = sys.argv[-2]
        label_value = sys.argv[-2].replace("results", "value")
    except:
        label = sys.argv[-1]
        label_thres = -1
else:
    try:
        label_thres = float(sys.argv[-2])
        label = sys.argv[-3]
        label_value = sys.argv[-3].replace("results", "value")
    except:
        label = sys.argv[-2]
        label_thres = -1


label_pred = label
if len(sys.argv) > 4 and sys.argv[-1] != "cv" and label_thres == -1:
    test_size = 0
else:
    test_size = 0.1
seed_num = np.random.randint(0,100)
print(seed_num)
classifier=sys.argv[1]
feature_table = pd.read_csv(sys.argv[2])


# data cleaning and preprocessing
try:
    feature_table = feature_table[feature_table[label] != "Error"]
except:
    pass
if label not in list(feature_table):
    label = "results"
iter_num = [nn for nn in label.replace('.','_').split('_') if nn.isdigit() and int(nn) % 100 == 0 and int(nn) != 0]
if len(iter_num) > 0:
    iters = ["iter_{}_{}_result".format(ii, label.replace('.','_').split('_')[2]) for ii in range(100, int(iter_num[0]) + 1, 500)]
    feature_table["new_iter_result"] = feature_table[iters].apply(lambda x: np.prod([1 if xx == "True" else 0 if xx == "False" else xx for xx in x]), axis=1)
    label = "new_iter_result"
#if "value" not in list(feature_table):
#    res_split = [ff.split(":") for ff in \
#            feature_table["results"]]
#    feature_table["results"], feature_table["value"] = \
#            zip(*res_split)
feature_table.dropna(how='all', inplace=True, subset=[label])
feature_table[label] = feature_table[label].replace("True", 1)\
        .replace("False", 0)
true_count = len(feature_table.loc[feature_table[label] > 0])
false_count = len(feature_table.loc[feature_table[label] == 0])
print(true_count+false_count)
#feature_table_true = feature_table.loc[feature_table[label] > 0].sample(n = min(true_count, false_count))
#feature_table = feature_table_true.append(feature_table.loc[feature_table[label] == 0].sample(n = min(true_count, false_count)))
#feature_table = feature_table.sample(frac=1).reset_index(drop=True)
if label_thres != -1:
    if "t_result" in label or "ks_result" in label:
        feature_table[label] = abs(feature_table[label_value]) > abs(label_thres)
    else:
        feature_table[label] = abs(feature_table[label_value]) < abs(label_thres)
feature_table.sort_values(by="program", inplace=True)
feature_table.drop_duplicates(subset ="program", inplace = True)

pos = feature_table[feature_table[label] == 1].index
neg = feature_table[feature_table[label] == 0].index
pos_samples =len(list(pos))
neg_samples = len(list(neg))
print("pos"+str(pos_samples))
print("neg" + str(neg_samples))
if pos_samples < neg_samples:
    new_neg_samples = feature_table[feature_table[label] == 0].sample(pos_samples).index
    common_indices = list(pos) + list(new_neg_samples)
else:
    new_pos_samples = feature_table[feature_table[label] == 1].sample(neg_samples).index
    common_indices = list(neg) + list(new_pos_samples)
feature_table=feature_table.ix[common_indices]

y = feature_table[label]
suppress = ["t_value", "t_result", "ks_value", "ks_result", "kl_value", "kl_result", "smkl_value", "smkl_result", "hell_value", "hell_result",  "value", "program", "results", "rhat1_value", "rhat1_result", "rhat2_value", "rhat2_result", "2100_nuts_result", "2100_hmc_result", "2100_vb_result"]
for ll in list(feature_table):
    if "results" in ll or "value" in ll:
        suppress.append(ll)
#    if "d_" in ll:
#        suppress.append(ll)
print('hi')


X = feature_table.drop(columns=suppress, errors="ignore")\
        .fillna(value=0).replace(np.inf, 9999999)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)#, random_state=seed_num)
pos=sum(y_train == True)
neg=sum(y_train == False)
print(pos)
print(neg)
if pos < neg:
    sample_weights = [float((pos+0.0)/neg) if x == False else 1 for x in y_train]
else:
    sample_weights = [float((neg+0.0)/pos) if x == True else 1 for x in y_train]
#print(sample_weights)
#error = sum([pp[0] != pp[1] for pp in zip(predictions,y_test)])
#print("Error rate: {}".format(error/float(len(y_test))))

if classifier == "svm":
    from sklearn.svm import SVC
    #clf = svm.SVC(kernel="linear", gamma=1, C=0.025)
    clf = SVC(probability=True, gamma="auto")
if classifier == "svml":
    from sklearn.svm import LinearSVC
    #clf = svm.SVC(kernel="linear", gamma=1, C=0.025)
    clf = LinearSVC()
elif classifier == "dtree":
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(criterion="entropy", max_depth=4)
elif classifier == "logistic":
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
elif classifier == "knn":
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
elif classifier == "lda":
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    clf = LinearDiscriminantAnalysis()
elif classifier == "nb":
    from sklearn.naive_bayes import GaussianNB
    clf = GaussianNB()
elif classifier == "rf":
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    if sys.argv[-1] == "cv":
        best_k, best_score = -1, -1
        clfs = {}
        for k in [20, 40, 60, 80, 100]:
            pipe = Pipeline([['sc', StandardScaler()], ['clf', RandomForestClassifier(n_estimators=k)]])
            pipe.fit(X_train, y_train)
            scores = cross_val_score(pipe, X_train, y_train, cv=10)
            print('rf-n_est={}\nValidation accuracy: {}'.format(k, scores.mean()))
            if scores.mean() > best_score:
                best_k, best_score = k, scores.mean()
            clfs[k] = pipe
            clf = clfs[best_k]
    else:
        clf = RandomForestClassifier(n_estimators=20)
elif classifier == "mlp":
    from sklearn.neural_network import MLPClassifier
    clf =  MLPClassifier(alpha=1)
elif classifier == "ada":
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier()


fit=clf.fit(np.array(X_train), np.array(y_train))

print("Train True/False: {}/{}".format(sum(y_train == True), sum(y_train == False)))
print("Test True/False: {}/{}".format(sum(y_test == True), sum(y_test == False)))
if len(X_test != 0):
    predictions = clf.predict(X_test)
    error = sum([pp[0] != pp[1] for pp in zip(predictions,y_test)])
    correct = sum([pp[0] == pp[1] for pp in zip(predictions,y_test)])
    #print("Accuracy score: {}".format(accuracy_score(y_test, predictions)))
    print("Error rate: {}/{} = {}".format(error, len(y_test), error/float(len(y_test))))
    print("Accuracy: {}/{} = {}".format(correct, len(y_test), correct/float(len(y_test))))
    TP = sum([pp[0] == pp[1] and pp[1] for pp in zip(predictions,y_test)])
    TN = sum([pp[0] == pp[1] and not pp[1] for pp in zip(predictions,y_test)])
    FP = sum([pp[0] != pp[1] and pp[0] for pp in zip(predictions,y_test)])
    FN = sum([pp[0] != pp[1] and not pp[0] for pp in zip(predictions,y_test)])
    print("Precision: {}/{} = {}".format(TP, TP + FP, TP/float(TP + FP)))
    print("Recall   : {}/{} = {}".format(TP, TP + FN, TP/float(TP + FN)))
    print("F1 score :       = {}".format(2/(1/(TP/float(TP + FP))+1/(TP/float(TP + FN)))))
    #print("Noisy:")
    #rf = RankPruning(clf=clf, seed=seed_num)
    #rf.fit(np.array(X_train), np.array(y_train))
    #predictions = rf.predict(X_test)
    #error = sum([pp[0] != pp[1] for pp in zip(predictions,y_test)])
    #print("Error rate: {}/{} = {}".format(error, len(y_test), error/float(len(y_test))))
if len(sys.argv) > 4 and sys.argv[-1] != "cv" and label_thres == -1:
    X_pred = pd.read_csv(sys.argv[3])
    if label_pred not in list(X_pred):
        label_pred = "results"
    # Get missing columns in the training test
    missing_cols = set(X_train.columns) - set(X_pred.columns )
    # Add a missing column in test set with default value equal to 0
    for mm in missing_cols:
        X_pred[mm] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    X_pred = X_pred[X_train.columns].fillna(value=0).replace(np.inf, 2**100)
    predictions = clf.predict(X_pred)
    y_test = pd.read_csv(sys.argv[3])[label_pred]
    #error = sum([pp[0] != pp[1] for pp in zip(predictions,y_test)])
    #print("Error rate: {}/{} = {}".format(error, len(y_test), error/float(len(y_test))))
    print("Error rate: {}/{} = {}".format(error, len(y_test), error/float(len(y_test))))
    print("Accuracy: {}/{} = {}".format(correct, len(y_test), correct/float(len(y_test))))
    TP = sum([pp[0] == pp[1] and pp[1] for pp in zip(predictions,y_test)])
    TN = sum([pp[0] == pp[1] and not pp[1] for pp in zip(predictions,y_test)])
    FP = sum([pp[0] != pp[1] and pp[0] for pp in zip(predictions,y_test)])
    FN = sum([pp[0] != pp[1] and not pp[0] for pp in zip(predictions,y_test)])
    print("Precision: {}/{} = {}".format(TP, TP + FP, TP/float(TP + FP)))
    print("Recall   : {}/{} = {}".format(TP, TP + FN, TP/float(TP + FN)))
    print("F1 score :       = {}".format(2/(1/(TP/float(TP + FP))+1/(TP/float(TP + FN)))))

    #print(list(predictions))
    #print("Noisy:")
    #rf = RankPruning(clf=clf, seed=seed_num)
    #rf.fit(np.array(X_train), np.array(y_train))
    #predictions = rf.predict(X_pred)
    #error = sum([pp[0] != pp[1] for pp in zip(predictions,y_test)])
    #print("Error rate: {}/{} = {}".format(error, len(y_test), error/float(len(y_test))))

def show_coefficients(classifier, clf, feature_names, top_features=20):
    if classifier == "svml":
        coef = clf.coef_.ravel()
    elif classifier == "rf" and sys.argv[-1] != "cv":
        coef = clf.feature_importances_
    elif classifier == "dtree":
        export_graphviz(clf, out_file='tree.dot', feature_names=feature_names)
        coef = clf.feature_importances_
    else:
        return
    top_positive_coefficients = np.argsort(coef)[-top_features:][::-1]

    #top_negative_coefficients = np.argsort(coef)[:top_features]
    #top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    feature_names = np.array(feature_names)
    print(list(zip(feature_names[top_positive_coefficients], map(lambda x: x, sorted(coef, reverse=True)))))

show_coefficients(classifier, clf, list(X_train))
