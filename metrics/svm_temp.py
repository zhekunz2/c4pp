#!/usr/bin/env python3

# ./svm_temp.py dtree feature_csv1115_2/reduce.csv hell_result

from sklearn import svm
from sklearn.model_selection import train_test_split
import csv
import pandas as pd
import numpy as np
import sys
from confidentlearning.classification import RankPruning
from sklearn.tree.export import export_graphviz

#label = "rhat2_result"
label = sys.argv[-1]
label_pred = label
if len(sys.argv) > 4:
    test_size = 0
else:
    test_size = 0.1
seed_num = np.random.randint(0,100)
print(seed_num)
classifier=sys.argv[1]
feature_table = pd.read_csv(sys.argv[2])


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
    clf1 = DecisionTreeClassifier(criterion="entropy", max_depth=4, max_features="sqrt")
    clf2 = DecisionTreeClassifier(criterion="entropy", max_depth=4, max_features="sqrt")
    clf3 = DecisionTreeClassifier(criterion="entropy", max_depth=4, max_features="sqrt")
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
    clf = RandomForestClassifier(n_estimators=20)
elif classifier == "mlp":
    from sklearn.neural_network import MLPClassifier
    clf =  MLPClassifier(alpha=1)
elif classifier == "ada":
    from sklearn.ensemble import AdaBoostClassifier
    clf = AdaBoostClassifier()

# data cleaning and preprocessing
try:
    feature_table = feature_table[feature_table[label] != "Error"]
except:
    pass
if label not in list(feature_table):
    label = "results"
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
print(true_count)
print(false_count)
feature_table_true1 = feature_table.loc[feature_table[label] > 0].iloc[:1018]
feature_table_true2 = feature_table.loc[feature_table[label] > 0].iloc[1018:2036]
feature_table_true3 = feature_table.loc[feature_table[label] > 0].iloc[2036:]
print(len(feature_table_true1))
print(len(feature_table_true2))
print(len(feature_table_true3))
#feature_table_true = feature_table.loc[feature_table[label] > 0].sample(n = min(true_count, false_count))
feature_table1 = feature_table_true1.append(feature_table.loc[feature_table[label] == 0])
feature_table2 = feature_table_true2.append(feature_table.loc[feature_table[label] == 0])
feature_table3 = feature_table_true3.append(feature_table.loc[feature_table[label] == 0])
feature_table1 = feature_table1.sample(frac=1).reset_index(drop=True)
feature_table2 = feature_table2.sample(frac=1).reset_index(drop=True)
feature_table3 = feature_table3.sample(frac=1).reset_index(drop=True)
suppress = ["t_value", "t_result", "ks_value", "ks_result", "kl_value", "kl_result", "smkl_value", "smkl_result", "hell_value", "hell_result", "program", "value", "results", "rhat1_value", "rhat1_result", "rhat2_value", "rhat2_result", "2100_nuts_result", "2100_hmc_result", "2100_vb_result"]
for ll in list(feature_table):
    if "d_" in ll:
        suppress.append(ll)
y1 = feature_table1[label]
X1 = feature_table1.drop(columns=suppress, errors="ignore") \
        .fillna(value=0).replace(np.inf, 2**100)
y2 = feature_table2[label]
X2 = feature_table2.drop(columns=suppress, errors="ignore") \
        .fillna(value=0).replace(np.inf, 2**100)
y3 = feature_table3[label]
X3 = feature_table3.drop(columns=suppress, errors="ignore") \
        .fillna(value=0).replace(np.inf, 2**100)
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=test_size, random_state=seed_num)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=test_size, random_state=seed_num)
X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=test_size, random_state=seed_num)
#
#error = sum([pp[0] != pp[1] for pp in zip(predictions,y_test)])
#print("Error rate: {}".format(error/float(len(y_test))))
X_test = pd.concat([X_test1, X_test2, X_test3])
y_test = pd.concat([y_test1, y_test2, y_test3])

print(len(X_train1))
print(len(y_train1))
clf1.fit(np.array(X_train1), np.array(y_train1))
clf2.fit(np.array(X_train2), np.array(y_train2))
clf3.fit(np.array(X_train3), np.array(y_train3))
#print("Train True/False: {}/{}".format(sum(y_train == True), sum(y_train == False)))
print("Test True/False: {}/{}".format(sum(y_test == True), sum(y_test == False)))
if len(X_test != 0):
    predictions3 = clf3.predict(X_test)
    predictions2 = clf2.predict(X_test)
    predictions1 = clf1.predict(X_test)
    predictions = [sum(x) > 1.5 for x in zip(predictions1, predictions2, predictions3)]
    error = sum([pp[0] != pp[1] for pp in zip(predictions,y_test)])
    print("Error rate: {}/{} = {}".format(error, len(y_test), error/float(len(y_test))))
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
if len(sys.argv) > 4:
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

def show_coefficients(classifier, clf, feature_names, filename, top_features=20):
    if classifier == "svml":
        coef = clf.coef_.ravel()
    elif classifier == "rf":
        coef = clf.feature_importances_
    elif classifier == "dtree":
        export_graphviz(clf, out_file=(filename+'.dot'), feature_names=feature_names)
        coef = clf.feature_importances_
    else:
        return
    top_positive_coefficients = np.argsort(coef)[-top_features:][::-1]

    #top_negative_coefficients = np.argsort(coef)[:top_features]
    #top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    feature_names = np.array(feature_names)
    print(list(zip(feature_names[top_positive_coefficients], map(lambda x: x, sorted(coef, reverse=True)))))

show_coefficients(classifier, clf1, list(X_train1), "tree1")
show_coefficients(classifier, clf2, list(X_train2), "tree2")
show_coefficients(classifier, clf3, list(X_train3), "tree3")
