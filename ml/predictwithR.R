library(rstan)
features=read.csv(file='feature.csv', header=TRUE, sep=',')
labels=read.csv(file='all_metrics_on_galeb.csv', header=TRUE, sep=',')
V=ncol(features)
N=nrow(features)
X_train=features
X_test
