#!/usr/bin/env Rscript
library(rstan)
N <- 1000
mu <- c(-0.75, 0.75);
sigma <- c(1, 1);
lambda <- 0.4
z <- rbinom(N, 1, lambda) + 1;
y <- rnorm(N, mu[z], sigma[z]);
theta <- 0.25

stan_rdump(c("N", "y", "theta"), file="gauss_mix_given_theta.data.R")
