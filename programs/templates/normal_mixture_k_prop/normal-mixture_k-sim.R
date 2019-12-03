library("rstan");
source("normal_mixture_k_prop_gen_data.R")
stan_rdump(c("K","N","y"),file="normal_mixture_k_prop.data.R")
#fit <- stan(file="normal_mixture_k.stan", data=c("K","N","y"), iter=1000, chains=1, init=0);
