



data {
  int<lower=0> Nc; 
  int<lower=0> Ni; 
  int xc[Nc];
  int wc[Nc];
  int dc[Nc];
  int wi[Ni];
  int di[Ni];
} 

parameters {
  real<lower=0,upper=1> phi[2, 2];
  real<lower=0,upper=1> q; 
  real beta0C; 
  real beta; 

} 

model {
  for (n in 1:Nc) {
    xc[n] ~ bernoulli(q); 
    dc[n] ~ bernoulli_logit(beta0C + beta * xc[n]); 
    wc[n] ~ bernoulli(phi[xc[n] + 1, dc[n] + 1]); 
  } 
  for (n in 1:Ni) {

    di[n] ~ bernoulli(inv_logit(beta0C + beta) * q + inv_logit(beta0C) * (1 - q)); 
    wi[n] ~ bernoulli(phi[1, di[n] + 1] * (1 - q) + phi[2, di[n] + 1] * q); 
  } 
  q ~ uniform(0, 1); 
  beta0C ~ normal(0, 320); 
  beta ~ normal(0, 320);
  for (i in 1:2) 
    for (j in 1:2) 
      phi[i, j] ~ uniform(0, 1); 
} 

generated quantities {
  real gamma1; 
  real gamma2; 
    gamma1 <- 1 / (1 + (1 + exp(beta0C + beta)) / (1 + exp(beta0C)) * (1 - q) / q); 
  gamma2 <- 1 / (1 + (1 + exp(-beta0C - beta)) / (1 + exp(-beta0C)) * (1 - q) / q);
} 


