data {
  int<lower=1> n;
  vector[n] y;
}
parameters {
  vector[n] mu;
  vector[n] seasonal;
  real<lower=0> sigma_level;
  real<lower=0> sigma_seas;
  real<lower=0> sigma_irreg;
}
transformed parameters {
  vector[n] yhat;
  yhat = mu + seasonal;
}
model {
  for(t in 12:n){
    vector[10] tmp;
    for(i in 1:11){
      tmp[i] = seasonal[t-12+i];
    }
    seasonal[t] ~ normal(- sum(tmp), sigma_seas);
  }
  

  for(t in 2:n)
    mu[t] ~ normal(mu[t-1], sigma_level);

  y ~ normal(yhat, sigma_irreg);
}
