data {
  int<lower=0> N;   int<lower=0> Npts;   int<lower=0> rat[Npts];   real x[Npts];
  real y[Npts];
  real xbar;
}

parameters {
  real alpha[N];
  real beta[N];

  real mu_alpha;
  real mu_beta;          // beta.c in original bugs model
  real<lower=0> sigma_y;       // sigma in original bugs model
  real<lower=0> sigma_alpha;
  real<lower=0> sigma_beta;
}


model {
  mu_alpha ~ normal(0, 100);
  mu_beta ~ normal(0, 100);
    alpha ~ normal(mu_alpha, sigma_alpha); // vectorized
  beta ~ normal(mu_beta, sigma_beta);  // vectorized
  for (n in 1:Npts){
    int irat;
    irat <- rat[n];
    y[n] ~ normal(alpha[irat] + beta[irat] * (x[n] - xbar), sigma_y);
  }
}
generated quantities {
  real alpha0;
  alpha0 <- mu_alpha - xbar * mu_beta;
}
