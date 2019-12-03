





data {
  int<lower=0> N1;                int<lower=0> N;                 real Yvec1[N1];          real tvec1[N1];          int<lower=0> idxn1[N1];         real y0[N]; 
} 

transformed data {
  real y0_mean; 
  y0_mean <- mean(y0); 
} 

parameters {
  real<lower=0> sigmasq_y; 
  real<lower=0> sigmasq_alpha; 
  real<lower=0> sigmasq_beta; 
  real<lower=0> sigma_mu0; 
  real gamma; 
  real alpha0; 
  real beta0; 
  real theta; 
  real mu0[N]; 
  real alpha[N]; 
  real beta[N]; 
} 

  
transformed parameters {
  real<lower=0> sigma_y; 
  real<lower=0> sigma_alpha; 
  real<lower=0> sigma_beta; 
  sigma_y <- sqrt(sigmasq_y); 
  sigma_alpha <- sqrt(sigmasq_alpha); 
  sigma_beta <- sqrt(sigmasq_beta); 
}
 
model {
  int oldn; 
  real m[N1]; 
  for (n in 1:N1) {
    oldn <- idxn1[n]; 
    m[n] <- alpha[oldn] + beta[oldn] * (tvec1[n] - 6.5) + gamma * (mu0[oldn] - y0_mean);
  }
  Yvec1 ~ normal(m, sigma_y);

  mu0 ~ normal(theta, sigma_mu0); 
    
  for (n in 1:N) y0[n] ~ normal(mu0[n], sigma_y); 

  alpha ~ normal(alpha0, sigma_alpha); 
  beta ~ normal(beta0, sigma_beta); 

  sigmasq_y ~ inv_gamma(.001, .001); 
  sigmasq_alpha ~ inv_gamma(.001, .001); 
  sigmasq_beta ~ inv_gamma(.001, .001); 
  sigma_mu0 ~ inv_gamma(.001, .001); 
  
  alpha0 ~ normal(0, 1000); 
  beta0 ~ normal(0, 1000); 
  gamma ~ normal(0, 1000); 
  theta ~ normal(0, 1000); 
 
}
