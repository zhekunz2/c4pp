
data {
  int<lower=0> NP; 
  int<lower=0> N_uc;
  int<lower=0> N_rc;
  real<lower=0> t_uc[N_uc]; 
  real<lower=0> t_rc[N_rc]; 
  int disease_uc[N_uc]; 
  int disease_rc[N_rc]; 
  int patient_uc[N_uc]; 
  int patient_rc[N_rc]; 
  int sex_uc[N_uc]; 
  int sex_rc[N_rc]; 
  int age_uc[N_uc]; 
  int age_rc[N_rc]; 
} 
parameters {
  real alpha; 
  real beta_age;
  real beta_sex;
  real beta_disease2; 
  real beta_disease3; 
  real beta_disease4; 
  real<lower=0> r; 
  real<lower=0> tau;
  real b[NP]; 
} 

transformed parameters {
  real sigma;
  real yabeta_disease[4];
  yabeta_disease[1] <- 0; 
  yabeta_disease[2] <- beta_disease2;
  yabeta_disease[3] <- beta_disease3;
  yabeta_disease[4] <- beta_disease4;
  sigma <- sqrt(1 / tau); 
}

model {  
  alpha ~ normal(0, 100); 
  beta_age ~ normal(0, 100); 
  beta_sex ~ normal(0, 100);
  beta_disease2 ~ normal(0, 100); 
  beta_disease3 ~ normal(0, 100); 
  beta_disease4 ~ normal(0, 100); 

  tau ~ gamma(0.001, 0.001);
  r ~ gamma(1, 0.001); 

  for (i in 1:NP) b[i] ~ normal(0, sigma);   
  for (i in 1:N_uc) {
    t_uc[i] ~ weibull(r, exp(-(alpha + beta_age * age_uc[i] + beta_sex * sex_uc[i] +
                               yabeta_disease[disease_uc[i]] + b[patient_uc[i]]) / r));
  } 
  for (i in 1:N_rc) {
    1 ~ bernoulli(exp(-pow(t_rc[i] / exp(-(alpha + beta_age * age_rc[i] + beta_sex * sex_rc[i] 
                                         + yabeta_disease[disease_rc[i]] + b[patient_rc[i]]) / r), r)));



  }
}

generated quantities {

}
