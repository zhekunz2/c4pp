data{
  int<lower=1>    t_max;          
  vector[t_max]       y;         

  vector[12]         m0;         
  cov_matrix[12]     C0;           
}

parameters{
  real              x0_mu;          
  vector[11]        x0_gamma;      
  vector[t_max]      x_mu;         
  vector[t_max]      x_gamma;       

  real<lower=0>      W_mu;        
  real<lower=0>      W_gamma;       
  cov_matrix[1]      V;            
}

model{
  for (t in 1:t_max){
    y[t] ~ normal(x_mu[t] + x_gamma[t], sqrt(V[1, 1]));
  }

  x0_mu ~ normal(m0[1], sqrt(C0[1, 1]));

    x_mu[1] ~ normal(x0_mu     , sqrt(W_mu));
  for(t in 2:t_max){
    x_mu[t] ~ normal( x_mu[t-1], sqrt(W_mu));
  }
  for (p in 1:11){
    x0_gamma[p] ~ normal(m0[p+1], sqrt(C0[(p+1), (p+1)]));
  }

  x_gamma[1] ~ normal(-sum(x0_gamma[1:11]),sqrt(W_gamma));
  for(t in 2:11){
    x_gamma[t] ~ normal(-sum(x0_gamma[t:11])-sum(x_gamma[1:(t-1)]),sqrt(W_gamma));
  }
  for(t in 12:t_max){
    x_gamma[t] ~ normal(-sum(x_gamma[(t-11):(t-1)]),sqrt(W_gamma));
  }
}
