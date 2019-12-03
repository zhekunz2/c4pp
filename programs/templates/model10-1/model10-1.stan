data{
  int<lower=1>   t_max;    
  vector[t_max]   y;       

  cov_matrix[1]   W;      
  cov_matrix[1]   V;      
  real           m0;     
  cov_matrix[1]  C0;    
}

parameters{
  real           x0;    
  vector[t_max]   x;  
}

model{
  for (t in 1:t_max){
    y[t] ~ normal(x[t], sqrt(V[1, 1]));
  }
  x0   ~ normal(m0, sqrt(C0[1, 1]));
  x[1] ~ normal(x0, sqrt(W[1, 1]));
  for (t in 2:t_max){
    x[t] ~ normal(x[t-1], sqrt(W[1, 1]));
  }
}
