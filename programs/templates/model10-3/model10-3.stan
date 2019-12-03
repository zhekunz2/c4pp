data{
  int<lower=1>    t_max; 
  matrix[1, t_max]    y;  

  matrix[1, 1]    G; 
  matrix[1, 1]    F;     
  vector[1]      m0;     
  cov_matrix[1]  C0;  
}

parameters{
  cov_matrix[1]   W;     
  cov_matrix[1]   V;      
}

model{
  y ~ gaussian_dlm_obs(F, G, V, W, m0, C0);

}
