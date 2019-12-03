data{
  int<lower=1>    t_max;         
  matrix[1, t_max]    y;          

  matrix[12, 12]      G;         
  matrix[12,  1]      F;          
  vector[12]         m0;           
  cov_matrix[12]     C0;           
}

parameters{
  real<lower=0>       W_mu;        
  real<lower=0>       W_gamma;   
  cov_matrix[1]       V;        
}

transformed parameters{
    matrix[12, 12]    W;         
    for (k in 1:12){             
      for (j in 1:12){
             if (j == 1 && k == 1){ W[j, k] = W_mu;     }
        else if (j == 2 && k == 2){ W[j, k] = W_gamma;  }
        else{                       W[j, k] = 0;        }
      }
    }
}

model{
  y ~ gaussian_dlm_obs(F, G, V, W, m0, C0);
}
