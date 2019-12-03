

data {
  int<lower=0> nChild; 
  int<lower=0> nInd; 
  real gamma[nInd, 4];      // -1 if missing
  real delta[nInd]; 
  int<lower=0> ncat[nInd]; 
  int grade[nChild, nInd];  // -1 if missing
} 
parameters {
  real theta[nChild]; 
}
model { 
  real p[nChild, nInd, 5];
  real Q[nChild, nInd, 4];
  theta ~ normal(0.0, 36); 
  for (i in 1:nChild) {

    for (j in 1:nInd) {
    
      for (k in 1:(ncat[j] - 1))
        Q[i, j, k] <- inv_logit(delta[j] * (theta[i] - gamma[j, k])); 
      p[i, j, 1] <- 1 - Q[i, j, 1];
      for (k in 2:(ncat[j] - 1))  
        p[i, j, k] <- Q[i, j, k - 1] - Q[i, j, k];
      p[i, j, ncat[j]] <- Q[i, j, ncat[j] - 1];

      if (grade[i, j] != -1)
        increment_log_prob(log(p[i, j, grade[i, j]]));  
    }
  }
}



