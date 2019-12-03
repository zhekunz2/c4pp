
data {
  int<lower=0> N;
  int<lower=0> NT;
  int<lower=0> obs_t[N]; 
  int<lower=0> t[NT + 1]; 
  int<lower=0> fail[N]; 
  real Z[N]; 
}

transformed data {
  int Y[N, NT];
  int dN[N, NT]; 
  real c;
  real r; 
  for(i in 1:N) {
    for(j in 1:NT) {
      Y[i, j] <- int_step(obs_t[i] - t[j] + .000000001);
      dN[i, j] <- Y[i, j] * fail[i] * int_step(t[j + 1] - obs_t[i] - .000000001);
    }
  }
  c <- 0.001; 
  r <- 0.1; 
}

parameters {
  real beta; 
  real<lower=0> dL0[NT]; 
} 

model {
  beta ~ normal(0, 1000);
  for(j in 1:NT) {
    dL0[j] ~ gamma(r * (t[j + 1] - t[j]) * c, c);
    for(i in 1:N) {
      dN[i, j] ~ poisson(Y[i, j] * exp(beta * Z[i]) * dL0[j]); 
    }     
  }
}
