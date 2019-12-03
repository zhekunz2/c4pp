


data {
  int<lower=0> N; 
  real y[N]; 
} 

parameters {

  real<lower=2,upper= 100> d; 
} 

model {
  y ~ student_t(d, 0, 1); 

} 
