



data {
  int<lower=1> N;
  int<lower=1> D;
  vector[D] x[N];
}
transformed data {
  matrix[N, N] K;
   vector[N] mu;
   K = cov_exp_quad(x, 1.0, 1.0);
   mu = rep_vector(0, N);
  for (n in 1:N)
    K[n, n] = K[n, n] + 0.1;
}
parameters {
  vector[N] y;
}
model {
  y ~ multi_normal(mu, K);
}