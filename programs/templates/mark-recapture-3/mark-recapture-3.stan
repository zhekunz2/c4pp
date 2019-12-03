
data {
  int<lower=0> M;
  int<lower=0> C;
  int<lower=0,upper=min(M,C)> R;
}
transformed data {
  real theta_max;
  theta_max <- M;         
  theta_max <- theta_max / (C - R + M);
}
parameters {
  real<lower=(C - R + M)> N;
}
transformed parameters {
  real<lower=0,upper=theta_max> theta;
  theta <- M / N;
}
model {
  increment_log_prob(-2 * log(N));
  R ~ binomial(C, theta);
}
