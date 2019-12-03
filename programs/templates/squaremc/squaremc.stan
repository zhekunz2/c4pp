
parameters {
  real<lower=-1,upper=1> x_raw;
  real<lower = -(1 - sqrt(1 - square(1 - fabs(x_raw)))),
       upper =  (1 - sqrt(1 - square(1 - fabs(x_raw))))> y_raw;
}
transformed parameters {
  real<lower=-1,upper=1> x;
  real<lower=-1,upper=1> y;
  x <- if_else(x_raw > 0, 1, -1) - x_raw;
  y <- if_else(y_raw > 0, 1, -1) - y_raw;
}
model {
  increment_log_prob(log1m(sqrt(1 - square(1 - fabs(x_raw)))));
}
