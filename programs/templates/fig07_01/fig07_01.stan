data {
  int<lower=1> n;
  vector[n] y;
  vector[n] x;
  vector[n] w;
}
parameters {
  real mu;
  vector[11] seas;
  real beta;
  real lambda;
  real<lower=0> sigma_irreg;
}
transformed parameters {
  vector[n] seasonal;
  vector[n] yhat;
  for(t in 1:11) {
    seasonal[t] = seas[t];
  }
  for(t in 12:n) {
    vector[10] tmp;
    for(i in 1:11){
      tmp[i] = seasonal[t-12+i];
    }
    seasonal[t] = - sum(tmp);
  }

  yhat = mu + beta * x + lambda * w;
}
model {
  y ~ normal(yhat + seasonal, sigma_irreg);
}
