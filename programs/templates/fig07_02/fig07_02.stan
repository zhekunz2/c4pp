data {
  int<lower=1> n;
  vector[n] y;
  vector[n] x;
  vector[n] w;
}
parameters {
  vector<lower=mean(y)-3*sd(y), upper=mean(y)+3*sd(y)>[n] mu;
  vector[n] seasonal;
  real beta;
  real lambda;

  positive_ordered[3] sigma;
}
transformed parameters {
  vector[n] yhat;
  yhat = mu + beta * x + lambda * w;
}
model {

  for(t in 12:n){
    vector[10] tmp;
    for(i in 1:11){
      tmp[i] = seasonal[t-12+i];
    }
    seasonal[t] ~ normal(- sum(tmp), sigma[1]);
  }

  for(t in 2:n)
    mu[t] ~ normal(mu[t-1], sigma[2]);

  y ~ normal(yhat + seasonal, sigma[3]);

  sigma ~ student_t(4, 0, 1);
}
