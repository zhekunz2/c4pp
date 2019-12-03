data{
 int N;
 int V;
 int N_test;
 int Y_train[N];
 int Y_test[N_test];
 matrix[N,V] X_train;
 matrix[N_test, V] X_test;
}
parameters{
  vector<lower=0>[V] coeff;
  vector<lower=0>[N] noise;
}
transformed data{

}
model{
  noise ~ normal(0, 1);
  coeff ~ normal(0, 1);
  for(i in 1:N){
    Y_train[i] ~ bernoulli_logit(X_train[i]*coeff + noise);
  }
}
generated quantities{
   int y_test[N_test];
   for(i in 1:N_test){
   y_test[i]  = bernoulli_rng(inv_logit(X_test[i]*coeff + noise));
   }
}