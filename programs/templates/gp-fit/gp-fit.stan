data{
vector [101] y;
real x[101];
int N;
}
transformed data{
vector [ N ] mu ;
mu = rep_vector(0,N);
}
parameters {
real<lower=0> rho;
real<lower=0> alpha;
real<lower=0> sigma;
}
model {
matrix [ N,N ] L_K ;
matrix [ N,N ] K ;
real sq_sigma ;
K = cov_exp_quad(x,alpha,rho);
sq_sigma = square(sigma);
for(n in 1:N){
K[n,n] = K[n,n]+sq_sigma;
}
L_K = cholesky_decompose(K);
rho ~ inv_gamma(5.0,5.0);
alpha ~ normal(0.0,1.0);
sigma ~ normal(0.0,1.0);
y ~ multi_normal_cholesky(mu,L_K);
}
