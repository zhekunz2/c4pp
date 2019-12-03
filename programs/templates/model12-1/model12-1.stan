functions{
  vector dlm_local_level_loglik(int n,
                                vector y, int[] miss,
                                vector V, vector W,
                                real m0, real C0) {
    // prior of state: p(theta | y_t, ..., y_{t-1})
    real a;
    real R;
    // likelihood of obs: p(y | y_t, ..., y_t-1)
    real f;
    real Q;
    real Q_inv;
    // posterior of states: p(theta_t | y_t, ..., y_t)
    real m;
    real C;
    // forecast error
    real e;
    // Kalman gain
    real K;
    // returned data
    vector[n] loglik;

    m = m0;
    C = C0;
    for (t in 1:n) {
      // PREDICT STATES
      // one step ahead predictive distribion of p(\theta_t | y_{1:(t-1)})
      a = m;
      R = C + W[t];
      m = a;
      C = R;
      if (int_step(miss[t])) {
        e = 0.0;
        Q_inv = 0.0;
      } else {
        // PREDICT OBS
        // one step ahead predictive distribion of p(y_t | y_{1:(t-1)})
        f = m;
        Q = C + V[t];
        Q_inv = 1.0 / Q;
        // forecast error
        e = y[t] - f;
        // Kalman gain
        K = C * Q_inv;
        // FILTER STATES
        // posterior distribution of p(\theta_t | y_{1:t})
        m = m + K * e;
        C = C - pow(K, 2) *  Q;
        loglik[t] = - 0.5 * (log(2 * pi()) + log(Q) + pow(e, 2) * Q_inv);
      }
    }
    return loglik;
  }
}
data{
  int<lower = 1>              t_max;  // Time series length
  vector[t_max]                   y;  // Observations
  
  int                   miss[t_max];  // Indicator for missing values
  real                           m0;  // Mean of prior distribution
  real<lower = 0>                C0;  // Variance of prior distribution
}

parameters{
  vector<lower = 0>[t_max]   lambda;  // Standard deviation of state noise (time-varying factor)
    real<lower = 0>          W_sqrt;  // Standard deviation of state noise (time-invariant base)
    real<lower = 0>          V_sqrt;  // Standard deviation of observation noise (time-invariant)
}

transformed parameters{
  vector[t_max]                   W;  
  vector[t_max]                   V;  
  real                      log_lik;  

  for (t in 1:t_max) {
    W[t] = pow(lambda[t] * W_sqrt, 2);
  }

  V = rep_vector(pow(V_sqrt, 2), t_max);

  log_lik = sum(dlm_local_level_loglik(t_max, y, miss, V, W, m0, C0));
}

model{
  target += log_lik;

  lambda ~ cauchy(0, 1);
}
