data {
  int<lower=0> N;//Games
  int<lower=0> N_pred;//Out-of-sample games
  int<lower=0> N_cov;//Nr covariates
  int<lower=0> K;//Teams
  int Z[N];// differenza goal
  int<lower=0> abs_Z[N];// differenza goal
  int<lower=0> lab_H[N];// label home
  int<lower=0> lab_A[N];// label away
  int<lower=0> lab_H_pred[N_pred];// label home oos
  int<lower=0> lab_A_pred[N_pred];// label away oos
  matrix[N, N_cov] X_H;// covariates home
  matrix[N_pred, N_cov] X_pred_H;// covariats home oos
  matrix[N, N_cov] X_A;// covariates away
  matrix[N_pred, N_cov] X_pred_A;// covariates away oos
  vector<lower=0>[N_cov] s2_x;// covariates variances
  vector[N_pred] home_pred; // no home effect in the final
  real<lower=0> sigma_HS; // esitmate sigma HS prior
  matrix[K, K-1] A_qr; // rotation matrix
  real<lower=0> p0;  // Expected number of large slopes
  real<lower=0> slab_scale; // Scale for large slopes
  real<lower=0> slab_df;// Effective degrees of freedom for large slopes
}

transformed data{
  real<lower=0> slab_scale2 = square(slab_scale);
  real<lower=0> half_nu = 0.5 * slab_df;
  real tau0 = (p0 / (1.0*N_cov - p0)) * (sigma_HS / sqrt(1.0 * N));
}

parameters {
  real<lower = 0> sigma_att;
  real<lower = 0> sigma_def;
  real home_eff;
  real theta;
  vector[K-1] att_raw;
  vector[K-1] def_raw;
  vector[N_cov] z_beta;
  vector<lower=0>[N_cov] lambda_HS;
  real<lower = 0> z_c2;
  real<lower = 0> z_tau;
}

transformed parameters{
  vector[K] att =  sigma_att * A_qr * att_raw;
  vector[K] def =  sigma_def * A_qr * def_raw;
  real<lower = 0> mu_H[N];
  real<lower = 0> mu_A[N];
  real<lower = 0> mu_H_out[N_pred];
  real<lower = 0> mu_A_out[N_pred];
  real tau = tau0 * z_tau;
  real c2 = slab_scale2 * z_c2;
  vector[N_cov] lambda_tilde = sqrt(c2 * square(lambda_HS) ./ (c2 + square(tau) * square(lambda_HS)));
  vector[N_cov] beta = tau * lambda_tilde .* z_beta;

  for(i in 1:N){
    mu_H[i] = exp(theta + home_eff + att[lab_H[i]]+ def[lab_A[i]]+X_H[i,] * beta);
    mu_A[i] = exp(theta + att[lab_A[i]]+ def[lab_H[i]] + X_A[i,] * beta);
  }
  
  for(i in 1:N_pred){
    mu_H_out[i] = exp(theta + home_eff*  home_pred[i] + att[lab_H_pred[i]]+ def[lab_A_pred[i]]+X_pred_H[i,] * beta);
    mu_A_out[i] = exp(theta + att[lab_A_pred[i]]+ def[lab_H_pred[i]] + X_pred_A[i,] * beta);
  }
 }


model {
  theta ~ normal(0,5);
  home_eff ~ normal(0,5);
  sigma_att ~ student_t(3,0,1);
  sigma_def ~ student_t(3,0,1);
   // regularized HS prior
  z_beta ~ normal(0,1);
  lambda_HS ~ cauchy(0, 1);
  z_tau ~ cauchy(0, 1);
  z_c2 ~ inv_gamma(half_nu, half_nu);
  
  att_raw ~ normal(0,  inv(sqrt(1 - inv(K))));
  def_raw ~ normal(0,  inv(sqrt(1 - inv(K))));
  for (n in 1:N) {
          target += -(mu_H[n]+mu_A[n]) + 0.5*Z[n]*log(mu_H[n]/mu_A[n])+
          log(modified_bessel_first_kind(abs_Z[n],2.0*sqrt(mu_H[n]*mu_A[n])));
  }
}

generated quantities{
  vector[N] log_lik;
  int<lower=0> Y_H_pred_in[N];
  int<lower=0> Y_A_pred_in[N];
  int<lower=0> Y_H_pred_out[N_pred];
  int<lower=0> Y_A_pred_out[N_pred];
  for (n in 1:N) {
    log_lik[n] = -(mu_H[n]+mu_A[n]) + 0.5*Z[n]*log(mu_H[n]/mu_A[n])+
          log(modified_bessel_first_kind(abs_Z[n],2.0*sqrt(mu_H[n]*mu_A[n])));
  }
  for(i in 1:N){
    Y_H_pred_in[i]=poisson_rng(mu_H[i]);
    Y_A_pred_in[i]=poisson_rng(mu_A[i]);
  }
  for(i in 1:N_pred){
    Y_H_pred_out[i]=poisson_rng(mu_H_out[i]);
    Y_A_pred_out[i]=poisson_rng(mu_A_out[i]);
  }

}
