##---------Matrix Completion Methods from Athey et al.------------##
## Load the (MCPanel) package from https://github.com/susanathey/MCPanel for
## Matrix Completion by Nuclear Norm Minimization (MCNNM)

library(MCPanel)
library(latex2exp)
library(glmnet)
library(ggplot2)

set.seed(3)

## Taking 3 scenarios mainly where N == T, N << T and N >> T 
#N <- 45 
 N <- 50 # Number of units # nolint
#T <- 60  
#T <- 25
 T <- 50 # Number of time-periods

R <- 2 # Rank of matrix # nolint
noise_sc <- 0.1 # Noise scale
delta_sc <- 0.1 # delta scale
gamma_sc <- 0.1 # gamma scale
N_t <- ceiling(N/2)
T0 <- ceiling(T/2)
num_runs <- 5
is_simul <- 0 # whether to use Simultaneous or Staggered Adoption

## Create Matrices
A <- replicate(R,rnorm(N)) # nolint
B <- replicate(T,rnorm(R)) # nolint
delta <- delta_sc*rnorm(N)
gamma <- gamma_sc*rnorm(T)
true_mat <- A %*% B + replicate(T,delta) + t(replicate(N,gamma))

## Matrices for RMSE values 
MCPanel_RMSE_test <- matrix(0L,num_runs,1)
EN_RMSE_test <- matrix(0L,num_runs,1)
ENT_RMSE_test <- matrix(0L,num_runs,1)
DID_RMSE_test <- matrix(0L,num_runs,1)
ADH_RMSE_test <- matrix(0L,num_runs,1)


## Run for different methods
for (i in c(1:num_runs)){
    noise <- noise_sc*replicate(T,rnorm(N))
    noisy_mat <- true_mat + noise

    treat_mat <- matrix(1L, N, T);
    t0 <- T0
    ## Simultaneuous (simul_adapt) or Staggered adoption (stag_adapt)
    if(is_simul == 1){
      treat_mat <- simul_adapt(noisy_mat, N_t, t0)
    }
    else{
      treat_mat <- stag_adapt(noisy_mat, N_t, t0)
    }
    Y_obs <- noisy_mat * treat_mat

    ## ------
    ## MC-NNM
    ## ------
    print("MCNNM Started")
    est_model_MCPanel <- mcnnm_cv(Y_obs, treat_mat, to_estimate_u = 1, to_estimate_v = 1) ## If N<<T it is better to only estimate u, if T<<<N it is better to only estimate v. # nolint
    est_model_MCPanel$Mhat <- est_model_MCPanel$L + replicate(T,est_model_MCPanel$u) + t(replicate(N,est_model_MCPanel$v)) # nolint 
    est_model_MCPanel$msk_err <- (est_model_MCPanel$Mhat - noisy_mat)*(1-treat_mat)
    est_model_MCPanel$test_RMSE <- sqrt((1/sum(1-treat_mat)) * sum(est_model_MCPanel$msk_err^2))
    MCPanel_RMSE_test[i] <- est_model_MCPanel$test_RMSE

    ## -----
    ## EN : It does Not cross validate on alpha (only on lambda) and keep alpha = 1 (LASSO).
    ##      Change num_alpha to a larger number, if you are willing to wait a little longer.
    ## -----
    print("EN Started")
    est_model_EN <- en_mp_rows(Y_obs, treat_mat, num_alpha = 1)
    est_model_EN_msk_err <- (est_model_EN - noisy_mat)*(1-treat_mat)
    est_model_EN_test_RMSE <- sqrt((1/sum(1-treat_mat)) * sum(est_model_EN_msk_err^2))
    EN_RMSE_test[i] <- est_model_EN_test_RMSE

    ## -----
    ## EN_T : It does Not cross validate on alpha (only on lambda) and keep alpha = 1 (LASSO).
    ## -----
    print("EN-T Started")
    est_model_ENT <- t(en_mp_rows(t(Y_obs), t(treat_mat), num_alpha = 1))
    est_model_ENT_msk_err <- (est_model_ENT - noisy_mat)*(1-treat_mat)
    est_model_ENT_test_RMSE <- sqrt((1/sum(1-treat_mat)) * sum(est_model_ENT_msk_err^2))
    ENT_RMSE_test[i] <- est_model_ENT_test_RMSE

   ## -----
    ## DID
    ## -----
    print("DID Started")
    est_model_DID <- DID(Y_obs, treat_mat)
    est_model_DID_msk_err <- (est_model_DID - noisy_mat)*(1-treat_mat)
    est_model_DID_test_RMSE <- sqrt((1/sum(1-treat_mat)) * sum(est_model_DID_msk_err^2))
    DID_RMSE_test[i] <- est_model_DID_test_RMSE

    ## -----
    ## ADH
    ## -----
    print("ADH Started")
    est_model_ADH <- adh_mp_rows(Y_obs, treat_mat)
    est_model_ADH_msk_err <- (est_model_ADH - noisy_mat)*(1-treat_mat)
    est_model_ADH_test_RMSE <- sqrt((1/sum(1-treat_mat)) * sum(est_model_ADH_msk_err^2))
    ADH_RMSE_test[i] <- est_model_ADH_test_RMSE
}

## Computing means and standard errors
MCPanel_avg_RMSE <- apply(MCPanel_RMSE_test,2,mean)
MCPanel_std_error <- apply(MCPanel_RMSE_test,2,sd)/sqrt(num_runs)

EN_avg_RMSE <- apply(EN_RMSE_test,2,mean)
EN_std_error <- apply(EN_RMSE_test,2,sd)/sqrt(num_runs)

ENT_avg_RMSE <- apply(ENT_RMSE_test,2,mean)
ENT_std_error <- apply(ENT_RMSE_test,2,sd)/sqrt(num_runs)

DID_avg_RMSE <- apply(DID_RMSE_test,2,mean)
DID_std_error <- apply(DID_RMSE_test,2,sd)/sqrt(num_runs)

ADH_avg_RMSE <- apply(ADH_RMSE_test,2,mean)
ADH_std_error <- apply(ADH_RMSE_test,2,sd)/sqrt(num_runs)

## Creating plots

df1 <-
  data.frame(
      y =  c(DID_avg_RMSE, EN_avg_RMSE, ENT_avg_RMSE, MCPanel_avg_RMSE, ADH_avg_RMSE),
      lb = c(DID_avg_RMSE - 1.96*DID_std_error, EN_avg_RMSE - 1.96*EN_std_error,
             ENT_avg_RMSE - 1.96*ENT_std_error, MCPanel_avg_RMSE - 1.96*MCPanel_std_error,
             ADH_avg_RMSE - 1.96*ADH_std_error),
      ub = c(DID_avg_RMSE + 1.96*DID_std_error, EN_avg_RMSE + 1.96*EN_std_error,
             ENT_avg_RMSE + 1.96*ENT_std_error, MCPanel_avg_RMSE + 1.96*MCPanel_std_error,
             ADH_avg_RMSE + 1.96*ADH_std_error),
      x = c(T0/T, T0/T ,T0/T, T0/T, T0/T),
      Method = c(replicate(length(T0),"DID"), replicate(length(T0),"EN"),
                 replicate(length(T0),"EN-T"), replicate(length(T0),"MC-NNM"),
                 replicate(length(T0),"SC-ADH")),
      Marker = c(replicate(length(T0),1), replicate(length(T0),2),
                 replicate(length(T0),3), replicate(length(T0),4),
                 replicate(length(T0),5))

    )
Marker = c(1,2,3,4,5)

p = ggplot(data = df1, aes(x, y, color=Method)) +
  geom_point(mapping=aes(shape=Method),size = 2, position=position_dodge(width=0.1)) + 
  geom_errorbar(
    aes(ymin = lb, ymax = ub),
    width = 0.1,
    linetype = "solid",
    position=position_dodge(width=0.1)) +
  theme_bw() +
  xlab(TeX('$T_0/T$')) +
  ylab("Average RMSE") +
  theme(axis.title=element_text(family="Times", size=14)) +
  theme(axis.text=element_text(family="Times", size=12)) +
  theme(legend.text=element_text(family="Times", size = 12)) +
  theme(legend.title=element_text(family="Times", size = 12))

p 

##----END-------------##