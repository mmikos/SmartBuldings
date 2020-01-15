library(rstanarm)
library(lme4)
library(ggplot2)
library(scales)
library(shinystan)
library(lubridate)
library(rethinking)
#library(parallel)

Sys.setenv("MC_CORES"= 8)
Sys.getenv("MC_CORES")
getOption("mc.cores", 8)

#data = read.csv("data_co2_occ_room1.csv")

setwd("C:/Users/mam/PycharmProjects/SmartBuldings/Hierarchical models")

data = read.csv("data_co2_occ_num.csv")

colnames(data)[colnames(data) == 'sensor_name'] <- 'room'

summary(data)

data$co2.rescaled <- rescale(data$co2) 

data$log_co2 <- scale(log(data$co2))

data$room_category = unclass(data$room)

data$day_of_the_week <- factor(weekdays(as.Date(data$X)))

data$day_of_the_week_category = unclass(data$day_of_the_week)

data$time<-format(strptime(data$X, "%Y-%m-%d %H:%M:%S"), "%H:%M:%S")

breaks <- hour(hm("00:00", "6:00", "12:00", "16:00", "23:59"))

labels <- c("Night", "Morning", "Afternoon", "Evening")

data$time_cat <- cut(x=hour(data$X), breaks = breaks, labels = labels, include.lowest=TRUE)

train_index <- sample(1:nrow(data), 0.75 * nrow(data))
test_index <- setdiff(1:nrow(data), train_index)

# Build X_train, y_train, X_test, y_test
X_train <- data[train_index, -15]

X_test <- data[test_index, -15]

(hist <- ggplot(data, aes(x = occupancy)) +
    geom_histogram(bins=100) +
    theme_classic())


model1 <- glmer(formula = occupancy ~ 1 + (1 | room_category), 
           data = data,
           family=poisson)

summary(model1)


model2 <- glmer(occupancy ~ 1 + co2.rescaled + (1 | room_category), 
                     data = data,
                     family=poisson)


summary(model2)

  
model1_stan <- stan_glmer(occupancy ~ 1 + (1 | room_category), 
                     data = data,
                     family=poisson,
                     seed = 334)

prior_summary(object = model1_stan)
sd(data$occupancy, na.rm = TRUE)

print(model1_stan, digits = 2)

summary(model1_stan, 
        pars = c("(Intercept)", "sigma", "Sigma[room_category:(Intercept),(Intercept)]"),
        probs = c(0.025, 0.975),
        digits = 2)


mu_a_sims <- as.matrix(model1_stan, 
                       pars = "(Intercept)")
# draws for 73 schools' school-level error
u_sims <- as.matrix(model1_stan, 
                    regex_pars = "b\\[\\(Intercept\\) room_category\\:")
# draws for 73 schools' varying intercepts               
a_sims <- as.numeric(mu_a_sims) + u_sims          

# Obtain sigma_y and sigma_alpha^2
# draws for sigma_y
s_y_sims <- as.matrix(model1_stan, 
                      pars = "sigma")
# draws for sigma_alpha^2
s__alpha_sims <- as.matrix(model1_stan, 
                           pars = "Sigma[room_category:(Intercept),(Intercept)]")


a_mean <- apply(X = a_sims,     # posterior mean
                MARGIN = 2,
                FUN = mean)
a_sd <- apply(X = a_sims,       # posterior SD
              MARGIN = 2,
              FUN = sd)

# Posterior median and 95% credible interval
a_quant <- apply(X = a_sims, 
                 MARGIN = 2, 
                 FUN = quantile, 
                 probs = c(0.025, 0.50, 0.975))
a_quant <- data.frame(t(a_quant))
names(a_quant) <- c("Q2.5", "Q50", "Q97.5")

# Combine summary statistics of posterior simulation draws
a_df <- data.frame(a_mean, a_sd, a_quant)
round(head(a_df), 2)

#####


model2_stan <- stan_glmer(occupancy ~ 1 + co2.rescaled + (1 | room_category), 
                          data = data,
                          family=poisson,
                          adapt_delta = 0.99,
                          cores = 8,
                          iter = 2000,
                          seed = 334)


priors <- prior_summary(model2_stan)
priors$prior$adjusted_scale

print(model2_stan, digits = 2)

summary(model2_stan)
model2_posterior_prob <- posterior_interval(model2_stan,
                                            prob = 0.95)
round(model2_posterior_prob, 2)

plot(model2_stan)
plot(model2_stan, "rhat")
plot(model2_stan, "ess")

#model3 <- stan_glmer(occupancy ~ co2 + (co2 | room), 
#                     data = X_train,
#                     family=poisson,
#                     cores = 4,
#                     iter = 500)


plot(model2_stan, plotfun = "trace")

launch_shinystan(model2_stan)
y_pred <- posterior_predict(model2_stan,
                           newdata=X_test)


pp_check(model2_stan)

loo_model2_stan <- loo(model2_stan, k_threshold = 0.7, cores = getOption("mc.cores", 1))
loo_model2_stan

kfold_model2_stan <- kfold(model2_stan, K=1000, cores = getOption("mc.cores", 8))

par(mfrow = 1:2, mar = c(5,3.8,1,0) + 0.1, las = 3)
plot(loo_model2, label_points = TRUE)

detach(package:rethinking, unload = T)
library(brms)
library(tidyverse)

brm_model2 <- 
  brm(data = data, family = poisson,
      occupancy ~ 1 + co2.rescaled + (1 | room_category),
      iter = 2000, cores = 8,
      seed = 334, control = list(adapt_delta = 0.9999999, max_treedepth=20))

mcmc_plot(brm_model2)

pairs(brm_model2)

print(brm_model2)

fitted(brm_model2) %>%
  as_tibble() %>%
  bind_cols(data)  %>%

  ggplot(aes(x = occupancy, y = Estimate)) +
  geom_abline(linetype = 2, color = "grey50", size = .5) +
  geom_point(size = 1.5, color = "firebrick4", alpha = 3/4) +
  geom_linerange(aes(ymin = Q2.5, ymax = Q97.5),
                 size = 1/4, color = "firebrick4") +
  geom_linerange(aes(ymin = Estimate - Est.Error, 
                     ymax = Estimate + Est.Error),
                 size = 1/2, color = "firebrick4") +
  labs(x = "Observed occupancy", 
       y = "Predicted occupancy") +
  theme_bw() +
  theme(panel.grid = element_blank())


library(ggthemes)

data %>%
  glimpse()

data_numbered <- 
  data %>%
  mutate(tank = 1:nrow(data))

tank_model <- 
  brm(data = data_numbered, family = binomial,
      1 | trials(room_category) ~ 1 + (1 | tank),
      prior = c(prior(normal(0, 1), class = Intercept),
                prior(cauchy(0, 1), class = sd)),
      iter = 4000, warmup = 1000, chains = 4, cores = 4,
      seed = 12)

post_mdn <- 
  coef(brm_model2, robust = T)$tank[, , ] %>% 
  as_tibble() %>% 
  bind_cols(data_numbered) %>%
  mutate(post_mdn = inv_logit_scaled(Estimate))




post_mdn %>%
  ggplot(aes(x = room)) +
  geom_hline(yintercept = inv_logit_scaled(median(post$b_Intercept)), linetype = 2, size = 1/4) +
  geom_vline(xintercept = c(16.5, 32.5), size = 1/4) +
  geom_point(aes(y = propsurv), color = "orange2") +
  geom_point(aes(y = post_mdn), shape = 1) +
  coord_cartesian(ylim = c(0, 1)) +
  scale_x_continuous(breaks = c(1, 16, 32, 48)) +
  labs(title    = "Multilevel shrinkage!",
       subtitle = "The empirical proportions are in orange while the model-\nimplied proportions are the black circles. The dashed line is\nthe model-implied average survival proportion.") +
  annotate("text", x = c(8, 16 + 8, 32 + 8, 40 + 8, 48 + 8), y = 0, 
           label = c("Room 1", "Room 2", "Room 3", "Room 4", "Room 5")) +
  theme_fivethirtyeight() +
  theme(panel.grid = element_blank())
  