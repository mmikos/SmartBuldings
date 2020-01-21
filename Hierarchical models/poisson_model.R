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

data$day_of_week <- factor(weekdays(as.Date(data$X)))

data$day_of_week_cat = unclass(factor(weekdays(as.Date(data$X))))

data$time<-format(strptime(data$X, "%Y-%m-%d %H:%M:%S"), "%H:%M:%S")

weekdays <- c('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday')

data$weekday <- factor((weekdays(as.Date(data$X)) %in% weekdays), levels=c(FALSE, TRUE), labels=c('weekend', 'weekday'))

breaks <- hour(hm("00:00", "6:00", "11:00", "14:00", "17:00", "23:59"))

labels <- c("Night", "Morning", "Afternoon","Late afternoon", "Evening")

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

#####


model2_stan <- stan_glmer(occupancy ~ 1 + co2.rescaled + (1 | room_category), 
                          data = data,
                          family=poisson,
                          adapt_delta = 0.99,
                          cores = 8,
                          iter = 2000,
                          seed = 334)


post_model2_stan <- posterior_samples(model2_stan, add_chain = T)


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

library(ggplot2)
library(bayesplot)
theme_set(bayesplot::theme_default())


prop_zero_test1 <- pp_check(model2_stan)

model2_stan_neg_bin <- stan_glmer(occupancy ~ 1 + co2.rescaled + (1 | room_category), 
                                  data = data,
                                  family = neg_binomial_2,
                                  adapt_delta = 0.99,
                                  cores = 8,
                                  iter = 2000,
                                  seed = 12)

prop_zero_test2 <- pp_check(model2_stan_neg_bin)

bayesplot_grid(prop_zero_test1 + ggtitle("Poisson"), 
               prop_zero_test2 + ggtitle("Negative Binomial"), 
               grid_args = list(ncol = 2))

#model3 <- stan_glmer(occupancy ~ co2 + (co2 | room), 
#                     data = X_train,
#                     family=poisson,
#                     cores = 4,
#                     iter = 500)


plot(model2_stan, plotfun = "trace")

launch_shinystan(model2_stan)
y_pred <- posterior_predict(model2_stan,
                           newdata=X_test)


loo_model2_stan_poiss <- loo(model2_stan_poiss, k_threshold = 0.7, cores = 1)
loo_model2_stan_neg_bin <- loo(model2_stan_neg_bin, k_threshold = 0.7, cores = 1)

loo_compare(loo_model2_stan_poiss, loo_model2_stan_neg_bin)


kfold_model2_stan <- kfold(model2_stan, K=1000, cores = getOption("mc.cores", 8))

par(mfrow = 1:2, mar = c(5,3.8,1,0) + 0.1, las = 3)
plot(loo_model2, label_points = TRUE)

detach(package:rethinking, unload = T)
library(brms)
library(tidyverse)

brm_poisson_model <- 
  brm(data = data, family = poisson,
      occupancy ~ 1 + co2.rescaled + (1 | room_category),
      iter = 2000, cores = 8,
      seed = 334, control = list(adapt_delta = 0.9999999, max_treedepth=20))

brm_zero_inflated_poisson_model <- 
  brm(data = data, family = zero_inflated_poisson,
      occupancy ~ 1 + co2.rescaled + (1 | room_category),
      iter = 3000, cores = 8, control = list(adapt_delta = 0.9999999),
      seed = 334)


brm_negbinomial_model <- 
  brm(data = data, family = negbinomial,
      occupancy ~ 1 + co2.rescaled + (1 | room_category),
      iter = 2000, cores = 8, control = list(adapt_delta = 0.9999999, max_treedepth=20),
      seed = 334)



mcmc_plot(brm_poisson_model)
mcmc_plot(brm_zero_inflated_poisson_model)
mcmc_plot(brm_negbinomial_model)
 
pairs(brm_poisson_model) 
pairs(brm_zero_inflated_poisson_model)
pairs(brm_negbinomial_model)

print(brm_poisson_model)


brm_zero_inflated_poisson_model_day <- brm(data = data, family = zero_inflated_poisson,
                                           occupancy ~ 1 + co2.rescaled + (1 | room) + (1 | day_of_week),
                                           iter = 2000, cores = 1, control = list(adapt_delta = 0.99999999, max_treedepth=20),
                                           seed = 334)

brm_zero_inflated_poisson_model_time <- brm(data = data, family = zero_inflated_poisson,
                                            occupancy ~ 1 + co2.rescaled + (1 | room) + (1 | day_of_week) + (1 | time_cat),
                                            iter = 2000, cores = 1, control = list(adapt_delta = 0.99999999, max_treedepth=20),
                                            seed = 334)

#### Poisson model

fitted(brm_zero_inflated_poisson_model_day) %>%
  as_tibble() %>%
  bind_cols(data)  %>%

  ggplot(aes(x = occupancy, y = Estimate)) +
  geom_abline(linetype = 2, color = "grey", size = .5) +
  geom_point(size = 1.5, color = "darkred", alpha = 3/4) +
  geom_linerange(aes(ymin = Q2.5, ymax = Q97.5),
                 size = 1/4, color = "darkred") +
  geom_linerange(aes(ymin = Estimate - Est.Error, 
                     ymax = Estimate + Est.Error),
                 size = 1/2, color = "darkred") +
  labs(x = "Observed occupancy", 
       y = "Predicted occupancy in Poisson model") +
  theme_bw() +
  theme(panel.grid = element_blank())


#### Zero Inflated Poisson model

fitted(brm_zero_inflated_poisson_model) %>%
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
       y = "Predicted occupancy in Zero Inflated Poisson model") +
  theme_bw() +
  theme(panel.grid = element_blank())

#### Negative Binomial model

fitted(brm_negbinomial_model) %>%
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
       y = "Predicted occupancy in Negative Binomial model") +
  theme_bw() +
  theme(panel.grid = element_blank())

#####

loo_brm_poisson_model <- loo(brm_poisson_model, k_threshold = 0.7, cores = getOption("mc.cores", 1), reloo = TRUE)
loo_brm_negbinomial_model <- loo(brm_negbinomial_model, k_threshold = 0.7, cores = getOption("mc.cores", 1))
loo_brm_zero_inflated_poisson_model <-loo(brm_zero_inflated_poisson_model, k_threshold = 0.7, cores = getOption("mc.cores", 1))

loo_compare(loo_brm_poisson_model, loo_brm_negbinomial_model, loo_brm_zero_inflated_poisson_model)


post_model2 <- posterior_samples(brm_model2, add_chain = T)

post_model2 %>% 
  sample_n(100) %>% 
  expand(nesting(iter, b_Intercept, sd_room_category__Intercept),
         x = seq(from = -4, to = 5, length.out = 100)) %>% 
  
  ggplot(aes(x = x, group = iter)) +
  geom_line(aes(y = dnorm(x, b_Intercept, sd_room_category__Intercept)),
            alpha = .2, color = "orange2") +
  labs(title = "Population survival distribution",
       subtitle = "The Gaussians are on the log-odds scale.") +
  scale_y_continuous(NULL, breaks = NULL) +
  coord_cartesian(xlim = c(-3, 4)) + 
  theme_fivethirtyeight() +
  theme(plot.title    = element_text(size = 13),
        plot.subtitle = element_text(size = 10))

library(ggthemes)

library(rethinking)
data(reedfrogs)
d <- reedfrogs

rm(reedfrogs)
detach(package:rethinking, unload = T)
library(brms)
library(tidyverse)

d <- 
  d %>%
  mutate(tank = 1:nrow(d))

b12.1 <- 
  brm(data = d, family = binomial,
      surv | trials(density) ~ 0 + factor(tank),
      prior(normal(0, 5), class = b),
      iter = 2000, warmup = 500, chains = 4, cores = 4,
      seed = 12)

b12.2 <- 
  brm(data = d, family = binomial,
      surv | trials(density) ~ 1 + (1 | tank),
      prior = c(prior(normal(0, 1), class = Intercept),
                prior(cauchy(0, 1), class = sd)),
      iter = 4000, warmup = 1000, chains = 4, cores = 4,
      seed = 12)


post <- posterior_samples(b12.2, add_chain = T)

post_mdn <- 
  coef(b12.2, robust = T)$tank[, , ] %>% 
  as_tibble() %>% 
  bind_cols(d) %>%
  mutate(post_mdn = inv_logit_scaled(Estimate))

post_mdn

data %>%
  glimpse()

data2$prop <- occupancy/(max(occupancy))

data_numer <- 
  data %>%
  mutate(room_number = 1:nrow(data))

data_numer%>%
  glimpse()

brm_model_numer <- 
  brm(data = data_numer, family = poisson,
      occupancy ~ 1 + co2.rescaled + (1 | room_number),
      prior = c(prior(normal(0, 1), class = Intercept),
                prior(cauchy(0, 1), class = sd)),
      iter = 2000, cores = 1, chains = 4,
      seed = 334)



post_mdn <- 
  coef(brm_model_numer, robust = T)$room_number[, , ] %>% 
  as_tibble() %>% 
  bind_cols(data_numer) %>%
  mutate(post_mdn = inv_logit_scaled(Estimate.Intercept))

set.seed(12)

post %>% 
  sample_n(100) %>% 
  expand(nesting(iter, b_Intercept, sd_room_number__Intercept),
         x = seq(from = -4, to = 5, length.out = 100)) %>% 
  
  ggplot(aes(x = x, group = iter)) +
  geom_line(aes(y = dnorm(x, b_Intercept, sd_room_number__Intercept)),
            alpha = .2, color = "orange2") +
  labs(title = "Population survival distribution",
       subtitle = "The Gaussians are on the log-odds scale.") +
  scale_y_continuous(NULL, breaks = NULL) +
  coord_cartesian(xlim = c(-3, 4)) + 
  theme_fivethirtyeight() +
  theme(plot.title    = element_text(size = 13),
        plot.subtitle = element_text(size = 10))

post_mdn %>%
  ggplot(aes(x = room_number)) +
  geom_hline(yintercept = inv_logit_scaled(median(post$b_Intercept)), linetype = 2, size = 1/4) +
  geom_vline(xintercept = c(16.5, 32.5), size = 1/4) +
  geom_point(aes(y = occupancy/max(occupancy)), color = "orange2") +
  geom_point(aes(y = post_mdn), shape = 1) +
  coord_cartesian(ylim = c(0, 1)) +
  scale_x_continuous(breaks = c(1, 16, 32, 48)) +
  labs(title    = "Multilevel shrinkage") +
  annotate("text", x = c(8, 16 + 8, 32 + 8, 40 + 8, 48 + 8), y = 0, 
           label = c("Room 1", "Room 2", "Room 3", "Room 4", "Room 5")) +
  theme_fivethirtyeight() +
  theme(panel.grid = element_blank())


