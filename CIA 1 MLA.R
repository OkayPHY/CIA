# Load necessary libraries
library(tidyverse)
library(psych)
library(DataExplorer)
library(car)
library(lmtest)
library(MASS)
library(Metrics)
library(glmnet)
library(dplyr)
library(nortest)
library(mvnormtest)
library(corrplot)

# Load the dataset
computer_data <- read.csv("computer.csv")
computer_data$VendorName <- as.factor(computer_data$VendorName)

# Summary of the dataset
str(computer_data)
summary(computer_data)

# Distribution plots
plot_density(computer_data)
plot_correlation(computer_data)

# Scatterplots for some features
plot(computer_data$MYCT, computer_data$PRP, col = "pink", xlab = "Machine Cycle Time (MYCT)", ylab = "Published Relative Performance (PRP)", main = "Scatterplot of MYCT vs. PRP")
plot(computer_data$MMIN, computer_data$PRP, col = "red", xlab = "Minimum Main Memory (MMIN)", ylab = "Published Relative Performance (PRP)", main = "Scatterplot of MMIN vs. PRP")
plot(computer_data$CACH, computer_data$PRP, col = "blue", xlab = "Cache Memory (CACH)", ylab = "Published Relative Performance (PRP)", main = "Scatterplot of CACH vs. PRP")

# Missing values
plot_missing(computer_data)

# Outlier Detection
boxplot_result <- boxplot(computer_data$PRP, plot = TRUE)
outliers <- boxplot_result$out
if (length(outliers) > 0) {
  print(paste("Outliers:", outliers))
} else {
  print("No outliers detected by IQR method.")
}
outlier_indices <- which(computer_data$PRP %in% outliers)
computer_data_clean <- computer_data[-outlier_indices, ]



# Data partitioning
set.seed(1234)
computer_data_clean <- computer_data_clean[order(runif(nrow(computer_data_clean))),]
computer_data_training <- computer_data_clean[1:167,]
computer_data_testing <- computer_data_clean[168:nrow(computer_data_clean),]

full_model <- lm(PRP ~ VendorName + MYCT + MMIN + MMAX + CACH + CHMIN + CHMAX, data = computer_data_training)
summary(full_model)

# Predictions
full_model_predictions <- predict(full_model, newdata = computer_data_testing)

# Performance Metrics
mse_full <- mse(computer_data_testing$PRP, full_model_predictions)
rmse_full <- rmse(computer_data_testing$PRP, full_model_predictions)
r2_full <- summary(full_model)$r.squared
adj_r2_full <- summary(full_model)$adj.r.squared

print(paste("Full Model - MSE:", mse_full))
print(paste("Full Model - RMSE:", rmse_full))
print(paste("Full Model - R^2:", r2_full))
print(paste("Full Model - Adjusted R^2:", adj_r2_full))

# Reduced Model using Step-wise Regression
step_model <- stepAIC(full_model, direction = "backward")
reduced_model <- lm(PRP ~ VendorName + MYCT + MMIN + MMAX + CACH + CHMIN, data = computer_data_training)
summary(reduced_model)

# Model diagnostics
par(mfrow = c(2, 2))
plot(reduced_model)

# VIF to check multicollinearity
vif_values <- vif(reduced_model)
print(vif_values)

# Durbin-Watson Test for autocorrelation of residuals
dw_test <- durbinWatsonTest(reduced_model)
print(dw_test)

# Component + Residual (CR) Plots to check linearity
crPlots(reduced_model)

# Non-constant Variance Test
ncv_test <- ncvTest(reduced_model)
print(ncv_test)

# Predictions
reduced_model_predictions <- predict(reduced_model, newdata = computer_data_testing)

# Performance Metrics
mse_reduced <- mse(computer_data_testing$PRP, reduced_model_predictions)
rmse_reduced <- rmse(computer_data_testing$PRP, reduced_model_predictions)
r2_reduced <- summary(reduced_model)$r.squared
adj_r2_reduced <- summary(reduced_model)$adj.r.squared

print(paste("Reduced Model - MSE:", mse_reduced))
print(paste("Reduced Model - RMSE:", rmse_reduced))
print(paste("Reduced Model - R^2:", r2_reduced))
print(paste("Reduced Model - Adjusted R^2:", adj_r2_reduced))

# Ridge Regression
X_computer <- model.matrix(PRP ~ VendorName + MMIN + MMAX + CACH + CHMIN, data = computer_data_clean)
Y_computer <- computer_data_clean$PRP
lambda_seq <- 10^seq(10, -2, length = 100)

set.seed(567)
part_computer <- sample(2, nrow(X_computer), replace = TRUE, prob = c(0.7, 0.3))
X_computer_train <- X_computer[part_computer == 1, ]
X_computer_test <- X_computer[part_computer == 2, ]
Y_computer_train <- Y_computer[part_computer == 1]
Y_computer_test <- Y_computer[part_computer == 2]

ridge_model <- glmnet(X_computer_train, Y_computer_train, alpha = 0, lambda = lambda_seq)
ridge_cv <- cv.glmnet(X_computer_train, Y_computer_train, alpha = 0)
best_lambda_ridge <- ridge_cv$lambda.min
print(paste("Optimal Lambda (min MSE):", best_lambda_ridge))

ridge_predictions <- predict(ridge_model, s = best_lambda_ridge, newx = X_computer_test)

mse_ridge <- mean((Y_computer_test - ridge_predictions)^2)
rmse_ridge <- sqrt(mse_ridge)
sst_ridge <- sum((Y_computer_test - mean(Y_computer_test))^2)
sse_ridge <- sum((Y_computer_test - ridge_predictions)^2)
r2_ridge <- 1 - (sse_ridge / sst_ridge)

print(paste("Ridge Regression - MSE:", mse_ridge))
print(paste("Ridge Regression - RMSE:", rmse_ridge))
print(paste("Ridge Regression - R^2:", r2_ridge))

# Lasso Regression
X <- model.matrix(PRP ~ VendorName + MMIN + MMAX + CACH + CHMIN, data = computer_data_clean)
Y <- computer_data_clean$PRP
lambda <- 10^seq(10, -2, length = 100)

set.seed(567)
part <- sample(2, nrow(X), replace = TRUE, prob = c(0.8, 0.2))
X_train <- X[part == 1, ]
X_test <- X[part == 2, ]
Y_train <- Y[part == 1]
Y_test <- Y[part == 2]

lasso_reg <- glmnet(X_train, Y_train, alpha = 1, lambda = lambda)
lasso_reg_cv <- cv.glmnet(X_train, Y_train, alpha = 1)
best_lambda <- lasso_reg_cv$lambda.min
print(paste("Optimal Lambda (min MSE):", best_lambda))

lasso_pred <- predict(lasso_reg, s = best_lambda, newx = X_test)

mse_lasso <- mean((Y_test - lasso_pred)^2)
rmse_lasso <- sqrt(mse_lasso)
sst <- sum((Y_test - mean(Y_test))^2)
sse <- sum((Y_test - lasso_pred)^2)
r2 <- 1 - (sse / sst)

print(paste("Lasso Regression - MSE:", mse_lasso))
print(paste("Lasso Regression - RMSE:", rmse_lasso))
print(paste("Lasso Regression - R^2:", r2))

