####################################################################
# Machine Learning - MIRI Master
# think_2ice <- Manel Alba

# LAB 3: Linear regression and beyond (Part2)
# version of March 2016
####################################################################

####################################################################
## Exercise 1:  use of the 'Auto' data set

## This exercise involves the use of the 'Auto' data set, which we introduced in a previous lab session
## We intend to develop a predictive model of fuel consumption given engine power

## Do the following (in separate sections)

# 1. Get the Auto data, redo the preprocessing; split the data into training (2/3) and test (1/3)
# 2. Fit a lm() model to predict 'L100km' with 'horsepower' in the training set
# 3. Inspect the results, residuals, and compute LOOCV and the predictive normalized root MSE
# 4. Do a scatter plot of both variables: do you think the assumption of constant variance is tenable?
#    If your answer is no or don't know, try a log-log model: predict log(L100km) with log(horsepower)
#    Before doing anything, replot the new data and re-answer the question.
# 5. Fit a second lm() model to predict log(L100km) with log(horsepower)
# 6. Inspect the results, residuals, and compute LOOCV and the predictive normalized root MSE
# 7. Do another modelling using ridge regression, and decide the best lambda using GCV
# 8. Do another modelling using LASSO regression
# 9. Decide which is your best model and give a final prediction in the test set

##########################################################################################################

## Exercise 2: Daily air quality measurements in New York, May to September 1973
# The goal is to develop a predictive model of Ozone concentration given solar radiation, wind and temperature

# Basic inspection of the data
?airquality
summary(airquality)
str(airquality)

# 8. Fix the missing values; most are located in the target variable itself (Ozone); since this is a time series,
#    you can either remove the rows or replace the missing values by interpolation; then deal with Solar.R (knn)
library(DMwR)
airquality1 <- knnImputation(airquality, k=10)
summary(airquality1)

# 9. Do any other pre-processing or visual inspection of the dataset that you think is adequate

par(mfrow = c(2,3))
for (i in 1:ncol(airquality1)){
  hist(airquality1[,i])
}
# Conclusions: looking at the hist, no further preprocessing is
# needed yet (until the model is defined)

#10. Split the data into training (2/3) and test (1/3); WARNING: do this randomly because the dataset
#    has an structure (days are consecutive, etc) or shuffle the data before partitioning

dim(airquality1)
test.size <- 1/3 * dim(airquality1)[1]
train.size <- 2/3 * dim(airquality1)[1]
# randomize rows in a matrix/data frame
# https://stat.ethz.ch/pipermail/r-help/2004-December/062499.html
n <- nrow(airquality1)
airquality1 <- airquality1[sample(n),]
test.data <- airquality1[1:test.size,]
train.data <- airquality1[(test.size+1):n,]

#11. Fit a lm() model to predict 'Ozone' with solar radiation, wind and temperature in the training set

# model1 <- glm ( Ozone ~ Solar.R + Wind + Temp , data=train.data, family = gaussian)
# glm with family = gaussian is the same as lm 
model1 <- lm ( Ozone ~ Solar.R + Wind + Temp , data=train.data)
#12. Inspect the results, residuals, and compute LOOCV and the predictive normalized root MSE

attributes(model1)
model1$coefficients
# Slide 13, NRMSE^2
m1.train.pred <- predict(model1,newdata = train.data)
print("MODEL 1: lm ( Ozone ~ Solar.R + Wind + Temp , data=train.data) ")
print("NRMSE squared for training data")
(m1.norm.root.mse.train <-  sqrt(sum((train.data$Ozone - m1.train.pred)^2)/((train.size-1)*var(train.data$Ozone))))
# norm.root.mse.train <- sqrt(model1$deviance/((train.size-1)*var(train.data[,1])))
print("NRMSE squared for test data")
m1.test.pred <- predict(model1,newdata = test.data)
(m1.norm.root.mse.test <- sqrt(sum((test.data$Ozone - m1.test.pred)^2)/((test.size-1)*var(test.data$Ozone))))
# LOOCV
(m1.LOOCV <- sum((model1$residuals/(1-ls.diag(model1)$hat))^2)/train.size)
# (R2.LOOCV = 1 - LOOCV*train.size/((train.size-1)*var(train.data$Ozone)))

#13. Try a second model by log-transforming the Ozone; is this an improvement? if it is, keep it

model2 <- lm (log(Ozone) ~ Solar.R + Wind + Temp , data=train.data)
model2$coefficients
# Slide 13, NRMSE^2
m2.train.pred <- predict(model2,newdata = train.data)
print("MODEL 2: lm (log(Ozone) ~ Solar.R + Wind + Temp , data=train.data)")
print("NRMSE squared for training data")
(m2.norm.root.mse.train <-  sqrt(sum((log(train.data$Ozone) - m2.train.pred)^2)/((train.size-1)*var(log(train.data$Ozone)))))
# norm.root.mse.train <- sqrt(model1$deviance/((train.size-1)*var(train.data[,1])))
m2.test.pred <- predict(model2,newdata = test.data)
print("NRMSE squared for test data")
(m2.norm.root.mse.test <- sqrt(sum((log(test.data$Ozone) - m2.test.pred)^2)/((test.size-1)*var(log(test.data$Ozone)))))
# LOOCV
(m2.LOOCV <- sum((model2$residuals/(1-ls.diag(model2)$hat))^2)/train.size)
# (m2.R2.LOOCV = 1 - m2.LOOCV*train.size/((train.size-1)*var(log(train.data$Ozone))))

#14. Try a third model that additionally includes 'Day'; is this an improvement? if it is, keep it

model3 <- lm ( log(Ozone) ~ Solar.R + Wind + Temp + Day , data=train.data)
model3$coefficients
# Slide 13, NRMSE^2
m3.train.pred <- predict(model3,newdata = train.data)
print("MODEL 3: lm ( log(Ozone) ~ Solar.R + Wind + Temp + Day , data=train.data)")
print("NRMSE squared for training data")
(m3.norm.root.mse.train <-  sqrt(sum((log(train.data$Ozone) - m3.train.pred)^2)/((train.size-1)*var(log(train.data$Ozone)))))
# norm.root.mse.train <- sqrt(model1$deviance/((train.size-1)*var(train.data[,1])))
m3.test.pred <- predict(model3,newdata = test.data)
print("NRMSE squared for test data")
(m3.norm.root.mse.test <- sqrt(sum((log(test.data$Ozone) - m3.test.pred)^2)/((test.size-1)*var(log(test.data$Ozone)))))
# LOOCV
(m3.LOOCV <- sum((model3$residuals/(1-ls.diag(model3)$hat))^2)/train.size)
#(m3.R2.LOOCV = 1 - m3.LOOCV*train.size/((train.size-1)*var(log(train.data$Ozone))))

#15. Try a last model that includes a second order polynomial on Wind, like this:
#        lm(log(Ozone) ~ Solar.R + Temp + poly(Wind,2), data = airquality)
#    is this an improvement? if it is, keep it

model4 <- lm(log(Ozone) ~ Solar.R + Temp + poly(Wind,2), data = train.data)
model4$coefficients
# Slide 13, NRMSE^2
print("MODEL 4: lm(log(Ozone) ~ Solar.R + Temp + poly(Wind,2), data = train.data)")
print("NRMSE squared for training data")
m4.train.pred <- predict(model4,newdata = train.data)
(m4.norm.root.mse.train <-  sqrt(sum((log(train.data$Ozone) - m4.train.pred)^2)/((train.size-1)*var(log(train.data$Ozone)))))
# norm.root.mse.train <- sqrt(model1$deviance/((train.size-1)*var(train.data[,1])))
print("NRMSE squared for test data")
m4.test.pred <- predict(model4,newdata = test.data)
(m4.norm.root.mse.test <- sqrt(sum((log(test.data$Ozone) - m4.test.pred)^2)/((test.size-1)*var(log(test.data$Ozone)))))
# LOOCV
(m4.LOOCV <- sum((model4$residuals/(1-ls.diag(model4)$hat))^2)/train.size)
# (m4.R2.LOOCV = 1 - LOOCV*train.size/((train.size-1)*var(log(train.data$Ozone))))

#16. Do another modelling using ridge regression, and decide the best lambda using GCV

#First I use a wide range of lambdas
lambdas <- 10^seq(-6,2,0.1)
select(lm.ridge(train.data$Ozone ~ train.data$Solar.R + train.data$Temp + train.data$Wind, lambda = lambdas))
lambdas <- seq(6,7,0.001)
select(lm.ridge(train.data$Ozone ~ train.data$Solar.R + train.data$Temp + train.data$Wind, lambda = lambdas))
# Best lambda at 6
model5 <- lm.ridge(train.data$Ozone ~ train.data$Solar.R + train.data$Temp + train.data$Wind, lambda = 6)
print("MODEL 5: lm.ridge(train.data$Ozone ~ train.data$Solar.R + train.data$Temp + train.data$Wind, lambda = 6)")
# Train squared error
print("NRMSE squared for training data")
(m5.norm.root.mse.train <- sqrt(model5$GCV))
# Test squared error
print("NRMSE squared for test data")
m5.test.pred <- scale(test.data[,2:4], center = TRUE, scale = model5$scales) %*% model5$coef + model5$ym
(m5.norm.root.mse.test <- sqrt(sum((test.data$Ozone - m5.test.pred)^2)/((test.size-1)*var(test.data$Ozone))))
# LOOCV
# How to calculate it in a ridge regression?
# (m5.LOOCV <- sum((model5$residuals/(1-ls.diag(model5)$hat))^2)/train.size)
# (m5.R2.LOOCV = 1 - m5.LOOCV*train.size/((train.size-1)*var(train.data$Ozone)))

#17. Do another modelling using LASSO regression

library(lars)
model6 <- lars(as.matrix(train.data[,2:4]),as.matrix(train.data[,1]),type = "lasso")
attributes(model6)
m6.lambda <- model6$lambda
m6.beta <- model6$beta
m6.train.pred<- sqrt(sum((train.data$Ozone - predict(model6, as.matrix(train.data[,2:4]), type="fit")$fit)^2)/(train.size-1)*var(train.data$Ozone))
m6.test.pred <- sqrt(sum((test.data$Ozone - predict(model6, as.matrix(test.data[,2:4]), type="fit")$fit)^2)/(test.size-1)*var(test.data$Ozone))
print("MODEL 6: lars(as.matrix(train.data[,2:4]),as.matrix(train.data[,1]),type = 'lasso')")
print("NRMSE squared for training data")
(m6.norm.root.mse.train <-  sqrt(sum((train.data$Ozone - m6.train.pred)^2)/((train.size-1)*var(train.data$Ozone))))
print("NRMSE squared for test data")
(m6.norm.root.mse.test <- sqrt(sum((test.data$Ozone - m6.test.pred)^2)/((test.size-1)*var(test.data$Ozone))))
# Why these values in Lasso and not in the interval [0,1]? 

#18. Decide which is your best model and give a final prediction in the test set

# Plot of both training and test NRMSE squared for all models (lower the better)
# Plot of LOOCV for all models (lower the better)
# Plot of the real values vs the predicted values for all models (more similar the better)

