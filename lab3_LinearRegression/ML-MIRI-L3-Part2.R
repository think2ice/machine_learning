####################################################################
# Machine Learning - MIRI Master
# author:
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

####################################################################
## Exercise 2: Daily air quality measurements in New York, May to September 1973

summary(airquality)

# for information on the dataset
?airquality

# as you can see, there are several missing values ... we should fix this first

# The goal is to develop a predictive model of Ozone concentration given solar radiation, wind and temperature

## Do the following (in separate sections)

# 8. Fix the missing values; most are located in the target variable itself (Ozone); since this is a time series,
#    you can either remove the rows or replace the missing values by interpolation; then deal with Solar.R (knn)
# 9. Do any other pre-processing or visual inspection of the dataset that you think is adequate
#10. Split the data into training (2/3) and test (1/3); WARNING: do this randomly because the dataset
#    has an structure (days are consecutive, etc) or shuffle the data before partitioning
#11. Fit a lm() model to predict 'Ozone' with solar radiation, wind and temperature in the training set
#12. Inspect the results, residuals, and compute LOOCV and the predictive normalized root MSE
#13. Try a second model by log-transforming the Ozone; is this an improvement? if it is, keep it
#14. Try a third model that additionally includes 'Day'; is this an improvement? if it is, keep it
#15. Try a last model that includes a second order polynomial on Wind, like this:
#        lm(log(Ozone) ~ Solar.R + Temp + poly(Wind,2), data = airquality)
#    is this an improvement? if it is, keep it
#16. Do another modelling using ridge regression, and decide the best lambda using GCV
#17. Do another modelling using LASSO regression
#18. Decide which is your best model and give a final prediction in the test set


# Your code starts here ...

# Exercise 2: Daily air quality measurements in New York,
# May to September 1973
# Basic inspection of the data
?airquality
summary(airquality)
str(airquality)
library(DMwR)

# 8: Fixing the missing values
# dealing with the NAs
airquality1 <- knnImputation(airquality, k=10)
summary(airquality1)

# 9: Further preprocessing
attach(airquality1)
par(mfrow = c(2,3))
hist(Ozone)
hist(Solar.R)
hist(Wind)
hist(Temp)
hist(Month)
hist(Day)
# Conclusions: looking at the hist, no further preprocessing is
# needed yet (until the model is defined)

# 10: Training and test data
dim(airquality1)
test_size <- 1/3 * dim(airquality1)[1]