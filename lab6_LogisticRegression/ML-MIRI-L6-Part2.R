####################################################################
# Machine Learning - MIRI Master
# author:
#think_2ice <- Manel Alba

# LAB 6: GLMs: logistic Regression and beyond (Part2)
# version of April 2016
####################################################################

####################################################################
## Exercise 1
####################################################################

## This exercise involves the use of the South African Coronary Heart Disease (CHD) data set (the one mentioned in the slides

## The task is to predict CHD using age at onset, current alcohol consumption, obesity levels, 
## cumulative tobacco, type-A behavior, and low density lipoprotein cholesterol as predictors

# CHD = Coronary heart disease

SAheart <- read.csv("MDSAheart.data")

## First some necessary pre-processing (this part should be lenghtier as you know but let's go to the point)

SAheart <- subset(SAheart, select=-row.names)
SAheart[,"famhist"] <- as.factor(SAheart[,"famhist"])
SAheart[,"chd"] <- factor(SAheart[,"chd"], labels=c("No","Yes"))

# There seem to be no missing values
# alcohol, ldl and tobacco have very skewed distributions! a log transform could be in order
# ---> it seems that quite a bit of people have large values (not surprising)

# so this is your departing data set

summary(SAheart) 

## Assignment for this lab session
## Do the following (in separate sections)

# 3. Decide if you log-transform some of the continuous variables
# 4. Create learning and test sets, selecting randomly 2/3 and 1/3 of the data
# 5. Fit a logistic regression model using only the learn data with Coronary Heart Disease (chd) as the target
#    and observe the results (summary):
#      - build a maximal model using the learn data
#      - try to simplify the model by eliminating the least important variables progressively
#      - refit the model using these variables
# 6. Interpret a couple of coefficients
# 7. Calculate the misclassification rate for your model in the learn data and in the leftout data (test data)
#    Try to come up with a more acceptable model by playing with how the classifier assigns classes according to probability

# Your code starts here ...

#0. Basic Analysis
summary(SAheart)
str(SAheart)

# 1. Perform a pairs() plot of the variables paired for a first exploration, with chd overlayed (in color)
cols <- character(nrow(SAheart))
cols[SAheart$chd == "Yes"] <- "red"
cols[SAheart$chd == "No"] <- "green"
pairs(SAheart, col=cols)

# 2. Perform various boxplots (notice that all but one of the predictors are continuous)
par(mfrow=c(3,3))
nums <- c(1,2,3,4,6,7,8,9)
for (i in nums){
  boxplot(SAheart[,i], main=colnames(SAheart)[i])
}
####################################################################
## Exercise 2
####################################################################

## The following data represents the number of news AIDS cases in Belgium, from 1981 onwards
## Data from the book by Venables & Ripley (Modern applied statistics in S)
# AIDS = Acquired inmune deficience system 

Year <- 1:13
Cases <- c(12,14,33,59,67,74,123,141,165,204,253,246,240)

plot(Year+1980, Cases, col="blue", xlab="Year", ylab="News AIDS cases")

# 1. Justify the fitting of a Poisson regression model
# 2. Fit the Poisson regression model and examine the results
# 3. Give the model using the obtained coefficients
# 4. Gather predictor, outcome and prediction together and plot everything
# 5. Analyse the residuals and decide a derived predictor; redo the model and compare
# Your code starts here ...


