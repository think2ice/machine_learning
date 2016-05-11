####################################################################
# Machine Learning - MIRI Master
# author:
# think_2ice <- Manel Alba

# LAB 11: Multilayer Perceptrons (Part 2)
# version of May 2016
####################################################################

####################################################################
## Exercise 1

## We continue with Example 3: mixture of distributions artificial 2D data

## In order to find the best network architecture, you are asked to explore the two methods:

## a) Explore different numbers of hidden units in one hidden layer, with no regularization
## b) Fix a large number of hidden units in one hidden layer, and explore different regularization values

## Very much as it was done in Example 1; you need first to split the data into training (2/3) and test (1/3)

## Once you select the best model overall (including the two methods), compute the test error and plot 
## the learned decision function against the data and the truth

####################################################################
set.seed (4)

# Create the function to produce the mixture distribution
mixture.distributions <- function (N,N2=N)
{
  z<- c(rbinom(1,N,1/3))
  z[2]<-rbinom(1,N-z[1],1/2)
  v<- c(rbinom(1,N,1/3))
  v[2]<-rbinom(1,N-v[1],1/2)
  x <- c(rnorm(z[1])/2+2, rnorm(z[2])+6, rnorm(N-z[1]-z[2])/2+3, rnorm(v[1])/1+4, rnorm(v[2])/3+1,
         rnorm(N-v[1]-v[2])+8)
  y <- c(rnorm(z[1])/2+1, rnorm(z[2])/3+1, rnorm(N-z[1]-z[2])/2+1.5, rnorm(v[1])/3+2, rnorm(v[2])/3+2,
         rnorm(N-v[1]-v[2])/3+1.5)
  target <- c(rep("red",N),rep("blue",N2))
  return(data.frame(x,y,target))
}
# Generate the data
N <- 400
dat <- mixture.distributions (N)
summary(dat)
dat <- dat[sample(nrow(dat)),]

# Prepare the test and train sets
learn <- sample(1:N, round(2*2*N/3))
nlearn <- length(learn)
ntest <- N - nlearn

# Scale the data
dat$x <- scale(dat$x)
dat$y <- scale(dat$y)

# Study and create the best model
library(caret)
(decays <- 10^seq(-3,0,by=0.1))
trc <- trainControl (method="repeatedcv", number=10, repeats=10)
model.10x10CV <- train (dat$target ~., data = dat, subset=learn, method='nnet', maxit = 500, trace = FALSE,
                        tuneGrid = expand.grid(.size=20,.decay=decays), trControl=trc)

model.10x10CV$results
model.10x10CV$bestTune

# And the corresponding test error
p2 <- as.factor(predict (model.10x10CV, newdata=dat[-learn,], type="raw"))
ntest
(t2 <- table(p2,dat$target[-learn]))
(error_rate.test <- 100*(1-sum(diag(t2))/ntest))
par(mfrow=c(1,2))
length(dat$target[-learn])
length(dat$target)
# Plot the results of the best model 
summary(dat)
# Plot the truth
plot(data$x,data$y, col=as.character(data$target), main= "Truth")


## Exercise 2

## This is Doppler's function

doppler <- function (x) { sqrt(x*(1-x))*sin(2.1*pi/(x+0.05)) }

## We are going to model this function in the interval (0.2,1)

a <- 0.2
b <- 1

doppler.data <- function (N, a, b) 
{
  x <- runif(N, a, b)
  t <- doppler(x) + rnorm(N, sd=0.1)
  dd <- data.frame(x,t)
  names(dd) <- c("x", "t")
  return (dd)
}

d <- doppler.data (500, a , b)

## The black points are the data, the blue line is the true underlying function

par(mfrow=c(1,1))

plot (d)
curve (doppler(x), a, b, col='blue', add=TRUE)

## 1. Set N = 100 and model the data using nnet(); compute the final test error and plot the learned function against
## the data and the true underlying function
## 2. Repeat for N = 200, 500, 1000 and 2000 (I suggest you program a function with N as a parameter)
## 3. Compare the results:
##      a) Are there differences in the models for different N? (size, decay)
##      b) Are there differences in prediction performance for different N? In other words, do the models 
##         get better, worse, or indifferent?


####################################################################
## Exercise 3

## This exercise involves the use of the South African Coronary Heart Disease (CHD) data set
## The task is to predict CHD using age at onset, current alcohol consumption, obesity levels, 
## cumulative tobacco, type-A behavior, and low density lipoprotein cholesterol as predictors

set.seed (3)

SAheart <- read.csv("MDSAheart.data")

## First some necessary pre-processing (this part should be lenghtier as you know but let's go to the point)

SAheart <- subset(SAheart, select=-row.names)
SAheart[,"famhist"] <- as.factor(SAheart[,"famhist"])
SAheart[,"chd"] <- factor(SAheart[,"chd"], labels=c("No","Yes"))

## There are no missing values; alcohol, ldl and tobacco have very skewed distributions, so we log transform them

SAheart[,"tobacco"] <- log(SAheart[,"tobacco"]+0.01)
SAheart[,"alcohol"] <- log(SAheart[,"alcohol"]+0.01)
SAheart[,"ldl"] <- log(SAheart[,"ldl"])

## We now standardize all continuous predictors

SAheart[,-c(5,10)] <- scale(subset(SAheart, select=-c(5,10)))

## So this is your departing data set

summary(SAheart) 

## Do the following (in separate sections)

# 1. Create learning and test sets, selecting randomly 2/3 and 1/3 of the data
# 2. Fit a MLP model using only the learn data with Coronary Heart Disease (chd) as the target
#    using 10x10 CV
# 3. Calculate the misclassification rate for your model in the learn data and in the leftout data (test data)
#    If you finished the previous assignment with this dataset (L3), you can now compare the obtained results

# Note this is a very challenging data set

# Your code starts here ...

